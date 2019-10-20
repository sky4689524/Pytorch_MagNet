import os
import sys
import torch
import torchvision
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import pickle

from optparse import OptionParser

from dataset import SampleDataset
from model import autoencoder,autoencoder2


def get_args():
    
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path',type='string',
                      default='data/samples', help='data path')
    parser.add_option('--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs')
    parser.add_option('--classes', dest='classes', default=28, type='int',
                      help='number of classes')
    parser.add_option('--channels', dest='channels', default=1, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=256, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=256, type='int',
                      help='image height')
    parser.add_option('--model_number', dest='model_number', default=1, type='int',
                      help='autoencoder number(1 or 2)')
    parser.add_option('--gpu', dest='gpu',type='string',
                      default='gpu', help='gpu or cpu')
    parser.add_option('--device1', dest='device1', default=0, type='int',
                      help='device1 index number')
    parser.add_option('--device2', dest='device2', default=-1, type='int',
                      help='device2 index number')
    parser.add_option('--device3', dest='device3', default=-1, type='int',
                      help='device3 index number')
    parser.add_option('--device4', dest='device4', default=-1, type='int',
                      help='device4 index number')

    (options, args) = parser.parse_args()
    return options

def train_net(model, args):

    data_path = args.data_path
    num_epochs = args.epochs
    gpu = args.gpu
    n_classes = args.classes
    data_width = args.width
    data_height = args.height
    
    # hyper parameter for training
    learning_rate = 1e-3
    v_noise = 0.1
    reg_strength = 1e-9
    
    
    # set device configuration
    device_ids = []
    
    if gpu == 'gpu' :
        
        if not torch.cuda.is_available() :
            print("No cuda available")
            raise SystemExit
            
        device = torch.device(args.device1)
        
        device_ids.append(args.device1)
        
        if args.device2 != -1 :
            device_ids.append(args.device2)
            
        if args.device3 != -1 :
            device_ids.append(args.device3)
        
        if args.device4 != -1 :
            device_ids.append(args.device4)
        
    
    else :
        device = torch.device("cpu")
    
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids = device_ids)
        
    model = model.to(device)

    # set image into training and validation dataset
    
    train_dataset = SampleDataset(data_path)

    print('total image : {}'.format(len(train_dataset)))

    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.2)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=4,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        train_dataset,
        batch_size=30,
        num_workers=4,
        sampler=valid_sampler
    )
    
    model_folder = os.path.abspath('./checkpoints')
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    
    if args.model_number == 1:
        model_path = os.path.join(model_folder, 'autoencoder1.pth')
    
    elif args.model_number == 2:
        model_path = os.path.join(model_folder, 'autoencoder2.pth')
       
    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    display_steps = 10
    best_loss = 1e10
    loss_history = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        model.train()
        epoch_size = 0
        running_loss = 0

        lambda2 = torch.tensor(reg_strength)
        l2_reg = torch.tensor(0.)

        for batch_idx, (data, _) in enumerate(train_loader):

            noise = v_noise * np.random.normal(size=np.shape(data))
            noise = torch.from_numpy(noise)

            noisy_train_data = data.double() + noise

            noisy_train_data.clamp(0.0, 1.0)

            noisy_train_data = noisy_train_data.to(device).float()

            data = data.to(device).float()

            optimizer.zero_grad()

            output = model(noisy_train_data)

            loss = criterion(output, data)

            for param in model.parameters():
                l2_reg += torch.norm(param)

            loss += lambda2 * l2_reg.detach().cpu()

            loss.backward()
            optimizer.step()

            if batch_idx % display_steps == 0:
                print('    ', end='')
                print('batch {:>3}/{:>3}, loss {:.4f}\r'\
                      .format(batch_idx+1, len(train_loader),loss.item()))

        # evalute
        print('Finished epoch {}, starting evaluation'.format(epoch+1))

        model.eval()

        lambda2 = torch.tensor(reg_strength)
        l2_reg = torch.tensor(0.)

        with torch.no_grad():
            for data, _ in val_loader:

                data = data.to(device).float()
                target = data.to(device).float()

                output = model(data)

                loss = criterion(output, target)

                for param in model.parameters():
                    l2_reg += torch.norm(param)

                loss += lambda2 * l2_reg.detach().cpu()

                running_loss += loss.item()

        validate_loss = running_loss / len(val_loader)

        if validate_loss < best_loss:

            print('best validation loss : {:.4f}'.format(validate_loss))

            best_loss = validate_loss

            print("saving best model")

            model_copy = copy.deepcopy(model)
            model_copy = model_copy.cpu()
            model_state_dict = model_copy.state_dict()
            torch.save(model_state_dict, model_path)
        
    return loss_history

if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes
    
    model = None
    
    if args.model_number == 1:
        model = autoencoder(n_channels)
    
    elif args.model_number == 2:
        model = autoencoder2(n_channels)
        
    else :
        print("wrong model number : must be 1 or 2")
        raise SystemExit
    
    print('Training Autoencoder {}'.format(str(args.model_number)))
    
    summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')
        
    loss_history = train_net(model, args)
    
    loss_folder = os.path.abspath('./checkpoints')
    loss_path = 'autoencoder' + str(args.model_number) + '_validation_losses'
    save_loss_path = os.path.join(loss_folder, loss_path)
    
    # save validation loss history
    with open(save_loss_path, 'wb') as fp:
        pickle.dump(loss_history, fp)