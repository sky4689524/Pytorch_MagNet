import numpy as np
import torch 
import torch.nn as nn
import os
import sys
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import copy
import pickle

from optparse import OptionParser

from util import make_one_hot
from dataset import SampleDataset
from model import UNet, SegNet, DenseNet, autoencoder,autoencoder2
from loss import dice_score
from worker import AEDetector, SimpleReformer, Classifier, operate, filters

def get_args():
    
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path',type='string',
                      default='data/samples', help='data path')
    parser.add_option('--model_path', dest='model_path',type='string',
                      default='checkpoints/', help='model_path')
    parser.add_option('--reformer_path', dest='reformer_path',type='string',
                      default='checkpoints/', help='reformer_path')
    parser.add_option('--detector_path', dest='detector_path',type='string',
                      default='checkpoints/', help='detector_path')
    parser.add_option('--classes', dest='classes', default=28, type='int',
                      help='number of classes')
    parser.add_option('--channels', dest='channels', default=1, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=256, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=256, type='int',
                      help='image height')
    parser.add_option('--model', dest='model', type='string',
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_option('--reformer', dest='reformer', type='string',
                      help='reformer name(autoencoder1 or autoencoder2)')
    parser.add_option('--detector', dest='detector', type='string',
                      help='detector name(autoencoder1 or autoencoder2)')
    parser.add_option('--gpu', dest='gpu',type='string',
                      default='gpu', help='gpu or cpu')
    parser.add_option('--model_device1', dest='model_device1', default=0, type='int',
                      help='device1 index number')
    parser.add_option('--model_device2', dest='model_device2', default=-1, type='int',
                      help='device2 index number')
    parser.add_option('--model_device3', dest='model_device3', default=-1, type='int',
                      help='device3 index number')
    parser.add_option('--model_device4', dest='model_device4', default=-1, type='int',
                      help='device4 index number')
    parser.add_option('--defense_model_device', dest='defense_model_device', default=0, type='int',
                      help='defense_model_device gpu index number')

    (options, args) = parser.parse_args()
    return options

def test(model, args):
    
    data_path = args.data_path
    n_channels = args.channels
    n_classes = args.classes
    data_width = args.width
    data_height = args.height
    gpu = args.gpu
    
    # Hyper paremter for MagNet
    thresholds = [0.01, 0.05, 0.001, 0.005]

    reformer_model = None
    
    if args.reformer == 'autoencoder1':
        
        reformer_model = autoencoder(n_channels)
    
    elif args.reformer == 'autoencoder2':
        
        reformer_model = autoencoder2(n_channels)
        
    else :
        print("wrong reformer model : must be autoencoder1 or autoencoder2")
        raise SystemExit
     
    print('reformer model')
    summary(reformer_model, input_size=(n_channels, data_height, data_width), device = 'cpu')
   
    detector_model = None
    
    if args.detector == 'autoencoder1':
        
        detector_model = autoencoder(n_channels)
    
    elif args.detector == 'autoencoder2':
        
        detector_model = autoencoder2(n_channels)
        
    else :
        print("wrong detector model : must be autoencoder1 or autoencoder2")
        raise SystemExit
        
    print('detector model')
    summary(detector_model, input_size=(n_channels, data_height, data_width), device = 'cpu')

    # set device configuration
    device_ids = []
    
    if gpu == 'gpu' :
        
        if not torch.cuda.is_available() :
            print("No cuda available")
            raise SystemExit
            
        device = torch.device(args.model_device1)
        device_defense = torch.device(args.defense_model_device)
        
        device_ids.append(args.model_device1)
        
        if args.model_device2 != -1 :
            device_ids.append(args.model_device2)
            
        if args.model_device3 != -1 :
            device_ids.append(args.model_device3)
        
        if args.model_device4 != -1 :
            device_ids.append(args.model_device4)
        
    else :
        device = torch.device("cpu")    
        device_defense = torch.device("cpu")
    
    detector = AEDetector(detector_model, device_defense, args.detector_path, p=2)
    reformer = SimpleReformer(reformer_model, device_defense, args.reformer_path)
    classifier = Classifier(model, device, args.model_path, device_ids)
        
    # set testdataset
        
    test_dataset = SampleDataset(data_path)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=10,
        num_workers=4,
    )
    
    print('test_dataset : {}, test_loader : {}'.format(len(test_dataset), len(test_loader)))
   

    # Defense with MagNet
    print('test start')
    
    for thrs in thresholds :
    
        print('----------------------------------------')

        counter = 0
        avg_score = 0.0
        thrs = torch.tensor(thrs)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):

                inputs = inputs.float()
                labels = labels.to(device).long()
                target = make_one_hot(labels[:,0,:,:], n_classes, device)

                operate_results = operate(reformer, classifier, inputs)

                all_pass, _ = filters(detector, inputs, thrs)

                if len(all_pass) == 0:
                    continue

                filtered_results = operate_results[all_pass]

                pred = filtered_results.to(device).float()

                target = target[all_pass]

                loss = dice_score(pred, target)

                avg_score += loss.data.cpu().numpy()

                # statistics
                counter += 1

                del inputs, labels, pred, target, loss

        if counter:
            avg_score = avg_score / counter
            print('threshold : {:.4f}, avg_score : {:.4f}'.format(thrs, avg_score))

        else :
            print('threshold : {:.4f} , no images pass from filter'.format(thrs))
    
    
if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes
    
    model = None
    
    if args.model == 'UNet':
        model = UNet(in_channels = n_channels, n_classes = n_classes)
    
    elif args.model == 'SegNet':
        model = SegNet(in_channels = n_channels, n_classes = n_classes)
        
    elif args.model == 'DenseNet':
        model = DenseNet(in_channels = n_channels, n_classes = n_classes)
    
    else :
        print("wrong model : must be UNet, SegNet, or DenseNet")
        raise SystemExit
    
    print('segmentation model')
    summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')
     
    test(model, args)