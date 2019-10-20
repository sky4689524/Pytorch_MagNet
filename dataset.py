from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import os.path
import pickle
import torch

class SampleDataset(Dataset):
    
    def __init__(self, root_dir):
    
        self.data_path = root_dir

        self.images = []
        self.labels = []
        
        # data form [images, labels]
        with open (self.data_path, 'rb') as fp:
            data = pickle.load(fp)
        
        for i in range(len(data)):
            
            self.images.append(data[i][0])
            self.labels.append(data[i][1])
             

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        image = self.images[index]
        labels = self.labels[index]
        
        torch_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        image = torch_transform(image)
        labels = torch_transform(labels)
            
        return (image, labels)