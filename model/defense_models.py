import torch
from torch import nn
import torch.nn.functional as F

class autoencoder(nn.Module):
    def __init__(self, in_channel = 1):
        super(autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 3, 3, padding = 1)
        self.conv2 = nn.Conv2d(3, 3, 3, padding = 1)
        self.conv3 = nn.Conv2d(3, 3, 3, padding = 1)
        self.conv4 = nn.Conv2d(3, 3, 3, padding = 1)
        self.conv5 = nn.Conv2d(3, in_channel, 3, padding = 1)
        
        self.avg = nn.AvgPool2d(2)
        
        self.up = nn.Upsample(scale_factor = 2)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = self.sig(self.conv1(x))
        x = self.avg(x)
        x = self.sig(self.conv2(x))
        x = self.sig(self.conv3(x))
        x = self.up(x)
        x = self.sig(self.conv4(x))
        x = self.sig(self.conv5(x))
        return x
    
class autoencoder2(nn.Module):
    def __init__(self, in_channel):
        super(autoencoder2, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 3, 3, padding = 1)
        self.conv2 = nn.Conv2d(3, 3, 3, padding = 1)
        self.conv3 = nn.Conv2d(3, in_channel, 3, padding = 1)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = self.sig(self.conv1(x))
        x = self.sig(self.conv2(x))
        x = self.sig(self.conv3(x))
        return x