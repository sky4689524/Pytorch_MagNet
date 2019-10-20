import torch
import torch.nn as nn
import torch.nn.functional as F


class encode_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3):
        super(encode_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride = 1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class decode_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3):
        super(decode_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride = 1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3):
        super(outconv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride = 1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class SegNet(nn.Module):
    def __init__(self, in_channels = 1, n_classes = 28):
        super().__init__()
        
        self.conv1_1 = encode_conv(in_channels, 64)
        self.conv1_2 = encode_conv(64, 64)
        
        self.conv2_1 = encode_conv(64, 128)
        self.conv2_2 = encode_conv(128, 128)
        
        self.conv3_1 = encode_conv(128, 256)
        self.conv3_2 = encode_conv(256, 256)
        self.conv3_3 = encode_conv(256, 256)
        
        self.conv4_1 = encode_conv(256, 512)
        self.conv4_2 = encode_conv(512, 512)
        self.conv4_3 = encode_conv(512, 512)
       
        self.conv5_1 = encode_conv(512, 512)
        self.conv5_2 = encode_conv(512, 512)
        self.conv5_3 = encode_conv(512, 512)
        
        self.deconv5_3 = decode_conv(512,512)
        self.deconv5_2 = decode_conv(512,512)
        self.deconv5_1 = decode_conv(512,512)
        
        self.deconv4_3 = decode_conv(512,512)
        self.deconv4_2 = decode_conv(512,512)
        self.deconv4_1 = decode_conv(512,256)
        
        self.deconv3_3 = decode_conv(256,256)
        self.deconv3_2 = decode_conv(256,256)
        self.deconv3_1 = decode_conv(256,128)
        
        self.deconv2_2 = decode_conv(128,128)
        self.deconv2_1 = decode_conv(128,64)
        
        self.deconv1_2 = decode_conv(64,64)
        self.deconv1_1 = outconv(64,n_classes)
    
    
    def forward(self, x):
        
        #Encoder
        
        dim0 = x.size()
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x, indices0 =  F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        dim1 = x.size()
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x, indices1 =  F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        dim2 = x.size()
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x, indices2 =  F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        dim3 = x.size()
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x, indices3 =  F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        dim4 = x.size()
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x, indices4 =  F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        # Decoder
        
        x = F.max_unpool2d(x, indices4, kernel_size = 2, stride = 2, output_size = dim4)
        x = self.deconv5_3(x)
        x = self.deconv5_2(x)
        x = self.deconv5_1(x)
        
        x = F.max_unpool2d(x, indices3, kernel_size = 2, stride = 2, output_size = dim3)
        x = self.deconv4_3(x)
        x = self.deconv4_2(x)
        x = self.deconv4_1(x)
        
        x = F.max_unpool2d(x, indices2, kernel_size = 2, stride = 2, output_size = dim2)
        x = self.deconv3_3(x)
        x = self.deconv3_2(x)
        x = self.deconv3_1(x)
        
        x = F.max_unpool2d(x, indices1, kernel_size = 2, stride = 2, output_size = dim1)
        x = self.deconv2_2(x)
        x = self.deconv2_1(x)
        
        x = F.max_unpool2d(x, indices0, kernel_size = 2, stride = 2, output_size = dim0)
        x = self.deconv1_2(x)
        x = self.deconv1_1(x)
        
        return x