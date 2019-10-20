import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class contraction_path(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.contract = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.contract(x)
        return x

class expansion_path(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.scale_factor = 2
        #self.expansion = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.expansion =  nn.ConvTranspose2d(in_ch , out_ch, 2, stride=2)
        
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        #x = F.interpolate(x, scale_factor = self.scale_factor, mode='bilinear', align_corners=True) 
        # x = self.expansion(x)
        x1 = self.expansion(x1)
        x = torch.cat([x2, x1], 1)
        x = self.conv(x)
        
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size = 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class UNet(nn.Module):
    def __init__(self, in_channels = 3, n_classes = 15):
        super().__init__()
        self.inc = double_conv(in_channels, 64)
        self.down1 = contraction_path(64, 128)
        self.down2 = contraction_path(128, 256)
        self.down3 = contraction_path(256, 512)
        self.middle = contraction_path(512, 1024)
        self.up4 = expansion_path(1024, 512)
        self.up3 = expansion_path(512, 256)
        self.up2 = expansion_path(256, 128)
        self.up1 = expansion_path(128, 64)
        #self.up4 = double_conv(512 + 1024, 512)
        #self.up3 = double_conv(256 + 512, 256)
        #self.up2 = double_conv(128 + 256, 128)
        #self.up1 = double_conv(64 + 128, 64)
        self.out = outconv(64, n_classes)
    
    
    def forward(self, x):
        down1 = self.inc(x)
        down2 = self.down1(down1)
        down3 = self.down2(down2)
        down4 = self.down3(down3)
        
        middle = self.middle(down4)
        
        out = self.up4(middle, down4)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)
        
        #out = F.upsample(middle, scale_factor=2)
        #out = torch.cat([down4, out], 1)
        #out = self.up4(out)
        
        #out = F.upsample(out, scale_factor=2)
        #out = torch.cat([down3, out], 1)
        #out = self.up3(out)
        
        #out = F.upsample(out, scale_factor=2)
        #out = torch.cat([down2, out], 1)
        #out = self.up2(out)
        
        #out = F.upsample(out, scale_factor=2)
        #out = torch.cat([down1, out], 1)
        #out = self.up1(out)
        
        x = self.out(out)
        
        return x