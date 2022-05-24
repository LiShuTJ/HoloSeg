# -*- coding: utf-8 -*-

##################################################
# Title: HoloSeg
# Author: S. Li, Q. Yan, C. Liu, Q. Chen
# Robotics and Artificial Intelligence Lab (RAIL)
# Tongji University
##################################################

'''
 This is the code for ICRA2022 paper:
     HoloSeg: An Efficient Holographic Segmentation 
         Network for Real-time Scene Parsing
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os

BN_MOMENTUM = 0.01

class SToD(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        x = rearrange(x, 'b c (h h2) (w w2) -> b (h2 w2 c) h w', h2=self.bs, w2=self.bs)
        return x
    

class DToS(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        x = rearrange(x, 'b (h2 w2 c) h w -> b c (h h2) (w w2)', h2=self.bs, w2=self.bs)
        return x
    

class DPL(nn.Module):
    def __init__(self, chann,  dilation=[1,1,1,1]):
        super().__init__()
        inter = chann // 4
        self.inter=inter
        self.pre = convbnrelu(chann, chann, 3)
        self.group1 = nn.Conv2d(inter, inter, 3, padding=dilation[0], bias=False, dilation=dilation[0])
        self.group2 = nn.Conv2d(inter, inter, 3, padding=dilation[1], bias=False, dilation=dilation[1]) 
        self.group3 = nn.Conv2d(inter, inter, 3, padding=dilation[2], bias=False, dilation=dilation[2])
        self.group4 = nn.Conv2d(inter, inter, 3, padding=dilation[3], bias=False, dilation=dilation[3])
        self.bn = nn.BatchNorm2d(chann, momentum=BN_MOMENTUM)
        
    def forward(self, x):
        inp = x
        x = self.pre(x)
        g1,g2,g3,g4 = torch.chunk(x, 4, dim=1)
        g1 = self.group1(g1)
        g2 = self.group2(g2)
        g3 = self.group3(g3)
        g4 = self.group4(g4)
        x = torch.cat([g1, g2, g3, g4], 1)
        x = self.bn(x)
        return F.relu(inp+x, True) 

    
class convbnrelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, bias=False, dilated=1, groups=1, relu=True):
        super().__init__()
        pad = int((kernel-1)//2) * dilated
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride=stride, dilation=dilated, padding=pad, bias=bias, groups=groups),
                nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM))
        self.relu = relu

    def forward(self, x):
        if self.relu:
            return F.relu(self.conv(x), True)
        else:
            return self.conv(x)


class resBlock_standard(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.c = nn.Sequential(
                convbnrelu(in_ch, out_ch, 3, stride=stride),
                convbnrelu(out_ch, out_ch, 3, relu=False)
                )
        self.sc = nn.Identity() if stride==1 and in_ch==out_ch else convbnrelu(in_ch, out_ch, 1, stride=stride, relu=False)
    
    def forward(self, x):
        return F.relu(self.sc(x)+self.c(x), True)
    

class LSP_D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, do=None):
        super().__init__()

        self.to_qkv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch*3, 1, bias=True))
        self.downsample = SToD(2)
        self.lo = convbnrelu(4*in_ch, out_ch, k, groups=4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.do = nn.Sequential()
        if do is not None:
            self.do = nn.Dropout2d(do)
        
    def forward(self, x):
        x = self.downsample(x)
        x = self.lo(x)
        pooled = self.pool(x)

        qkv = self.to_qkv(pooled).chunk(3, dim = 1)
        
        q, k, v = map(lambda t: t.squeeze(2), qkv)
        attn = torch.bmm(q, k.permute(0, 2, 1))
        attn = torch.softmax(attn, dim=2)
        attn = torch.bmm(attn, v)
        attn = attn.unsqueeze(2)
        x = x+attn
        
        x = self.do(x)
       
        return x
    

class LSP_U(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, do=None):
        super().__init__()
        self.to_qkv = nn.Sequential(
            nn.Conv2d(4*out_ch, out_ch*12, 1, bias=True))
        self.upsample = DToS(2)
        self.lo = convbnrelu(in_ch, 4*out_ch, k, groups=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.do = nn.Sequential()
        if do is not None:
            self.do = nn.Dropout2d(do)
        
    def forward(self, x):
        x = self.lo(x)
        pooled = self.pool(x)
        qkv = self.to_qkv(pooled).chunk(3, dim = 1)
        q, k, v = map(lambda t: t.squeeze(2), qkv)
        attn = torch.bmm(q, k.permute(0, 2, 1))
        attn = torch.softmax(attn, dim=2)
        attn = torch.bmm(attn, v)
        attn = attn.unsqueeze(2)
        x = x+attn
        x = self.upsample(x)
        x = self.do(x)
       
        return x  
    

class HoloSeg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = convbnrelu(3, 64, 7, stride=2)
        
        self.conv2 = nn.Sequential(
                    resBlock_standard(64, 64, 2),
                    resBlock_standard(64, 64),
                    )
        
        self.conv3 = nn.Sequential(
                    resBlock_standard(64, 128, 2),
                    resBlock_standard(128, 128),
                    )
        self.conv4 = nn.Sequential(
                    LSP_D(128, 256, 1),
                    DPL(256, [1,2,5,9]),
                    DPL(256, [1,2,5,9]),
                    )
        self.conv5 = nn.Sequential(
                    LSP_D(256, 512, 1),
                    DPL(512, [1,2,5,9]),
                    DPL(512, [1,2,5,9]),
                    )
        
        # the RFR
        self.t8_16 = nn.Sequential(
                    LSP_D(128, 256, 1)
                    )
        
        self.t16_8 = nn.Sequential(
                    LSP_U(256, 128, 1)
                    )
        
        self.t8_32 = nn.Sequential(
                    LSP_D(128, 256, 1),
                    LSP_D(256, 512, 1)
                    )
        self.t16_32 = nn.Sequential(
                    LSP_D(256, 512, 1)
                    )
        self.t32_16 = LSP_U(512, 256, 1)
        self.t32_16_8 =  nn.Sequential(
                    LSP_U(256, 128, 1),
                    )

        # the seghead
        self.pred8 = convbnrelu(128, num_classes, 1)
        self.pred16 = nn.Sequential(
                        convbnrelu(256, num_classes*4, 1),
                        DToS(2)
                        )
        self.pred32 = nn.Sequential(
                        convbnrelu(512, num_classes*4*4, 1),
                        DToS(2),
                        DToS(2)
                        )

        self.score = nn.Conv2d(num_classes*3, num_classes, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        curr_8 = x
        x = self.conv4(x)
        curr_16 = x
        x = x+self.t8_16(curr_8)
        curr_8 = curr_8+self.t16_8(curr_16)
        x = self.conv5(x)
        curr_32 = x+self.t8_32(curr_8)+self.t16_32(curr_16)
        curr_16 = curr_16+self.t32_16(x)
        curr_8 = curr_8+self.t32_16_8(curr_16)
        pred32 = self.pred32(curr_32)
        pred16 = self.pred16(curr_16)
        pred8 = self.pred8(curr_8)
        score = self.score(torch.cat([pred8, pred16, pred32], 1))
        return score  

    def load_weights(self, path, location='cpu'):
        if os.path.isfile(path):
            pretrained_dict = torch.load(path, map_location=location)
            print("[INFO] LOADING PRETRAINED MODEL: ", path)
            pretrained_dict = {k.replace('model.', ''): v for k, v in pretrained_dict.items()}
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                 print('=> loading {} pretrained model {}'.format(k, path))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
    
