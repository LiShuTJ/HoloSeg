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
from torch.nn import functional as F
import torch.distributed as dist
import numpy as np
import os

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, weight=None):
        super().__init__()
        self.ignore_label = ignore_label
        self.cri = nn.CrossEntropyLoss(weight=weight, 
                                       ignore_index=255)
        
    def forward(self, im, lb):
        ih, iw = im.size(2), im.size(3)
        h, w = lb.size(1), lb.size(2)
        if ih != h or iw != w:
            im = F.interpolate(
                    input=im, size=(h, w), mode='bilinear', align_corners=False)

        loss = self.cri(im, lb)

        return loss

class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def get_confusion_matrix(label, pred, size, num_class, ignore=255):
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def reduce_tensor(inp):
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def Sec2HMS(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    hms = '%dh%02dm%02ds' % (h, m, s)
    return hms
