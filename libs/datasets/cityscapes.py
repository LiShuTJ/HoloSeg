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
from torch.utils.data import Dataset

import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class cityscapes(Dataset):
    def __init__(self,
                 data_root,
                 list_path,
                 image_size=[512, 1024],
                 crop_size=[512, 1024],
                 test_size=[512, 1024],
                 num_classes=19,
                 multiscale=[0.75, 1.5],
                 flip=True,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 ignore_label=255,
                 mode='train'
                 ):
        super().__init__()
        
        self.data_root = data_root
        self.list_path = list_path
        self.image_size = image_size
        self.crop_size = crop_size
        self.test_size = test_size
        self.num_classes = num_classes
        self.multiscale = multiscale
        self.flip = flip
        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label
        self.mode = mode
        
        self.img_list = [line.strip().split() for line in open(list_path)]
        self.files = self.collect_files()
        
        self.remapping = {-1: ignore_label, 0: ignore_label,
                             1: ignore_label, 2: ignore_label,
                             3: ignore_label, 4: ignore_label,
                             5: ignore_label, 6: ignore_label,
                             7: 0, 8: 1, 9: ignore_label,
                             10: ignore_label, 11: 2, 12: 3,
                             13: 4, 14: ignore_label, 15: ignore_label,
                             16: ignore_label, 17: 5, 18: ignore_label,
                             19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                             25: 12, 26: 13, 27: 14, 28: 15,
                             29: ignore_label, 30: ignore_label,
                             31: 16, 32: 17, 33: 18}
        
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
                                       1.0166, 0.9969, 0.9754, 1.0489,
                                       0.8786, 1.0023, 0.9539, 0.9843,
                                       1.1116, 0.9037, 1.0865, 1.0955,
                                       1.0865, 1.1529, 1.0507]).cuda()
        
        self.train_transform = A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.RandomScale([0.75, 1.5], p=0.5),
                A.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255, p=1),
                A.RandomCrop(crop_size[0], crop_size[1], p=1),
                A.HorizontalFlip(p=0.5),           
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
            )
        
        self.test_transform = A.Compose(
            [
                A.Resize(test_size[0], test_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
            )
    
    def collect_files(self):
        files = []
        if self.mode in ['train', 'val']:
            for item  in self.img_list:
                im, lb = item
                name = os.path.splitext(os.path.basename(lb))[0]
                files.append({
                    "im": im,
                    "lb": lb,
                    "name": name})
        elif self.mode in ['test']:
            for item  in self.img_list:
                im = item
                name = os.path.splitext(os.path.basename(im))[0]
                files.append({
                    "im": im,
                    "name": name})
        else:
            raise NotImplementedError()
        print("[INFO]:", len(files), "files have been collected for", self.mode)
        return files
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.remapping.items():
                label[temp == k] = v
        else:
            for k, v in self.remapping.items():
                label[temp == k] = v
        return label
    
    
    def __getitem__(self, idx):
        item = self.files[idx]
        img = cv2.imread(os.path.join(self.data_root, item["im"]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.mode in ['test']:
            transformed = self.test_transform(image=img, mask=None)
            img = transformed['image']
            return img

        lb = cv2.imread(os.path.join(self.data_root, item["lb"]), cv2.IMREAD_GRAYSCALE)
        lb = self.convert_label(lb)
        lb = lb.astype(np.float32)
        if self.mode in ['train']:
            transformed = self.train_transform(image=img, mask=lb)
        else:
            transformed = self.test_transform(image=img, mask=lb)
        img = transformed["image"]
        lb = transformed["mask"]
        return img, lb
    
    def __len__(self):
        return len(self.files)
            
        
        
