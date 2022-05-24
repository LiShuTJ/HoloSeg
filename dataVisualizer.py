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

import numpy as np
import cv2
import os
import albumentations as A

train_transform = A.Compose(
    [
        A.Resize(512, 1024),
        A.RandomScale(0.5, p=0.5),
        A.PadIfNeeded(512, 1024, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255, p=1),
        A.RandomCrop(512, 1024, p=1),
        A.HorizontalFlip(p=0.5)
    ])

all_file = []
path = './images'
for f in os.listdir(path): 
    f_name = os.path.join(path, f)
    all_file.append(f_name)

key = -1
while key != 27: # ESC 
    idx = np.random.randint(0, len(all_file))
    img = cv2.imread(all_file[idx], cv2.IMREAD_COLOR)
    img = img[:,:,::-1]
    
    transformed = train_transform(image=img)
    img_after = transformed["image"]
    img = cv2.resize(img, (1024, 512))
    img = img[:,:,::-1]
    img_after = img_after[:,:,::-1]
    images = np.concatenate([img, img_after], 0)

    cv2.imshow("images", images)
    key = cv2.waitKey()

cv2.destroyAllWindows()
        