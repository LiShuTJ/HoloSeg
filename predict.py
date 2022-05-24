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
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from argparse import ArgumentParser
import numpy as np
import os
import cv2

from libs.config.config import parse_config
from libs.model.icra import HoloSeg

# define color palette
palette = np.asarray([
        (128, 64, 128),     (244, 35, 232),     (70, 70, 70),   (102, 102, 156),
        (190, 153, 153),    (153, 153, 153),    (250, 170, 30), (220, 220, 0),
        (107, 142, 35),     (152, 251, 152),    (70, 130, 180), (220, 20, 60),
        (255, 0, 0),        (0, 0, 142),        (0, 0, 70), (0, 60, 100),
        (0, 80, 100),       (0, 0, 230),        (119, 11, 32)
], dtype=np.uint8)


def get_args():
    parser = ArgumentParser(description="This is the evaluation code for ICRA 2022 paper: HoloSeg")

    parser.add_argument(
           '--cfg', 
            help='config file', 
            type=str, 
            default='config/holoseg_cityscapes.yaml')
    parser.add_argument(
            "--image",
            type=str,
            default = './images/example0.png',
            help=("Path to image."))
    parser.add_argument(
            "--weights",
            type=str,
            default = './weights/holoseg_cityscapes.pth',
            help=("Path to weights."))
    parser.add_argument(
            "--cpu",
            dest = 'gpu',
            action = 'store_false',
            help=("CPU only."))
    parser.add_argument(
            "--save_path",
            type=str,
            default = 'Predictions.png',
            help=("Path to save predictions."))

    return parser.parse_args()

if __name__ == '__main__':
    # get args and configs
    args = get_args()
    config = parse_config(args)
    # check existence
    assert os.path.exists(args.image), "Image \"{0}\" doesn't exist.".format(args.image)
    assert os.path.exists(args.weights), "Trained weights \"{0}\" doesn't exist.".format(args.weights)

    # define image transform
    pred_transform = A.Compose(
    [
        A.Resize(config.test_image_size[0], config.test_image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # read an input
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    img = image[:,:,::-1]
    im_lb = pred_transform(image=img)
    img = im_lb['image'].unsqueeze(0)
    
    # define model
    model = HoloSeg(config.num_classes)
    
    # fit cuda settings
    map_location = 'cpu'
    if args.gpu:
        image = image.cuda()
        model = model.cuda()
        map_location = 'cuda:0'

    # get prediction
    ori_size = img.shape[2:]
    model.load_weights(args.weights, map_location)
    model = model.eval()
    output = model(img)
    pred = F.interpolate(output, size=ori_size, mode='bilinear', align_corners=False)
    pred = pred.argmax(1)[0].numpy()
    pred = palette[pred]
    pred = pred[:,:,::-1]

    # save and visualize
    image = cv2.resize(image,(config.test_image_size[1], config.test_image_size[0]))
    integ = cv2.addWeighted(image, 0.5, pred, 0.5, 0)
    cv2.imwrite(args.save_path, integ)
    cv2.imshow("Predictions", integ)
    cv2.waitKey()
    cv2.destroyAllWindows()
