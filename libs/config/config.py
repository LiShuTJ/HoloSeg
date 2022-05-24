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

import yaml

class configuration():
    def __init__(self, args):
        super().__init__()
        config = open(args.cfg, encoding='utf-8')
        config = yaml.load(config, Loader=yaml.SafeLoader)
    
        # parse 'settings'
        self.apex = config['SETTINGS']['APEX']
        self.gpus = config['SETTINGS']['GPUS']
        self.workers = config['SETTINGS']['WORKERS']
        self.output_dir = config['SETTINGS']['OUTPUT_DIR']
        self.print_freq = config['SETTINGS']['PRINT_FREQ']
        self.val_freq = config['SETTINGS']['VAL_FREQ']
        
        # parse 'model'
        self.model_name = config['MODEL']['NAME']
        self.num_classes = config['MODEL']['NUM_CLASSES']
        self.pretrained = config['MODEL']['PRETRAINED']
        
        # parse 'dataset'
        self.dataset_name = config['DATASET']['NAME']
        self.dataset_root = config['DATASET']['ROOT_DIR']
        self.train_set = config['DATASET']['TRAIN_SET']
        self.test_set = config['DATASET']['TEST_SET']
        
        # parse 'train'
        self.train_image_size = config['TRAIN']['IMAGE_SIZE']
        self.train_crop_size = config['TRAIN']['CROP_SIZE']
        self.train_batch_size = config['TRAIN']['BATCH_SIZE']
        self.train_epochs = config['TRAIN']['EPOCHS']
        self.train_lr = config['TRAIN']['LR']
        self.train_wd = config['TRAIN']['WD']
        self.aug_multiscale = config['TRAIN']['DATA_AUG']['MULTISCALE']
        self.aug_flip = config['TRAIN']['DATA_AUG']['FLIP']
        self.sync_bn = config['TRAIN']['SYNC_BN']
        self.resume = config['TRAIN']['RESUME']
        
        # parse 'test'
        self.test_image_size = config['TEST']['IMAGE_SIZE']
        self.test_batch_size = config['TEST']['BATCH_SIZE']

def parse_config(args):
    cfg = configuration(args)
    return cfg
