import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import sunrgbd_utils as utils

class sunrgbd_object(object):
    ''' Load and parse object data '''
    def __init__(self, root_dir, split='training'):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 10335
        elif split == 'testing':
            self.num_samples = 2860
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.pc_dir = os.path.join(self.split_dir, 'pc')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        self.label_dir = os.path.join(self.split_dir, 'label')
        # self.label_dimension_dir = os.path.join(self.split_dir, 'label_dimension')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        return utils.load_image(img_filename)

    def get_pointcloud(self, idx):
        depth_filename = os.path.join(self.pc_dir, '%06d.mat'%(idx))
        return utils.load_depth_points(depth_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return utils.SUNRGBD_Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert(self.split=='training')
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return utils.read_sunrgbd_label(label_filename)
