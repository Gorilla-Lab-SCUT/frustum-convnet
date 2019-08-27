''' Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017

Modified by Zhixin Wang
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import pickle
import sys
import os

import numpy as np
import torch
import logging
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from configs.config import cfg

from datasets.data_utils import rotate_pc_along_y, extract_pc_in_box3d, compute_box_3d, roty, project_image_to_rect
from datasets.dataset_info import KITTICategory

logger = logging.getLogger(__name__)

class ProviderDataset(Dataset):

    def __init__(self, npoints, split,
                 random_flip=False, random_shift=False, 
                 one_hot=True,
                 from_rgb_detection=False,
                 overwritten_data_path='',
                 extend_from_det=False):

        super(ProviderDataset, self).__init__()
        self.npoints = npoints
        self.split = split
        self.random_flip = random_flip
        self.random_shift = random_shift

        self.one_hot = one_hot
        self.from_rgb_detection = from_rgb_detection

        root_data = cfg.DATA.DATA_ROOT
        car_only = cfg.DATA.CAR_ONLY
        people_only = cfg.DATA.PEOPLE_ONLY

        if not overwritten_data_path:
            if not from_rgb_detection:
                if split == 'val':
                    split += '_det'
                    
                if car_only:
                    overwritten_data_path = os.path.join(root_data, 'frustum_caronly_%s.pickle' % (split))
                elif people_only:
                    overwritten_data_path = os.path.join(root_data, 'frustum_pedcyc_%s.pickle' % (split))
                else:
                    overwritten_data_path = os.path.join(root_data, 'frustum_carpedcyc_%s.pickle' % (split))
            else:
                if car_only:
                    overwritten_data_path = os.path.join(root_data,
                                                         'frustum_caronly_%s_rgb_detection_refine.pickle' % (split))
                elif people_only:
                    overwritten_data_path = os.path.join(root_data, 'frustum_pedcyc_%s_rgb_detection_refine.pickle' % (split))
                else:
                    overwritten_data_path = os.path.join(
                        root_data, 'frustum_carpedcyc_%s_rgb_detection_refine.pickle' % (split))

        if from_rgb_detection:
            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)
                self.prob_list = pickle.load(fp)
                # self.image_coord_list = pickle.load(fp)

                self.calib_list = pickle.load(fp)

                self.pred_box3d_list = pickle.load(fp)
                self.pred_box3d_size_list = pickle.load(fp)
                self.pred_box3d_angle_list = pickle.load(fp)
        else:
            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box3d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.label_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.heading_list = pickle.load(fp)
                self.size_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.calib_list = pickle.load(fp)

                self.pred_box3d_list = pickle.load(fp)
                self.pred_box3d_size_list = pickle.load(fp)
                self.pred_box3d_angle_list = pickle.load(fp)

            if extend_from_det:
                extend_det_file = overwritten_data_path.replace('.', '_det.')
                assert os.path.exists(extend_det_file), extend_det_file
                with open(extend_det_file, 'rb') as fp:
                    # extend
                    self.id_list.extend(pickle.load(fp))
                    self.box3d_list.extend(pickle.load(fp))
                    self.input_list.extend(pickle.load(fp))
                    self.label_list.extend(pickle.load(fp))
                    self.type_list.extend(pickle.load(fp))
                    self.heading_list.extend(pickle.load(fp))
                    self.size_list.extend(pickle.load(fp))
                    self.frustum_angle_list.extend(pickle.load(fp))
                    self.box2d_list.extend(pickle.load(fp))
                    self.calib_list.extend(pickle.load(fp))
                    self.pred_box3d_list.extend(pickle.load(fp))
                    self.pred_box3d_size_list.extend(pickle.load(fp))
                    self.pred_box3d_angle_list.extend(pickle.load(fp))
                logger.info('load dataset from {}'.format(extend_det_file))

        logger.info('load dataset from {}'.format(overwritten_data_path))

    def get_center_view_box3d(self, box3d_center, box3d_angle, ref_center, ref_angle):

        box3d_center = box3d_center - ref_center
        box3d_angle = box3d_angle - ref_angle
        box3d_center = rotate_pc_along_y(box3d_center[np.newaxis, :], ref_angle).squeeze(0)

        return box3d_center, box3d_angle

    def get_center_view_point(self, point, ref_center, ref_angle):

        point = point - ref_center
        point = rotate_pc_along_y(point, ref_angle)

        return point

    def normalize_input(self, pc, pred_box_center, pred_box_angle, gt_box_center=None, gt_box_angle=None):
        # translate center to reference point
        # rotate coordinate to reference angle
        ref_center = pred_box_center
        rot_angle = pred_box_angle

        pc = pc - ref_center
        pc = rotate_pc_along_y(pc, rot_angle)
        if gt_box_center is not None and gt_box_angle is not None:
            gt_box_center = gt_box_center - ref_center
            gt_box_angle = gt_box_angle - pred_box_angle
            gt_box_center = rotate_pc_along_y(gt_box_center[np.newaxis, :], rot_angle).squeeze(0)

        return pc, gt_box_center, gt_box_angle

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):

        rotate_to_center = cfg.DATA.RTC
        with_extra_feat = cfg.DATA.WITH_EXTRA_FEAT

        point_set = self.input_list[index].copy()
        pred_box3d = self.pred_box3d_list[index].copy()

        pred_box3d_center = (pred_box3d[0, :] + pred_box3d[6, :]) / 2
        pred_box3d_angle = self.pred_box3d_angle_list[index]
        pred_box3d_size = self.pred_box3d_size_list[index].copy()

        cls_type = self.type_list[index]
        assert cls_type in KITTICategory.CLASSES, cls_type
        size_class = KITTICategory.CLASSES.index(cls_type)

        # Compute one hot vector
        if self.one_hot:
            one_hot_vec = np.zeros((3))
            one_hot_vec[size_class] = 1

        if rotate_to_center:
            point_set[:, :3] = self.get_center_view_point(point_set[:, :3], pred_box3d_center, pred_box3d_angle)

        if not with_extra_feat:
            point_set = point_set[:, :3]

        # Resample
        if self.npoints > 0:
            # choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
            choice = np.random.choice(point_set.shape[0], self.npoints, point_set.shape[0] < self.npoints)

        else:
            choice = np.random.permutation(len(point_set.shape[0]))

        point_set = point_set[choice, :]

        # P = self.calib_list[index]['P2'].reshape(3, 4)

        if rotate_to_center:
            pred_box3d_center_rot, pred_box3d_angle_rot = self.get_center_view_box3d(
                pred_box3d_center, pred_box3d_angle, pred_box3d_center, pred_box3d_angle)

        else:
            pred_box3d_center_rot = pred_box3d_center
            pred_box3d_angle_rot = pred_box3d_angle

        pred_box3d = compute_box_3d(pred_box3d_center_rot, pred_box3d_size, pred_box3d_angle_rot)

        ref1, ref2, ref3, ref4 = self.generate_ref(pred_box3d)

        if self.from_rgb_detection:

            data_inputs = {
                'point_cloud': np.transpose(point_set, (1, 0)).astype(np.float32),
                'center_ref1': np.transpose(ref1, (1, 0)).astype(np.float32),
                'center_ref2': np.transpose(ref2, (1, 0)).astype(np.float32),
                'center_ref3': np.transpose(ref3, (1, 0)).astype(np.float32),
                'center_ref4': np.transpose(ref4, (1, 0)).astype(np.float32),

                'rgb_prob': np.array([self.prob_list[index]]).astype(np.float32),
                'rot_angle': np.array([pred_box3d_angle]).astype(np.float32),
                'ref_center': pred_box3d_center.astype(np.float32),
            }
            if not rotate_to_center:
                data_inputs.update({'rot_angle': torch.zeros(1)})
                data_inputs.update({'ref_center': torch.zeros(3)})

            if self.one_hot:
                data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})

            return data_inputs

        # mask_label = self.label_list[index].copy()
        # mask_label = mask_label[choice]

        box3d = self.box3d_list[index].copy()
        heading_angle = self.heading_list[index]
        box3d_size = self.size_list[index].copy()

        box3d_center = (box3d[0, :] + box3d[6, :]) / 2

        if rotate_to_center:
            box3d_center, heading_angle = self.get_center_view_box3d(
                box3d_center, heading_angle, pred_box3d_center, pred_box3d_angle)

        # box3d_rot = compute_box_3d(box3d_center, box3d_size, heading_angle)
        # self.check(point_set, ref1, box3d_rot, pred_box3d)


        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle

                ref1[:, 0] *= -1
                ref2[:, 0] *= -1
                ref3[:, 0] *= -1
                ref4[:, 0] *= -1

        if self.random_shift:
            s1 = cfg.DATA.STRIDE[0]
            l, w, h = self.size_list[index]
            dist = np.sqrt(np.sum(l ** 2 + w ** 2))
            shift = np.clip(np.random.randn() * dist * 0.1, -s1 * 2, 2 * s1)
            point_set[:, 2] += shift
            box3d_center[2] += shift

        # angle_class, angle_residual = angle2class(heading_angle, NUM_HEADING_BIN)

        # self.check(box, P, box3d_center, self.size_list[index], heading_angle)
        labels = self.generate_labels(box3d_center, box3d_size, heading_angle, ref2)

        data_inputs = {
            'point_cloud': np.transpose(point_set, (1, 0)).astype(np.float32),
            'label': labels.astype(np.int64),
            'box3d_center': box3d_center.astype(np.float32),
            'box3d_heading': np.array([heading_angle], dtype=np.float32),
            'box3d_size': box3d_size.astype(np.float32),
            'size_class': np.array([size_class], dtype=np.int64),

            'center_ref1': np.transpose(ref1, (1, 0)).astype(np.float32),
            'center_ref2': np.transpose(ref2, (1, 0)).astype(np.float32),
            'center_ref3': np.transpose(ref3, (1, 0)).astype(np.float32),
            'center_ref4': np.transpose(ref4, (1, 0)).astype(np.float32),

            'rot_angle': np.array([pred_box3d_angle]).astype(np.float32),
            'ref_center': pred_box3d_center.astype(np.float32),
        }

        if not rotate_to_center:
            data_inputs.update({'rot_angle': torch.zeros(1)})

        if self.one_hot:
            data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})

        return data_inputs

    def generate_labels(self, center, dimension, angle, ref_xyz):
        box_corner1 = compute_box_3d(center, dimension * 0.3, angle)
        box_corner2 = compute_box_3d(center, dimension * 0.6, angle)

        labels = np.zeros(len(ref_xyz))
        inside1 = extract_pc_in_box3d(ref_xyz, box_corner1)
        inside2 = extract_pc_in_box3d(ref_xyz, box_corner2)

        labels[inside2] = -1
        labels[inside1] = 1

        if inside1.sum() == 0:
            dis = np.sqrt(((ref_xyz - center) ** 2).sum(1))
            argmin = np.argmin(dis)
            labels[argmin] = 1

        return labels

    def generate_ref(self, pred_box3d):
        # 8*3
        s1, s2, s3, s4 = cfg.DATA.STRIDE

        cz = ((pred_box3d[0, :] + pred_box3d[6, :]) / 2)[2]
        z1 = np.min(pred_box3d[:, 2], 0)
        z2 = np.max(pred_box3d[:, 2], 0)
        front = pred_box3d[:, 2] < cz
        below = pred_box3d[:, 2] > cz

        cxcycz1 = np.mean(pred_box3d[front], 0)
        cxcycz2 = np.mean(pred_box3d[below], 0)

        delta = cxcycz2 - cxcycz1

        cz1 = np.arange(z1, z2, s1) + s1 / 2.
        cz2 = np.arange(z1, z2, s2) + s2 / 2.
        cz3 = np.arange(z1, z2, s3) + s3 / 2.
        cz4 = np.arange(z1, z2, s4) + s4 / 2.

        # line equation
        cx1 = (cz1 - cxcycz1[2]) / delta[2] * (delta[0]) + cxcycz1[0]
        cy1 = (cz1 - cxcycz1[2]) / delta[2] * (delta[1]) + cxcycz1[1]
        xyz1 = np.zeros((len(cz1), 3))
        xyz1[:, 0] = cx1
        xyz1[:, 1] = cy1
        xyz1[:, 2] = cz1

        cx2 = (cz2 - cxcycz1[2]) / delta[2] * (delta[0]) + cxcycz1[0]
        cy2 = (cz2 - cxcycz1[2]) / delta[2] * (delta[1]) + cxcycz1[1]
        xyz2 = np.zeros((len(cz2), 3))
        xyz2[:, 0] = cx2
        xyz2[:, 1] = cy2
        xyz2[:, 2] = cz2

        cx3 = (cz3 - cxcycz1[2]) / delta[2] * (delta[0]) + cxcycz1[0]
        cy3 = (cz3 - cxcycz1[2]) / delta[2] * (delta[1]) + cxcycz1[1]
        xyz3 = np.zeros((len(cz3), 3))
        xyz3[:, 0] = cx3
        xyz3[:, 1] = cy3
        xyz3[:, 2] = cz3

        cx4 = (cz4 - cxcycz1[2]) / delta[2] * (delta[0]) + cxcycz1[0]
        cy4 = (cz4 - cxcycz1[2]) / delta[2] * (delta[1]) + cxcycz1[1]
        xyz4 = np.zeros((len(cz4), 3))
        xyz4[:, 0] = cx4
        xyz4[:, 1] = cy4
        xyz4[:, 2] = cz4

        return xyz1, xyz2, xyz3, xyz4


def collate_fn(batch):
    # TODO improve compatibility
    bs = len(batch)
    names = ['center_ref1', 'center_ref2', 'center_ref3', 'center_ref4', 'label']
    max_l = dict(zip(names, [0] * len(names)))
    for i in range(bs):
        for k in names:
            if k in batch[i]:
                if k == 'label':
                    assert batch[i][k].ndim == 1
                    length = batch[i][k].shape[0]
                else:
                    length = batch[i][k].shape[1]
                if length > max_l[k]:
                    max_l[k] = length

    for i in range(bs):
        for k in names:
            if k in batch[i]:
                if k == 'label':
                    assert batch[i][k].ndim == 1
                    length = batch[i][k].shape[0]
                    if length < max_l[k]:
                        pad = max_l[k] - length
                        batch[i][k] = np.pad(batch[i][k], ((0, pad)), mode='edge')
                else:
                    length = batch[i][k].shape[1]
                    if length < max_l[k]:
                        pad = max_l[k] - length
                        batch[i][k] = np.pad(batch[i][k], ((0, 0), (0, pad)), mode='edge')

    return default_collate(batch)


def from_prediction_to_label_format(center, angle, size, rot_angle, ref_center):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = size
    ry = angle + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()
    tx = tx + ref_center[0]
    ty = ty + ref_center[1]
    tz = tz + ref_center[2]
    ty += h / 2.0
    return h, w, l, tx, ty, tz, ry


def compute_alpha(box_lidar_center, ry):
    # box_lidar_center: (x, y, z) lidar coordinate
    # ry: estimated box orientation
    return -np.arctan2(-box_lidar_center[1], box_lidar_center[0]) + ry


if __name__ == '__main__':

    cfg.DATA.DATA_ROOT = 'kitti/data/pickle_data_refine'
    cfg.DATA.RTC = True
    cfg.DATA.CAR_ONLY = True
    cfg.DATA.PEOPLE_ONLY = False

    dataset = ProviderDataset(512, split='val', random_flip=True, one_hot=True, random_shift=True)
    cfg.DATA.STRIDE = (0.1, 0.2, 0.4, 0.8)

    for i in range(len(dataset)):
        data = dataset[i]

        for name, value in data.items():
            print(name, value.shape)

        input()
    '''
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    tic = time.time()
    for i, data_dict in enumerate(train_loader):
       
        # for key, value in data_dict.items():
        #     print(key, value.shape)

        print(time.time() - tic)
        tic = time.time()

        # input()
    '''
