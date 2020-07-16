import math
import time
import pickle
import sys
import os
import numpy as np
import logging

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from configs.config import cfg
from datasets.dataset_info import DATASET_INFO
from datasets.dataset_info import SUNRGBDCategory

# from datasets.data_utils import rotate_pc_along_y, compute_box_3d
from datasets.data_utils import rotate_pc_along_y, compute_box_3d, extract_pc_in_box3d, roty

logger = logging.getLogger(__name__)


def project_image_to_camera(uv_depth, K):
    n = uv_depth.shape[0]

    c_u, c_v = K[0, 2], K[1, 2]
    f_u, f_v = K[0, 0], K[1, 1]

    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v
    pts_3d_camera = np.zeros((n, 3))
    pts_3d_camera[:, 0] = x
    pts_3d_camera[:, 1] = y
    pts_3d_camera[:, 2] = uv_depth[:, 2]
    return pts_3d_camera


def project_image_to_upright_camera(uv_depth, K, Rtilt):

    pts_3d_camera = project_image_to_camera(uv_depth, K)

    # X, Y, Z -> X, Z, -Y
    # pts_3d_depth = np.dot(pts_3d_camera, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    pts_3d_depth = pts_3d_camera[:, [0, 2, 1]] * np.array([1, 1, -1])

    pts_3d_upright_depth = np.transpose(np.dot(Rtilt, np.transpose(pts_3d_depth)))

    # X, Y, Z -> X, -Z, Y
    # pts_3d = np.dot(pts_3d_upright_depth, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    pts_3d = pts_3d_upright_depth[:, [0, 2, 1]] * np.array([1, -1, 1])

    return pts_3d


class ProviderDataset(Dataset):

    def __init__(self, npoints, split,
                 random_flip=False, random_shift=False, one_hot=True,
                 from_rgb_detection=False,
                 overwritten_data_path=None,
                 extend_from_det=False):

        super(ProviderDataset, self).__init__()
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.one_hot = one_hot
        self.from_rgb_detection = from_rgb_detection

        root_data = cfg.DATA.DATA_ROOT

        assert split in ['train', 'val']

        if not from_rgb_detection:
            if split == 'train':
                overwritten_data_path = os.path.join(root_data, 'sunrgbd_%s_aug5x.pickle' % split)

            elif split == 'val':
                overwritten_data_path = os.path.join(root_data, 'sunrgbd_%s.pickle' % split)

        with open(overwritten_data_path, 'rb') as f:
            data_dict = pickle.load(f)

        if from_rgb_detection:
            self.id_list = data_dict['id']
            self.input_list = data_dict['input']
            self.box2d_list = data_dict['box2d']
            self.type_list = data_dict['type']
            self.prob_list = data_dict['box2d_prob']
            self.frustum_angle_list = data_dict['frustum_angle']
            self.calib_K_list = data_dict['calib_K']
            self.calib_R_list = data_dict['calib_R']
        else:
            self.id_list = data_dict['id']
            self.box2d_list = data_dict['box2d']
            self.box3d_list = data_dict['box3d']
            self.type_list = data_dict['type']
            self.frustum_angle_list = data_dict['frustum_angle']
            self.calib_K_list = data_dict['calib_K']
            self.calib_R_list = data_dict['calib_R']
            self.input_list = data_dict['input']
            self.label_list = data_dict['label']
            self.heading_list = data_dict['box3d_heading']
            self.size_list = data_dict['box3d_size']

        logger.info('load dataset from {}'.format(overwritten_data_path))

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):

        rotate_to_center = cfg.DATA.RTC
        with_extra_feat = cfg.DATA.WITH_EXTRA_FEAT

        rot_angle = self.get_center_view_rot_angle(index)

        cls_type = self.type_list[index]
        assert cls_type in SUNRGBDCategory.CLASSES, cls_type
        size_class = SUNRGBDCategory.CLASSES.index(cls_type)

        # Compute one hot vector
        if self.one_hot:
            one_hot_vec = np.zeros(len(SUNRGBDCategory.CLASSES))
            one_hot_vec[size_class] = 1

        # Get point cloud
        if rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]

        if not with_extra_feat:
            point_set = point_set[:, :3]

        # Resample
        if self.npoints > 0:
            # choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
            choice = np.random.choice(point_set.shape[0], self.npoints, point_set.shape[0] < self.npoints)

        else:
            # choice = np.random.permutation(len(point_set.shape[0]))
            choice = np.arange(point_set.shape[0])

        point_set = point_set[choice, :]

        box = self.box2d_list[index]
        K = self.calib_K_list[index]
        R = self.calib_R_list[index]

        ref1, ref2, ref3, ref4, ref5 = self.generate_ref(box, K, R)

        if rotate_to_center:
            ref1 = self.get_center_view(ref1, index)
            ref2 = self.get_center_view(ref2, index)
            ref3 = self.get_center_view(ref3, index)
            ref4 = self.get_center_view(ref4, index)
            ref5 = self.get_center_view(ref5, index)

        if self.from_rgb_detection:

            data_inputs = {
                'point_cloud': torch.FloatTensor(point_set).transpose(1, 0),
                'rot_angle': torch.FloatTensor([rot_angle]),
                'rgb_prob': torch.FloatTensor([self.prob_list[index]]),

                'center_ref1': torch.FloatTensor(ref1).transpose(1, 0),
                'center_ref2': torch.FloatTensor(ref2).transpose(1, 0),
                'center_ref3': torch.FloatTensor(ref3).transpose(1, 0),
                'center_ref4': torch.FloatTensor(ref4).transpose(1, 0),
                'center_ref5': torch.FloatTensor(ref5).transpose(1, 0),

            }

            if not rotate_to_center:
                data_inputs.update({'rot_angle': torch.zeros(1)})

            if self.one_hot:
                data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})

            return data_inputs

        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index].astype(np.int64)
        seg = seg[choice]

        # Get center point of 3D box
        if rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        box3d_size = self.size_list[index]

        # Data Augmentation
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
                ref5[:, 0] *= -1

        if self.random_shift:
            max_depth = cfg.DATA.MAX_DEPTH
            l, w, h = self.size_list[index]
            dist = np.sqrt(np.sum(l ** 2 + w ** 2))
            shift = np.clip(np.random.randn() * dist * 0.2, -0.5 * dist, 0.5 * dist)
            shift = np.clip(shift + box3d_center[2], 0, max_depth) - box3d_center[2]
            point_set[:, 2] += shift
            box3d_center[2] += shift

            height_shift = np.random.random() * 0.4 - 0.2  # randomly shift +-0.2 meters
            point_set[:, 1] += height_shift
            box3d_center[1] += height_shift

        labels_ref2 = self.generate_labels(box3d_center, self.size_list[index], heading_angle, ref2)

        data_inputs = {
            'point_cloud': torch.FloatTensor(point_set).transpose(1, 0),
            'rot_angle': torch.FloatTensor([rot_angle]),

            'center_ref1': torch.FloatTensor(ref1).transpose(1, 0),
            'center_ref2': torch.FloatTensor(ref2).transpose(1, 0),
            'center_ref3': torch.FloatTensor(ref3).transpose(1, 0),
            'center_ref4': torch.FloatTensor(ref4).transpose(1, 0),
            'center_ref5': torch.FloatTensor(ref5).transpose(1, 0),

            'cls_label': torch.LongTensor(labels_ref2.astype(np.int64)),

            'box3d_center': torch.FloatTensor(box3d_center),
            'box3d_heading': torch.FloatTensor([heading_angle]),
            'box3d_size': torch.FloatTensor(box3d_size),
            'size_class': torch.LongTensor([size_class]),
            'seg_label': torch.LongTensor(seg.astype(np.int64))

        }

        if not rotate_to_center:
            data_inputs.update({'rot_angle': torch.zeros(1)})

        if self.one_hot:
            data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})

        return data_inputs

    def generate_labels(self, center, dimension, angle, ref_xyz):
        box_corner1 = compute_box_3d(center, dimension * 0.5, angle)
        box_corner2 = compute_box_3d(center, dimension, angle)

        labels = np.zeros(len(ref_xyz))
        inside1 = extract_pc_in_box3d(ref_xyz, box_corner1)
        inside2 = extract_pc_in_box3d(ref_xyz, box_corner2)

        labels[inside2] = -1
        labels[inside1] = 1

        if inside1.sum() == 0:
            dis = np.sqrt(((ref_xyz - center)**2).sum(1))
            argmin = np.argmin(dis)
            labels[argmin] = 1
        return labels


    def generate_ref(self, box, K, R):

        s1, s2, s3, s4, s5 = cfg.DATA.STRIDE
        max_depth = cfg.DATA.MAX_DEPTH

        z1 = np.arange(0, max_depth, s1) + s1 / 2.
        z2 = np.arange(0, max_depth, s2) + s2 / 2.
        z3 = np.arange(0, max_depth, s3) + s3 / 2.
        z4 = np.arange(0, max_depth, s4) + s4 / 2.
        z5 = np.arange(0, max_depth, s5) + s5 / 2.

        cx, cy = (box[0] + box[2]) / 2., (box[1] + box[3]) / 2.,

        xyz1 = np.zeros((len(z1), 3))
        xyz1[:, 0] = cx
        xyz1[:, 1] = cy
        xyz1[:, 2] = z1
        xyz1_rect = project_image_to_upright_camera(xyz1, K, R)

        xyz2 = np.zeros((len(z2), 3))
        xyz2[:, 0] = cx
        xyz2[:, 1] = cy
        xyz2[:, 2] = z2
        xyz2_rect = project_image_to_upright_camera(xyz2, K, R)

        xyz3 = np.zeros((len(z3), 3))
        xyz3[:, 0] = cx
        xyz3[:, 1] = cy
        xyz3[:, 2] = z3
        xyz3_rect = project_image_to_upright_camera(xyz3, K, R)

        xyz4 = np.zeros((len(z4), 3))
        xyz4[:, 0] = cx
        xyz4[:, 1] = cy
        xyz4[:, 2] = z4
        xyz4_rect = project_image_to_upright_camera(xyz4, K, R)

        xyz5 = np.zeros((len(z5), 3))
        xyz5[:, 0] = cx
        xyz5[:, 1] = cy
        xyz5[:, 2] = z5
        xyz5_rect = project_image_to_upright_camera(xyz5, K, R)

        return xyz1_rect, xyz2_rect, xyz3_rect, xyz4_rect, xyz5_rect

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] +
                        self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] +
                        self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0),
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view,
                                 self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set,
                                 self.get_center_view_rot_angle(index))

    def get_center_view(self, point_set, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(point_set)
        return rotate_pc_along_y(point_set,
                                 self.get_center_view_rot_angle(index))


def from_prediction_to_label_format(center, angle, size, rot_angle, ref_center=None):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = size
    ry = angle + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()

    if ref_center is not None:
        tx = tx + ref_center[0]
        ty = ty + ref_center[1]
        tz = tz + ref_center[2]

    return tx, ty, tz, l, w, h, ry


def collate_fn(batch):
    return default_collate(batch)


if __name__ == '__main__':

    cfg.DATA.DATA_ROOT = './sunrgbd/data/pickle_data'
    cfg.DATA.RTC = True
    cfg.DATA.STRIDE = (0.05, 0.1, 0.2, 0.4, 0.8)
    dataset = ProviderDataset(1024, split='val', random_flip=False, one_hot=True, random_shift=False)

    for i in range(len(dataset)):
        data = dataset[i]

        for name, value in data.items():
            print(name, value.shape)

        input()


