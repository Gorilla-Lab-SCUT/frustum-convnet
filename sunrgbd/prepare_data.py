''' Helper class and functions for loading SUN RGB-D objects

Author: Charles R. Qi
Date: October 2017

Modified by Zhixin Wang
'''

import os
import sys
import numpy as np
import pickle
import argparse
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import sunrgbd_utils as utils
from sunrgbd_object import sunrgbd_object
from sunrgbd_utils import random_shift_box2d, extract_pc_in_box3d


def ravel_hash(coord):
    assert coord.ndim == 2

    coord -= coord.min(0)
    coord_max = coord.max(0) + 1

    keys = np.zeros(coord.shape[0], dtype=np.int64)

    for i in range(coord.shape[1] - 1):
        keys += coord[:, i]
        keys *= coord_max[i + 1]
    keys += coord[:, -1]

    return keys


def down_sample(x, voxel_size=(0.05, )):

    if isinstance(voxel_size, float):
        voxel_size = (voxel_size, )

    if len(voxel_size) == 1:
        voxel_size = voxel_size * 3

    voxel_size = np.array(voxel_size, dtype=np.float32)
    voxel_index = np.floor(x / voxel_size).astype(np.int64, copy=False)
    hash_keys = ravel_hash(voxel_index)
    _, idx = np.unique(hash_keys, return_index=True)

    return idx


def get_box3d_dim_statistics(my_sunrgbd_dir, idx_filename, type_whitelist):
    dataset = sunrgbd_object(my_sunrgbd_dir)
    dimension_list = []
    type_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.classname not in type_whitelist:
                continue
            dimension_list.append(np.array([obj.l, obj.w, obj.h]))
            type_list.append(obj.classname)

    print("number of objects: {} ".format(len(type_list)))
    print("categories:", list(sorted(type_whitelist)))

    # Get average box size for different categories
    for class_type in sorted(set(type_list)):
        cnt = 0
        box3d_list = []
        for i in range(len(dimension_list)):
            if type_list[i] == class_type:
                cnt += 1
                box3d_list.append(dimension_list[i])

        median_box3d = np.median(box3d_list, 0)
        print("\'%s\': np.array([%f,%f,%f])," %
              (class_type, median_box3d[0] * 2, median_box3d[1] * 2, median_box3d[2] * 2))


def read_det_file(det_file):
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    # data_idx, type_list, prob, box2d
    with open(det_file, 'rt') as f:
        for line in f:
            t = line.rstrip().split(" ")
            id_list.append(int(t[0]))
            type_list.append(t[1])
            prob_list.append(float(t[2]))
            box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))

    return id_list, type_list, box2d_list, prob_list


def read_det_pkl_file(det_file):
    classes = [
        '__background__', 'bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk', 'door', 'dresser',
        'garbage_bin', 'lamp', 'monitor', 'night_stand', 'pillow', 'sink', 'sofa', 'table', 'tv', 'toilet'
    ]
    with open(det_file, 'rb') as f:
        dets = pickle.load(f)

    num_classes = len(dets)
    num_images = len(dets[0])
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []

    for i in range(num_images):
        for c in range(1, num_classes):
            det = dets[c][i]
            for j in range(len(det)):
                id_list.append((i + 1))
                type_list.append(classes[c])
                prob_list.append(det[j][4])
                box2d_list.append(det[j][:4])

    return id_list, type_list, box2d_list, prob_list


def extract_frustum_data(sunrgbd_dir,
                         idx_filename,
                         split,
                         output_filename,
                         type_whitelist,
                         perturb_box2d=False,
                         augmentX=1,
                         with_down_sample=False):
    dataset = sunrgbd_object(sunrgbd_dir, split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = []  # int number
    box2d_list = []  # [xmin,ymin,xmax,ymax]
    box3d_list = []  # (8,3) array in upright depth coord
    input_list = []  # channel number = 6, xyz,rgb in upright depth coord
    label_list = []  # 1 for roi object, 0 for clutter
    type_list = []  # string e.g. bed
    heading_list = []  # face of object angle, radius of clockwise angle from positive x axis in upright camera coord
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axis (clockwise)

    img_coord_list = []
    calib_K_list = []
    calib_R_list = []

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)
        objects = dataset.get_label_objects(data_idx)

        pc_upright_depth = dataset.get_pointcloud(data_idx)
        pc_upright_camera = np.zeros_like(pc_upright_depth)
        pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
        pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]

        if with_down_sample:
            idx = down_sample(pc_upright_camera[:, :3], 0.01)
            # print(len(idx), len(pc_upright_camera))
            pc_upright_camera = pc_upright_camera[idx]
            pc_upright_depth = pc_upright_depth[idx]

        # img = dataset.get_image(data_idx)
        # img_height, img_width, img_channel = img.shape
        pc_image_coord, _ = calib.project_upright_depth_to_image(pc_upright_depth)

        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.classname not in type_whitelist:
                continue
            # 2D BOX: Get pts rect backprojected
            box2d = obj.box2d
            for _ in range(augmentX):
                if perturb_box2d:
                    xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                    # print(xmin,ymin,xmax,ymax)
                else:
                    xmin, ymin, xmax, ymax = box2d
                box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & (
                    pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)
                coord_in_box_fov = pc_image_coord[box_fov_inds, :]
                pc_in_box_fov = pc_upright_camera[box_fov_inds, :]
                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                uvdepth = np.zeros((1, 3))
                uvdepth[0, 0:2] = box2d_center
                uvdepth[0, 2] = 20  # some random depth
                box2d_center_upright_camera = calib.project_image_to_upright_camera(uvdepth)
                # print('UVdepth, center in upright camera: ', uvdepth, box2d_center_upright_camera)
                frustum_angle = -1 * np.arctan2(
                    box2d_center_upright_camera[0, 2],
                    box2d_center_upright_camera[0, 0])  # angle as to positive x-axis as in the Zoox paper
                # print('Frustum angle: ', frustum_angle)
                # 3D BOX: Get pts velo in 3d box
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib)
                box3d_pts_3d = calib.project_upright_depth_to_upright_camera(box3d_pts_3d)
                try:
                    _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                except Exception as e:
                    print(e)
                    continue

                label = np.zeros((pc_in_box_fov.shape[0]))
                label[inds] = 1
                box3d_size = np.array([2 * obj.l, 2 * obj.w, 2 * obj.h])
                # Subsample points..
                num_point = pc_in_box_fov.shape[0]
                if num_point > 2048:
                    choice = np.random.choice(pc_in_box_fov.shape[0], 2048, replace=False)
                    coord_in_box_fov = coord_in_box_fov[choice, :]
                    pc_in_box_fov = pc_in_box_fov[choice, :]
                    label = label[choice]
                # Reject object with too few points
                if np.sum(label) < 5:
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin, ymin, xmax, ymax], dtype=np.float32))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_box_fov.astype(np.float32))
                label_list.append(label.astype(np.bool))
                type_list.append(obj.classname)
                heading_list.append(obj.heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)
                img_coord_list.append(coord_in_box_fov.astype(np.float32))
                calib_K_list.append(calib.K)
                calib_R_list.append(calib.Rtilt)

                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]

    print('Average pos ratio: ', pos_cnt / float(all_cnt))
    print('Average npoints: ', float(all_cnt) / len(id_list))

    data_dict = {
        'id': id_list,
        'box2d': box2d_list,
        'box3d': box3d_list,
        'box3d_size': box3d_size_list,
        'box3d_heading': heading_list,
        'type': type_list,
        'input': input_list,
        'frustum_angle': frustum_angle_list,
        'label': label_list,
        'calib_K': calib_K_list,
        'calib_R': calib_R_list,
        # 'image_coord': img_coord_list,
    }

    with open(output_filename, 'wb') as f:
        pickle.dump(data_dict, f, -1)

    print("save in {}".format(output_filename))


def extract_frustum_data_from_rgb_detection(sunrgbd_dir,
                                            det_file,
                                            split,
                                            output_filename,
                                            type_whitelist,
                                            valid_id_list=None,
                                            with_down_sample=False):

    dataset = sunrgbd_object(sunrgbd_dir, split)
    if det_file.split('.')[-1] == 'txt':
        det_id_list, det_type_list, det_box2d_list, det_prob_list = read_det_file(det_file)
    else:
        det_id_list, det_type_list, det_box2d_list, det_prob_list = read_det_pkl_file(det_file)

    cache_id = -1
    cache = None

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    img_coord_list = []
    calib_K_list = []
    calib_R_list = []

    for det_idx in range(len(det_id_list)):
        data_idx = det_id_list[det_idx]
        if valid_id_list is not None and data_idx not in valid_id_list:
            continue

        if det_type_list[det_idx] not in type_whitelist:
            continue

        print('det idx: %d/%d, data idx: %d' % (det_idx, len(det_id_list), data_idx))
        if cache_id != data_idx:
            calib = dataset.get_calibration(data_idx)
            pc_upright_depth = dataset.get_pointcloud(data_idx)
            pc_upright_camera = np.zeros_like(pc_upright_depth)
            pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
            pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]

            if with_down_sample:
                idx = down_sample(pc_upright_camera[:, :3], 0.01)
                # print(len(idx), len(pc_upright_camera))
                pc_upright_camera = pc_upright_camera[idx]
                pc_upright_depth = pc_upright_depth[idx]

            # img = dataset.get_image(data_idx)
            # img_height, img_width, img_channel = img.shape
            pc_image_coord, _ = calib.project_upright_depth_to_image(pc_upright_depth)
            cache = [calib, pc_upright_camera, pc_image_coord]
            cache_id = data_idx
        else:
            calib, pc_upright_camera, pc_image_coord = cache

        # 2D BOX: Get pts rect backprojected
        xmin, ymin, xmax, ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & (
            pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)

        coord_in_box_fov = pc_image_coord[box_fov_inds, :]
        pc_in_box_fov = pc_upright_camera[box_fov_inds, :]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_upright_camera = calib.project_image_to_upright_camera(uvdepth)
        frustum_angle = -1 * np.arctan2(
            box2d_center_upright_camera[0, 2],
            box2d_center_upright_camera[0, 0])  # angle as to positive x-axis as in the Zoox paper
        # Subsample points..
        num_point = pc_in_box_fov.shape[0]
        if num_point > 2048:
            choice = np.random.choice(pc_in_box_fov.shape[0], 2048, replace=False)
            coord_in_box_fov = coord_in_box_fov[choice, :]
            pc_in_box_fov = pc_in_box_fov[choice, :]

        # Pass objects that are too small
        if len(pc_in_box_fov) < 5:
            continue

        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov.astype(np.float32))
        frustum_angle_list.append(frustum_angle)

        img_coord_list.append(coord_in_box_fov.astype(np.float32))
        calib_K_list.append(calib.K)
        calib_R_list.append(calib.Rtilt)

    data_dict = {
        'id': id_list,
        'type': type_list,
        'box2d': box2d_list,
        'box2d_prob': prob_list,
        'input': input_list,
        'frustum_angle': frustum_angle_list,
        'calib_K': calib_K_list,
        'calib_R': calib_R_list,
        # 'image_coord': img_coord_list,
    }

    with open(output_filename, 'wb') as f:
        pickle.dump(data_dict, f, -1)

    print("save in {}".format(output_filename))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_train',
                        action='store_true',
                        help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_val_rgb_detection',
                        action='store_true',
                        help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--num_classes', default=10, type=int, help='19 or 10 categories, default 10')
    parser.add_argument('--save_dir',
                        default='sunrgbd/data/pickle_data',
                        type=str,
                        help='directory to save data, default[sunrgbd/data/pickle_data]')
    parser.add_argument('--gen_avg_dim', action='store_true', help='get average dimension of each class')

    args = parser.parse_args()

    my_sunrgbd_dir = 'sunrgbd/mysunrgbd'  # change if you do not set default path

    if args.num_classes == 10:
        type_whitelist = [
            'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub'
        ]
    elif args.num_classes == 19:
        type_whitelist = [
            'bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk', 'door', 'dresser', 'garbage_bin', 'lamp',
            'monitor', 'night_stand', 'pillow', 'sink', 'sofa', 'table', 'tv', 'toilet'
        ]
    else:
        assert False, 'please set correct num_classes'

    type_whitelist = set(type_whitelist)

    if args.gen_avg_dim:
        get_box3d_dim_statistics(my_sunrgbd_dir, 'sunrgbd/image_sets/train.txt', type_whitelist)

    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.gen_train:
        extract_frustum_data(my_sunrgbd_dir,
                             'sunrgbd/image_sets/train.txt',
                             'training',
                             output_filename=os.path.join(save_dir, 'sunrgbd_train_aug5x.pickle'),
                             type_whitelist=type_whitelist,
                             perturb_box2d=True,
                             augmentX=5,
                             with_down_sample=False)

    if args.gen_val:
        extract_frustum_data(my_sunrgbd_dir,
                             'sunrgbd/image_sets/val.txt',
                             'training',
                             output_filename=os.path.join(save_dir, 'sunrgbd_val.pickle'),
                             type_whitelist=type_whitelist,
                             perturb_box2d=False,
                             augmentX=1,
                             with_down_sample=False)

    if args.gen_val_rgb_detection:
        extract_frustum_data_from_rgb_detection(my_sunrgbd_dir,
                                                './sunrgbd/rgb_detections/sunrgbd_rgb_det_val_classes19_mAP50.2.txt',
                                                'training',
                                                os.path.join(save_dir,'sunrgbd_rgb_det_val.pickle'),
                                                type_whitelist=type_whitelist)
