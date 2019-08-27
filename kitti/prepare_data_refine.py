import argparse
import copy
import os
import pickle
import sys
import time

import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

import kitti_util as utils
from kitti_object import kitti_object
from draw_util import get_lidar_in_image_fov

from ops.pybind11.rbbox_iou import rbbox_iou_3d
from utils.box_util import box3d_iou


def extract_boxes(objects, type_whitelist, remove_diff=False):
    boxes_2d = []
    boxes_3d = []

    filter_objects = []

    for obj_idx in range(len(objects)):
        obj = objects[obj_idx]
        if obj.type not in type_whitelist:
            continue

        if remove_diff:
            if obj.occlusion > 2 or obj.truncation > 0.5 or obj.ymax - obj.ymin < 25:
                continue

        boxes_2d += [obj.box2d]

        l, w, h = obj.l, obj.w, obj.h
        cx, cy, cz = obj.t
        ry = obj.ry
        cy = cy - h / 2
        boxes_3d += [np.array([cx, cy, cz, l, w, h, ry])]
        filter_objects += [obj]

    if len(boxes_3d) != 0:
        boxes_3d = np.stack(boxes_3d, 0)
        boxes_2d = np.stack(boxes_2d, 0)

    return filter_objects, boxes_2d, boxes_3d


def compute_box_3d_obj_array(obj_array):
    '''
    cx, cy, cz, l, w, h, ry
    '''

    cx, cy, cz, l, w, h, angle = obj_array

    R = utils.roty(angle)

    # 3d bounding box corners

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + cx
    corners_3d[1, :] = corners_3d[1, :] + cy
    corners_3d[2, :] = corners_3d[2, :] + cz
    # print 'cornsers_3d: ', corners_3d

    return np.transpose(corners_3d, (1, 0))


def compute_box_3d_obj(cx, cy, cz, l, w, h, ry):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = utils.roty(ry)

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + cx
    corners_3d[1, :] = corners_3d[1, :] + cy
    corners_3d[2, :] = corners_3d[2, :] + cz

    return np.transpose(corners_3d)


def single_overlap(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    x_h = min(box1[3], box2[3]) - max(box1[1], box2[1])

    if x_w <= 0 or x_h <= 0 or area1 <= 0 or area2 <= 0:
        return 0

    return (x_w * x_h) / (area1 + area2 - (x_w * x_h))


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)
    return pc[box2d_roi_inds, :], box2d_roi_inds


def random_shift_box2d(box2d, img_height, img_width, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio
    xmin, ymin, xmax, ymax = box2d
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    assert xmin < xmax and ymin < ymax

    while True:
        cx2 = cx + w * r * (np.random.random() * 2 - 1)
        cy2 = cy + h * r * (np.random.random() * 2 - 1)
        h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        new_box2d = np.array(
            [cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])

        new_box2d[[0, 2]] = np.clip(new_box2d[[0, 2]], 0, img_width - 1)
        new_box2d[[1, 3]] = np.clip(new_box2d[[1, 3]], 0, img_height - 1)

        if w2 > 0 and h2 > 0:
            return new_box2d


def random_shift_box3d(box3d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio
    xmin, ymin, zmin, xmax, ymax, zmax = box3d

    l = xmax - xmin
    h = ymax - ymin
    w = zmax - zmin

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0

    assert xmin < xmax and ymin < ymax and zmin < zmax

    while True:
        cx2 = cx + l * r * (np.random.random() * 2 - 1)
        cy2 = cy + h * r * (np.random.random() * 2 - 1)
        cz2 = cz + w * r * (np.random.random() * 2 - 1)

        l2 = l * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1

        new_box3d = np.array([cx2 - l2 / 2.0, cy2 - h2 / 2.0, cz2 - w2 / 2.0,
                              cx2 + l2 / 2.0, cy2 + h2 / 2.0, cz2 + w2 / 2.0])

        o = single_overlap(box3d[[0, 2, 3, 5]], new_box3d[[0, 2, 3, 5]])
        if l2 > 0 and h2 > 0 and w2 > 0 and o <= 0.8 and o >= 0.5:
            return new_box3d


def random_shift_rotate_box3d(obj_array, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio

    cx, cy, cz, l, w, h, angle = obj_array
    # [-pi, pi] -> [0, 2pi]
    angle = angle + np.pi

    assert l > 0 and w > 0 and h > 0

    while True:
        l1 = l + l * r * (np.random.random() * 2 - 1)
        h1 = h + h * r * (np.random.random() * 2 - 1)
        w1 = w + w * r * (np.random.random() * 2 - 1)

        cx1 = cx + l * r * (np.random.random() * 2 - 1)
        cy1 = cy + h * r * (np.random.random() * 2 - 1)
        cz1 = cz + w * r * (np.random.random() * 2 - 1)

        angle1 = angle + r * (np.random.random() * 2 - 1) * np.pi

        angle1 = angle1 % (2 * np.pi)

        if l1 > 0 and h1 > 0 and w1 > 0:
            angle1 = angle1 - np.pi
            assert angle1 > (-np.pi - 0.001) and angle1 < (np.pi + 0.001)
            new_box3d = np.array([cx1, cy1, cz1, l1, w1, h1, angle1])
            # new_box3d_corners = compute_box_3d_obj_array(new_box3d)
            # box3d_corners = compute_box_3d_obj_array(obj_array)
            # ious = box3d_iou(box3d_corners, new_box3d_corners)
            # print(ious[0], ious[1])

            return new_box3d


def extract_frustum_data(idx_filename, split, output_filename,
                         perturb_box2d=False, augmentX=1, type_whitelist=['Car'], remove_diff=False):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = []  # int number
    box3d_list = []  # (8,3) array in rect camera coord
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    label_list = []  # 1 for roi object, 0 for clutter
    type_list = []  # string e.g. Car
    heading_list = []  # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    gt_box2d_list = []
    calib_list = []

    enlarge_box3d_list = []
    enlarge_box3d_size_list = []
    enlarge_box3d_angle_list = []

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                 calib, 0, 0, img_width, img_height, True)

        pc_rect = pc_rect[img_fov_inds, :]
        pc_image_coord = pc_image_coord[img_fov_inds]

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist:
                continue

            if remove_diff:
                box2d = objects[obj_idx].box2d
                xmin, ymin, xmax, ymax = box2d
                if objects[obj_idx].occlusion > 2 or objects[obj_idx].truncation > 0.5 or ymax - ymin < 25:
                    continue

            # 2D BOX: Get pts rect backprojected
            box2d = objects[obj_idx].box2d
            obj = objects[obj_idx]

            l, w, h = obj.l, obj.w, obj.h
            cx, cy, cz = obj.t
            ry = obj.ry
            cy = cy - h / 2

            obj_array = np.array([cx, cy, cz, l, w, h, ry])

            box3d_pts_3d = compute_box_3d_obj_array(obj_array)

            ratio = 1.2
            enlarge_obj_array = obj_array.copy()
            enlarge_obj_array[3:6] = enlarge_obj_array[3:6] * ratio

            for _ in range(augmentX):

                if perturb_box2d:
                    # print(box3d_align)

                    enlarge_obj_array = random_shift_rotate_box3d(
                        enlarge_obj_array, 0.05)
                    box3d_corners_enlarge = compute_box_3d_obj_array(
                        enlarge_obj_array)

                else:
                    box3d_corners_enlarge = compute_box_3d_obj_array(
                        enlarge_obj_array)

                _, inds = extract_pc_in_box3d(pc_rect, box3d_corners_enlarge)

                pc_in_cuboid = pc_rect[inds]
                pc_box_image_coord = pc_image_coord[inds]

                _, inds = extract_pc_in_box3d(pc_in_cuboid, box3d_pts_3d)

                label = np.zeros((pc_in_cuboid.shape[0]))
                label[inds] = 1

                _, inds = extract_pc_in_box3d(pc_rect, box3d_pts_3d)

                # print(np.sum(label), np.sum(inds))

                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if np.sum(label) == 0:
                    continue

                box3d_center = enlarge_obj_array[:3]

                frustum_angle = -1 * np.arctan2(box3d_center[2],
                                                box3d_center[0])

                id_list.append(data_idx)
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_cuboid)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)

                gt_box2d_list.append(box2d)
                calib_list.append(calib.calib_dict)
                enlarge_box3d_list.append(box3d_corners_enlarge)
                enlarge_box3d_size_list.append(enlarge_obj_array[3:6])
                enlarge_box3d_angle_list.append(enlarge_obj_array[-1])
                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_cuboid.shape[0]

    print('total_objects %d' % len(id_list))
    print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt) / len(id_list)))

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp, -1)
        pickle.dump(box3d_list, fp, -1)
        pickle.dump(input_list, fp, -1)
        pickle.dump(label_list, fp, -1)
        pickle.dump(type_list, fp, -1)
        pickle.dump(heading_list, fp, -1)
        pickle.dump(box3d_size_list, fp, -1)
        pickle.dump(frustum_angle_list, fp, -1)
        pickle.dump(gt_box2d_list, fp, -1)
        pickle.dump(calib_list, fp, -1)

        pickle.dump(enlarge_box3d_list, fp, -1)
        pickle.dump(enlarge_box3d_size_list, fp, -1)
        pickle.dump(enlarge_box3d_angle_list, fp, -1)

    print('save in {}'.format(output_filename))


def extract_frustum_det_data(idx_filename, split, output_filename, res_label_dir,
                             perturb_box2d=False, augmentX=1, type_whitelist=['Car'], remove_diff=False):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = []  # int number
    box3d_list = []  # (8,3) array in rect camera coord
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    label_list = []  # 1 for roi object, 0 for clutter
    type_list = []  # string e.g. Car
    heading_list = []  # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    gt_box2d_list = []
    calib_list = []

    enlarge_box3d_list = []
    enlarge_box3d_size_list = []
    enlarge_box3d_angle_list = []

    pos_cnt = 0
    all_cnt = 0
    thresh = 0.5 if 'Car' in type_whitelist else 0.25

    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix

        # objects = dataset.get_label_objects(data_idx)
        gt_objects = dataset.get_label_objects(data_idx)
        gt_objects, gt_boxes_2d, gt_boxes_3d = extract_boxes(
            gt_objects, type_whitelist, remove_diff)

        if len(gt_objects) == 0:
            continue

        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                 calib, 0, 0, img_width, img_height, True)

        pc_rect = pc_rect[img_fov_inds, :]
        pc_image_coord = pc_image_coord[img_fov_inds]

        label_filename = os.path.join(res_label_dir, '%06d.txt' % (data_idx))

        objects = utils.read_label(label_filename)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist:
                continue

            obj = objects[obj_idx]

            l, w, h = obj.l, obj.w, obj.h
            cx, cy, cz = obj.t
            ry = obj.ry
            cy = cy - h / 2

            obj_array = np.array([cx, cy, cz, l, w, h, ry])
            ratio = 1.2
            enlarge_obj_array = obj_array.copy()
            enlarge_obj_array[3:6] = enlarge_obj_array[3:6] * ratio

            overlap = rbbox_iou_3d(obj_array.reshape(-1, 7), gt_boxes_3d)
            overlap = overlap[0]
            max_overlap = overlap.max(0)
            max_idx = overlap.argmax(0)
            # print(max_overlap)
            if max_overlap < thresh:
                continue

            gt_obj = gt_objects[max_idx]
            gt_box2d = gt_objects[max_idx].box2d

            l, w, h = gt_obj.l, gt_obj.w, gt_obj.h
            cx, cy, cz = gt_obj.t
            ry = gt_obj.ry
            cy = cy - h / 2

            gt_obj_array = np.array([cx, cy, cz, l, w, h, ry])

            box3d_pts_3d = compute_box_3d_obj_array(gt_obj_array)

            for _ in range(augmentX):

                if perturb_box2d:
                    # print(box3d_align)

                    enlarge_obj_array = random_shift_rotate_box3d(
                        enlarge_obj_array, 0.05)
                    box3d_corners_enlarge = compute_box_3d_obj_array(
                        enlarge_obj_array)

                else:
                    box3d_corners_enlarge = compute_box_3d_obj_array(
                        enlarge_obj_array)

                _, inds = extract_pc_in_box3d(pc_rect, box3d_corners_enlarge)

                pc_in_cuboid = pc_rect[inds]
                # pc_box_image_coord = pc_image_coord[inds]

                _, inds = extract_pc_in_box3d(pc_in_cuboid, box3d_pts_3d)

                label = np.zeros((pc_in_cuboid.shape[0]))
                label[inds] = 1

                # _, inds = extract_pc_in_box3d(pc_rect, box3d_pts_3d)

                # print(np.sum(label), np.sum(inds))

                # Get 3D BOX heading
                heading_angle = gt_obj.ry
                # Get 3D BOX size
                box3d_size = np.array([gt_obj.l, gt_obj.w, gt_obj.h])

                # Reject too far away object or object without points
                if np.sum(label) == 0:
                    continue

                box3d_center = enlarge_obj_array[:3]

                frustum_angle = -1 * np.arctan2(box3d_center[2],
                                                box3d_center[0])

                id_list.append(data_idx)
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_cuboid)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)

                gt_box2d_list.append(gt_box2d)
                calib_list.append(calib.calib_dict)
                enlarge_box3d_list.append(box3d_corners_enlarge)
                enlarge_box3d_size_list.append(enlarge_obj_array[3:6])
                enlarge_box3d_angle_list.append(enlarge_obj_array[-1])
                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_cuboid.shape[0]

    print('total_objects %d' % len(id_list))
    print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt) / len(id_list)))

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp, -1)
        pickle.dump(box3d_list, fp, -1)
        pickle.dump(input_list, fp, -1)
        pickle.dump(label_list, fp, -1)
        pickle.dump(type_list, fp, -1)
        pickle.dump(heading_list, fp, -1)
        pickle.dump(box3d_size_list, fp, -1)
        pickle.dump(frustum_angle_list, fp, -1)
        pickle.dump(gt_box2d_list, fp, -1)
        pickle.dump(calib_list, fp, -1)
        pickle.dump(enlarge_box3d_list, fp, -1)
        pickle.dump(enlarge_box3d_size_list, fp, -1)
        pickle.dump(enlarge_box3d_angle_list, fp, -1)

    print('save in {}'.format(output_filename))


def get_box3d_dim_statistics(idx_filename):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'))
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type == 'DontCare':
                continue
            dimension_list.append(np.array([obj.l, obj.w, obj.h]))
            type_list.append(obj.type)
            ry_list.append(obj.ry)

    with open('box3d_dimensions.pickle', 'wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)


def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))
    return id_list, type_list, box2d_list, prob_list


def read_det_pkl_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    with open(det_filename, 'r') as fn:
        results = pickle.load(fn)

    id_list = results['id_list']
    type_list = results['type_list']
    box2d_list = results['box2d_list']
    prob_list = results['prob_list']

    return id_list, type_list, box2d_list, prob_list


def extract_frustum_data_rgb_detection(idx_filename, split, output_filename, res_label_dir,
                                       type_whitelist=['Car'],
                                       img_height_threshold=5,
                                       lidar_point_threshold=1):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''

    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []  # angle of 2d box center from pos x-axis
    box3d_pred_list = []

    calib_list = []
    enlarge_box3d_list = []
    enlarge_box3d_size_list = []
    enlarge_box3d_angle_list = []

    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix

        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                 calib, 0, 0, img_width, img_height, True)

        pc_image_coord = pc_image_coord[img_fov_inds]
        pc_rect = pc_rect[img_fov_inds]

        label_filename = os.path.join(res_label_dir, '%06d.txt' % (data_idx))

        objects = utils.read_label(label_filename)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist:
                continue

            # 2D BOX: Get pts rect backprojected
            box2d = objects[obj_idx].box2d
            xmin, ymin, xmax, ymax = box2d

            obj = objects[obj_idx]

            l, w, h = obj.l, obj.w, obj.h
            cx, cy, cz = obj.t
            ry = obj.ry
            cy = cy - h / 2

            obj_array = np.array([cx, cy, cz, l, w, h, ry])

            box3d_pts_3d = compute_box_3d_obj_array(obj_array)

            ratio = 1.2
            enlarge_obj_array = obj_array.copy()
            enlarge_obj_array[3:6] = enlarge_obj_array[3:6] * ratio
            box3d_pts_3d_l = compute_box_3d_obj_array(enlarge_obj_array)

            _, inds = extract_pc_in_box3d(pc_rect, box3d_pts_3d_l)
            pc_in_cuboid = pc_rect[inds]
            pc_box_image_coord = pc_image_coord[inds]

            box3d_center = enlarge_obj_array[:3]

            frustum_angle = -1 * np.arctan2(box3d_center[2],
                                            box3d_center[0])

            # Pass objects that are too small
            if ymax - ymin < img_height_threshold or xmax - xmin < 1 or \
                    len(pc_in_cuboid) < lidar_point_threshold:
                continue

            id_list.append(data_idx)
            input_list.append(pc_in_cuboid.astype(np.float32, copy=False))
            type_list.append(objects[obj_idx].type)
            frustum_angle_list.append(frustum_angle)
            prob_list.append(obj.score)
            box2d_list.append(box2d)

            box3d_pred_list.append(box3d_pts_3d)

            enlarge_box3d_list.append(box3d_pts_3d_l)
            enlarge_box3d_size_list.append(enlarge_obj_array[3:6])
            enlarge_box3d_angle_list.append(enlarge_obj_array[-1])

            calib_list.append(calib.calib_dict)

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp, -1)
        pickle.dump(box2d_list, fp, -1)
        pickle.dump(input_list, fp, -1)
        pickle.dump(type_list, fp, -1)
        pickle.dump(frustum_angle_list, fp, -1)
        pickle.dump(prob_list, fp, -1)

        pickle.dump(calib_list, fp, -1)

        pickle.dump(enlarge_box3d_list, fp, -1)
        pickle.dump(enlarge_box3d_size_list, fp, -1)
        pickle.dump(enlarge_box3d_angle_list, fp, -1)

    print(len(id_list))
    print('save in {}'.format(output_filename))


def write_2d_rgb_detection(det_filename, split, result_dir):
    ''' Write 2D detection results for KITTI evaluation.
        Convert from Wei's format to KITTI format. 

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        result_dir: string, folder path for results dumping
    Output:
        None (will write <xxx>.txt files to disk)

    Usage:
        write_2d_rgb_detection("val_det.txt", "training", "results")
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split)

    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    # map from idx to list of strings, each string is a line without \n
    results = {}
    for i in range(len(det_id_list)):
        idx = det_id_list[i]
        typename = det_type_list[i]
        box2d = det_box2d_list[i]
        prob = det_prob_list[i]
        output_str = typename + " -1 -1 -10 "
        output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results:
            results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_train', action='store_true',
                        help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true',
                        help='Generate val split frustum data with GT 2D boxes')

    parser.add_argument('--gen_val_det', action='store_true',
                        help='Generate val split frustum data with DET boxes')

    parser.add_argument('--gen_val_rgb_detection', action='store_true',
                        help='Generate val split frustum data with RGB detection 2D boxes')

    parser.add_argument('--car_only', action='store_true',
                        help='Only generate cars')
    parser.add_argument('--people_only', action='store_true',
                        help='Only generate person')

    parser.add_argument('--save_dir', default=None, type=str, help='data directory to save data')

    parser.add_argument('--gen_from_folder', default=None,
                        type=str, help='Generate frustum data from folder')

    args = parser.parse_args()

    np.random.seed(3)

    if args.save_dir is None:
        save_dir = 'kitti/data/pickle_data_refine'
    else:
        save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    elif args.people_only:
        type_whitelist = ['Pedestrian', 'Cyclist']
        output_prefix = 'frustum_pedcyc_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    if args.gen_train:
        extract_frustum_data(
            os.path.join(BASE_DIR, 'image_sets/train.txt'),
            'training',
            os.path.join(save_dir, output_prefix + 'train.pickle'),
            perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist)

    # TODO only use gt box 2d
    if args.gen_val:
        extract_frustum_data(
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(save_dir, output_prefix + 'val.pickle'),
            perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist, remove_diff=True)

    if args.gen_val_det:

        if args.people_only:
            res_label_dir = './output/people_train/val_nms/result/data'
        elif args.car_only:
            res_label_dir = './output/car_train/val_nms/result/data'
        else:
            assert False

        extract_frustum_det_data(
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(save_dir, output_prefix + 'val_det.pickle'),
            res_label_dir,
            perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist, remove_diff=True)

    if args.gen_val_rgb_detection:
        if args.people_only:
            res_label_dir = './output/people_train/val_nms/result/data'
        elif args.car_only:
            res_label_dir = './output/car_train/val_nms/result/data'
        else:
            assert False

        extract_frustum_data_rgb_detection(
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(save_dir, output_prefix +
                         'val_rgb_detection_refine.pickle'),
            res_label_dir,

            type_whitelist=type_whitelist)

    if args.gen_from_folder:

        res_label_dir = args.gen_from_folder
        postfix = 'val_rgb_detection_refine.pickle'
        save_dir = os.path.join(res_label_dir, '..')

        # TODO support any image set
        extract_frustum_data_rgb_detection(
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(save_dir, output_prefix + postfix),
            res_label_dir,
            type_whitelist=type_whitelist)

