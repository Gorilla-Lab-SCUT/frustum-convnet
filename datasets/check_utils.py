import math
import time
import pickle
import sys
import os
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from datasets.data_utils import project_image_to_rect, compute_box_3d


def adjust_coord_for_view(points):
    return points[:, [2, 0, 1]] * np.array([1, -1, -1])


def draw_box3d(corners, ax):
    '''
    8, 3

    '''
    order = np.array([
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        4, 5,
        5, 6,
        6, 7,
        7, 4,
        3, 7,
        0, 4,
        2, 6,
        1, 5]).reshape(-1, 2)

    for i in range(len(order)):
        ax.plot(corners[order[i], 0], corners[order[i], 1], corners[order[i], 2])


def draw_points(pts, ax):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])


def check_box_frustum(box, P, center, dimension, angle):

    x1, y1, x2, y2 = box
    box_corner = compute_box_3d(center, dimension, angle, P)  # 8, 3

    z1 = np.arange(0, 70, 0.1)

    xyz1 = np.zeros((len(z1), 3))
    xyz1[:, 0] = x1
    xyz1[:, 1] = y1
    xyz1[:, 2] = z1
    xyz1_rect = project_image_to_rect(xyz1, P)

    xyz1[:, 0] = x2
    xyz1[:, 1] = y2
    xyz1[:, 2] = z1
    xyz2_rect = project_image_to_rect(xyz1, P)

    xyz1[:, 0] = x1
    xyz1[:, 1] = y2
    xyz1[:, 2] = z1
    xyz3_rect = project_image_to_rect(xyz1, P)

    xyz1[:, 0] = x2
    xyz1[:, 1] = y1
    xyz1[:, 2] = z1
    xyz4_rect = project_image_to_rect(xyz1, P)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    draw_box3d(box_corner, ax)
    draw_points(xyz1_rect, ax)
    draw_points(xyz2_rect, ax)
    draw_points(xyz3_rect, ax)
    draw_points(xyz4_rect, ax)

    plt.show()


def check_norm(self, points, ref_points, gt_box3d_corners, pred_box3d_corners):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    points = adjust_coord_for_view(points)
    ref_points = adjust_coord_for_view(ref_points)
    gt_box3d_corners = adjust_coord_for_view(gt_box3d_corners)
    pred_box3d_corners = adjust_coord_for_view(pred_box3d_corners)

    # ax.set_aspect('equal')

    # ax.axis('equal')
    ax.set_axis_on()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    draw_points(points, ax)
    draw_points(ref_points, ax)
    draw_box3d(gt_box3d_corners, ax)
    draw_box3d(pred_box3d_corners, ax)

    plt.show()
