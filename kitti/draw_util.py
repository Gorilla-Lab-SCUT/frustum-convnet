''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
'''

import os
import sys
import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))

from kitti_object import kitti_object, kitti_object_video
import kitti_util as utils


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo[:, :3])
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def show_image_with_boxes(img, objects, calib, show3d=True, color=(0, 255, 0), scores=[], colors=[], show=True):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox

    if len(colors) != 0:
        assert len(colors) == len(objects)

    for i, obj in enumerate(objects):
        if obj.type == 'DontCare':
            continue
        if len(colors) != 0:
            color = colors[i]
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                      (int(obj.xmax), int(obj.ymax)), color, 2)

        if len(scores) != 0:
            # Show text.
            area = (obj.xmax - obj.xmin) * (obj.ymax - obj.ymin)

            if area > 25:
                cv2.putText(img1, '%.2f' % scores[i], (int((obj.xmin + obj.xmax) / 2), max(int(obj.ymin) - 3, 0)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=0.8, color=color, thickness=1, lineType=cv2.LINE_AA)

        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color)

            if len(scores) != 0:
                xx1, yy1 = np.min(box3d_pts_2d, 0)
                xx2, yy2 = np.max(box3d_pts_2d, 0)
                if (xx2 - xx1) * (yy2 - yy1) > 25:
                    cv2.putText(img2, '%.2f' % scores[i], (int((xx1 + xx2) / 2), max(int(yy1) - 3, 0)),
                                cv2.FONT_HERSHEY_PLAIN, fontScale=0.8, color=color, thickness=1, lineType=cv2.LINE_AA)

    if show:
        Image.fromarray(img1).show()
        if show3d:
            Image.fromarray(img2).show()

    return img1, img2


def show_lidar_with_boxes(pc_velo, objects, calib,
                          img_fov=False, img_width=None, img_height=None):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''
    if 'mlab' not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print('All point num: ', pc_velo.shape[0])
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0,
                                         img_width, img_height)
        print('FOV point num: ', pc_velo.shape[0])
    draw_lidar(pc_velo, fig=fig)

    for obj in objects:
        if obj.type == 'DontCare':
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
        mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.5, 0.5, 0.5),
                    tube_radius=None, line_width=1, figure=fig)
    mlab.show(1)


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
                                                              calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i, 0])),
                         int(np.round(imgfov_pts_2d[i, 1]))),
                   2, color=tuple(color), thickness=-1)
    Image.fromarray(img).show()
    return img


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
    return image


def dataset_viz():
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))

    for data_idx in range(len(dataset)):
        # Load data from dataset
        objects = dataset.get_label_objects(data_idx)
        objects[0].print_object()
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        print('Image shape: ', img.shape)
        pc_velo = dataset.get_lidar(data_idx)[:, 0:3]
        calib = dataset.get_calibration(data_idx)

        # Draw 2d and 3d boxes on image
        show_image_with_boxes(img, objects, calib, False)
        input()
        # Show all LiDAR points. Draw 3d box in LiDAR point cloud
        show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
        input()


def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, 'dataset/2011_09_26/')
    dataset = kitti_object_video(
        os.path.join(video_path, '2011_09_26_drive_0023_sync/image_02/data'),
        os.path.join(video_path, '2011_09_26_drive_0023_sync/velodyne_points/data'),
        video_path)
    print(len(dataset))
    for i in range(len(dataset)):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        Image.fromarray(img).show()
        draw_lidar(pc)
        input()
        pc[:, 0:3] = dataset.get_calibration().project_velo_to_rect(pc[:, 0:3])
        draw_lidar(pc)
        input()
    return


if __name__ == '__main__':
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    dataset_viz()
