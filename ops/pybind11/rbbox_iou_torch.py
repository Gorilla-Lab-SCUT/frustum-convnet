import numpy as np
import torch

from . import box_ops_cc


# import nms


def bbox_overlaps_1d(ex, gt):
    '''
    N, 2
    K, 2

    '''
    N = ex.shape[0]
    K = gt.shape[0]

    ex_area = (ex[:, 1] - ex[:, 0]).view(N, 1)
    gt_boxes_area = (gt[:, 1] - gt[:, 0]).view(1, K)

    boxes = ex.view(N, 1, 2).expand(N, K, 2)
    query_boxes = gt.view(1, K, 2).expand(N, K, 2)

    ih = (torch.min(boxes[:, :, 1], query_boxes[:, :, 1]) -
          torch.max(boxes[:, :, 0], query_boxes[:, :, 0]))

    ih[ih < 0] = 0
    h_overlap = ih / (ex_area + gt_boxes_area - ih)

    return h_overlap


def bbox_overlaps_2d(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0]) *
                     (gt_boxes[:, 3] - gt_boxes[:, 1])).view(1, K)

    anchors_area = ((anchors[:, 2] - anchors[:, 0]) *
                    (anchors[:, 3] - anchors[:, 1])).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:, :, 2], query_boxes[:, :, 2]) -
          torch.max(boxes[:, :, 0], query_boxes[:, :, 0]))
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:, :, 3], query_boxes[:, :, 3]) -
          torch.max(boxes[:, :, 1], query_boxes[:, :, 1]))
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def bbox_overlaps_3d(anchors, gt_boxes):
    # anchors: (N, 6) ndarray of float
    # gt_boxes: (K, 6) ndarray of float
    # (x1, y1, z1, x2, y2, z2)

    assert anchors.dim() == 2 and gt_boxes.dim() == 2

    anchors = anchors[:, [0, 2, 3, 5, 1, 4]]
    gt_boxes = gt_boxes[:, [0, 2, 3, 5, 1, 4]]

    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0])
                     * (gt_boxes[:, 3] - gt_boxes[:, 1])
                     * (gt_boxes[:, 5] - gt_boxes[:, 4])).view(1, K)
    anchors_area = ((anchors[:, 2] - anchors[:, 0])
                    * (anchors[:, 3] - anchors[:, 1])
                    * (anchors[:, 5] - anchors[:, 4])).view(N, 1)

    gt_area_zero = (gt_boxes_area == 0)
    anchors_area_zero = (anchors_area == 0)

    boxes = anchors.view(N, 1, 6).expand(N, K, 6)
    query_boxes = gt_boxes.view(1, K, 6).expand(N, K, 6)

    il = (torch.min(boxes[:, :, 2], query_boxes[:, :, 2]) - torch.max(boxes[:, :, 0], query_boxes[:, :, 0]))
    il[il < 0] = 0

    iw = (torch.min(boxes[:, :, 3], query_boxes[:, :, 3]) - torch.max(boxes[:, :, 1], query_boxes[:, :, 1]))
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:, :, 5], query_boxes[:, :, 5]) - torch.max(boxes[:, :, 4], query_boxes[:, :, 4]))
    ih[ih < 0] = 0

    inter = il * iw * ih
    ua = anchors_area + gt_boxes_area - inter

    overlaps = inter / ua

    overlaps.masked_fill_(gt_area_zero.expand(N, K), 0)
    overlaps.masked_fill_(anchors_area_zero.expand(N, K), -1)

    return overlaps


def rbbox2corner(boxes_2d):
    '''
    boxes_2d: n, 4 (cx, cz, l, w, r)
    return n, 4, 2

    '''
    l = boxes_2d[:, 2]
    w = boxes_2d[:, 3]
    r = boxes_2d[:, 4]
    # x0y0, x0y1, x1y1, x1y0, clockwise start with minimum point
    x_corners = torch.stack([-l / 2, -l / 2, l / 2, l / 2], 1)  # n, 4
    z_corners = torch.stack([-w / 2, w / 2, w / 2, -w / 2], 1)  # n, 4
    corners = torch.stack([x_corners, z_corners], 1)  # n, 2, 4

    rot = torch.stack([torch.cos(r), torch.sin(r), -torch.sin(r), torch.cos(r)], 1).view(-1, 2, 2)  # n, 2, 2

    # (n, 2, 2) * (n, 2, 4) -> n, 2, 4
    corners = torch.bmm(rot, corners)
    corners = corners + boxes_2d[:, :2].unsqueeze(-1)
    return corners.transpose(2, 1)


def boxes3d2corners(boxes_3d):
    """ b, 7 (cx, cy, cz, l, w, h, r)"""

    N = boxes_3d.shape[0]
    centers = boxes_3d[:, :3]
    l = boxes_3d[:, 3]  # (N)
    w = boxes_3d[:, 4]  # (N)
    h = boxes_3d[:, 5]  # (N)
    headings = boxes_3d[:, 6]
    # print l,w,h
    x_corners = torch.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], 1)  # (N,8)
    y_corners = torch.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], 1)  # (N,8)
    z_corners = torch.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], 1)  # (N,8)
    corners = torch.stack([x_corners, y_corners, z_corners], 1)  # (N,3,8)
    # print x_corners, y_corners, z_corners
    c = torch.cos(headings)
    s = torch.sin(headings)
    ones = boxes_3d.new(N).fill_(1)
    zeros = boxes_3d.new(N).fill_(0)
    row1 = torch.stack([c, zeros, s], 1)  # (N,3)
    row2 = torch.stack([zeros, ones, zeros], 1)
    row3 = torch.stack([-s, zeros, c], 1)
    R = torch.stack([row1, row2, row3], 1)  # (N,3,3)
    # (N,3,3) * ((N,3,8))
    corners_3d = torch.bmm(R, corners)
    corners_3d = corners_3d + centers.unsqueeze(-1)  # (N,3,8)
    corners_3d = corners_3d.transpose(2, 1)  # (N,8,3)
    return corners_3d


def corner2standup(corners):
    x1y1 = torch.min(corners, 1)[0]  # n, 2
    x2y2 = torch.max(corners, 1)[0]

    return torch.cat([x1y1, x2y2], 1)  # n,4


def rbbox_iou(boxes_2d, qboxes_2d, standup_thresh=0.0):
    boxes_corners = rbbox2corner(boxes_2d)
    qboxes_corners = rbbox2corner(qboxes_2d)

    boxes_standup = corner2standup(boxes_corners)
    qboxes_standup = corner2standup(qboxes_corners)

    standup_iou = bbox_overlaps_2d(boxes_standup, qboxes_standup)

    boxes_corners = boxes_corners.cpu().numpy()
    qboxes_corners = qboxes_corners.cpu().numpy()
    standup_iou = standup_iou.cpu().numpy()

    return box_ops_cc.rbbox_iou(boxes_corners, qboxes_corners, standup_iou, standup_thresh)


def rbbox_iou_3d(boxes_3d, qboxes_3d, standup_thresh=0.0):
    '''
    boxes_3d: (cx, cy, cz, l, w, h, r) n, 7

    '''

    bbox_corner_3d = boxes3d2corners(boxes_3d)  # n, 8, 3
    qbbox_corner_3d = boxes3d2corners(qboxes_3d)

    bbox_standup = torch.cat([torch.min(bbox_corner_3d, 1)[0], torch.max(bbox_corner_3d, 1)[0]], 1)  # n, 6
    qbbox_standup = torch.cat([torch.min(qbbox_corner_3d, 1)[0], torch.max(qbbox_corner_3d, 1)[0]], 1)

    standup_iou = bbox_overlaps_3d(bbox_standup, qbbox_standup)

    bbox_corner_3d = bbox_corner_3d.cpu().numpy()
    qbbox_corner_3d = qbbox_corner_3d.cpu().numpy()
    standup_iou = standup_iou.cpu().numpy()

    o = box_ops_cc.rbbox_iou_3d(bbox_corner_3d, qbbox_corner_3d, standup_iou, 0)

    return o


if __name__ == '__main__':
    # import matplotlib.pyplot as plt

    # line1 = np.concatenate([boxes_corners[0], boxes_corners[0][[0]]], 0)
    # line2 = np.concatenate([qboxes_corners[0], qboxes_corners[0][[0]]], 0)

    # plt.plot(line1[:, 0], line1[:, 1])
    # plt.plot(line2[:, 0], line2[:, 1])

    # plt.show()
    # cx, cy, cz, l, w, h, r
    box1 = np.array([[0, 0.2, 0.3, 2.2, 3, 1, 0.78 * np.pi]])
    box2 = np.array([[1.5, 0.4, 0, 2.1, 3, 1.2, 0.5 * np.pi]])

    box1 = torch.from_numpy(box1)
    box2 = torch.from_numpy(box2)

    box1_bev = box1[:, [0, 2, 3, 4, 6]]
    box2_bev = box2[:, [0, 2, 3, 4, 6]]

    o = rbbox_iou(box1_bev, box2_bev)
    print(o)

    o = rbbox_iou_3d(box1, box2)
    print(o)
