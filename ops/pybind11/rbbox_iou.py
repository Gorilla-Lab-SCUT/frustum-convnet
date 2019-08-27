import numpy as np

from . import box_ops_cc


# import nms


def bbox_overlaps_1d(ex, gt):
    '''
    N, 2
    K, 2

    '''
    N = ex.shape[0]
    K = gt.shape[0]

    z1 = ex[:, 0]
    z2 = ex[:, 1]

    z1_gt = gt[:, 0]
    z2_gt = gt[:, 1]

    z1 = np.broadcast_to(z1.reshape(N, 1), (N, K))
    z2 = np.broadcast_to(z2.reshape(N, 1), (N, K))

    z1_gt = np.broadcast_to(z1_gt.reshape(1, K), (N, K))
    z2_gt = np.broadcast_to(z2_gt.reshape(1, K), (N, K))

    i_z = np.minimum(z2, z2_gt) - np.maximum(z1, z1_gt)
    i_z[i_z < 0] = 0
    h_overlap = i_z / (z2_gt - z1_gt + z2 - z1 - i_z)

    return h_overlap


def bbox_overlaps_2d(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])).reshape(1, K)
    anchors_area = ((anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])).reshape(N, 1)

    boxes = np.broadcast_to(anchors.reshape(N, 1, 4), (N, K, 4))
    query_boxes = np.broadcast_to(gt_boxes.reshape(1, K, 4), (N, K, 4))

    iw = (np.minimum(boxes[:, :, 2], query_boxes[:, :, 2]) - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0]))
    iw[iw < 0] = 0

    ih = (np.minimum(boxes[:, :, 3], query_boxes[:, :, 3]) - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1]))
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def bbox_overlaps_3d(anchors, gt_boxes):
    # anchors: (N, 6) ndarray of float
    # gt_boxes: (K, 6) ndarray of float

    anchors = anchors[:, [0, 2, 3, 5, 1, 4]]
    gt_boxes = gt_boxes[:, [0, 2, 3, 5, 1, 4]]

    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0])
                     * (gt_boxes[:, 3] - gt_boxes[:, 1])
                     * (gt_boxes[:, 5] - gt_boxes[:, 4])).reshape(1, K)
    anchors_area = ((anchors[:, 2] - anchors[:, 0])
                    * (anchors[:, 3] - anchors[:, 1])
                    * (anchors[:, 5] - anchors[:, 4])).reshape(N, 1)

    boxes = np.broadcast_to(anchors.reshape(N, 1, 6), (N, K, 6))
    query_boxes = np.broadcast_to(gt_boxes.reshape(1, K, 6), (N, K, 6))

    il = (np.minimum(boxes[:, :, 2], query_boxes[:, :, 2]) - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0]))
    il[il < 0] = 0

    iw = (np.minimum(boxes[:, :, 3], query_boxes[:, :, 3]) - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1]))
    iw[iw < 0] = 0

    ih = (np.minimum(boxes[:, :, 5], query_boxes[:, :, 5]) - np.maximum(boxes[:, :, 4], query_boxes[:, :, 4]))
    ih[ih < 0] = 0

    inter = il * iw * ih
    ua = anchors_area + gt_boxes_area - inter

    overlaps = inter / ua

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
    x_corners = np.stack([-l / 2, -l / 2, l / 2, l / 2], 1)  # n, 4
    z_corners = np.stack([-w / 2, w / 2, w / 2, -w / 2], 1)  # n, 4
    corners = np.stack([x_corners, z_corners], 1)  # n, 2, 4

    rot = np.stack([np.cos(r), np.sin(r), -np.sin(r), np.cos(r)], 1).reshape(-1, 2, 2)  # n, 2, 2

    # (n, 2, 2) * (n, 2, 4) -> n, 2, 4
    corners = np.einsum('bij,bjk->bik', rot, corners)
    corners = corners + np.expand_dims(boxes_2d[:, :2], -1)
    return corners.transpose((0, 2, 1))


def boxes3d2corners(boxes_3d):
    """ b, 7 (cx, cy, cz, l, w, h, r)"""

    N = boxes_3d.shape[0]
    centers = boxes_3d[:, :3]
    l = boxes_3d[:, 3]  # (N)
    w = boxes_3d[:, 4]  # (N)
    h = boxes_3d[:, 5]  # (N)
    headings = boxes_3d[:, 6]
    # print l,w,h
    x_corners = np.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], 1)  # (N,8)
    y_corners = np.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], 1)  # (N,8)
    z_corners = np.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], 1)  # (N,8)
    corners = np.stack([x_corners, y_corners, z_corners], 1)  # (N,3,8)
    # print x_corners, y_corners, z_corners
    c = np.cos(headings)
    s = np.sin(headings)
    ones = np.ones(N, dtype=boxes_3d.dtype)
    zeros = np.zeros(N, dtype=boxes_3d.dtype)
    row1 = np.stack([c, zeros, s], 1)  # (N,3)
    row2 = np.stack([zeros, ones, zeros], 1)
    row3 = np.stack([-s, zeros, c], 1)
    R = np.stack([row1, row2, row3], 1)  # (N,3,3)
    # (N,3,3) * ((N,3,8))
    corners_3d = np.einsum('bij,bjk->bik', R, corners)
    corners_3d = corners_3d + np.expand_dims(centers, 2)  # (N,3,8)
    corners_3d = np.transpose(corners_3d, (0, 2, 1))  # (N,8,3)
    return corners_3d


def corner2standup(corners):
    x1y1 = np.min(corners, 1)  # n, 2
    x2y2 = np.max(corners, 1)

    return np.concatenate([x1y1, x2y2], 1)  # n,4


def rbbox_iou(boxes_2d, qboxes_2d, standup_thresh=0.0):
    boxes_corners = rbbox2corner(boxes_2d)
    qboxes_corners = rbbox2corner(qboxes_2d)

    boxes_standup = corner2standup(boxes_corners)
    qboxes_standup = corner2standup(qboxes_corners)

    standup_iou = bbox_overlaps_2d(boxes_standup, qboxes_standup)

    return box_ops_cc.rbbox_iou(boxes_corners, qboxes_corners, standup_iou, standup_thresh)


def rbbox_iou_3d(boxes_3d, qboxes_3d, standup_thresh=0.0):
    '''
    boxes_3d, qboxes_3d: (cx, cy, cz, l, w, h, r) n, 7

    '''

    bbox_corner_3d = boxes3d2corners(boxes_3d)  # n, 8, 3
    qbbox_corner_3d = boxes3d2corners(qboxes_3d)

    bbox_standup = np.concatenate([np.min(bbox_corner_3d, 1), np.max(bbox_corner_3d, 1)], 1)  # n, 6
    qbbox_standup = np.concatenate([np.min(qbbox_corner_3d, 1), np.max(qbbox_corner_3d, 1)], 1)

    standup_iou = bbox_overlaps_3d(bbox_standup, qbbox_standup)

    o = box_ops_cc.rbbox_iou_3d(bbox_corner_3d, qbbox_corner_3d, standup_iou, 0)

    return o


def rbbox_iou_3d_pair(boxes_3d, qboxes_3d):
    '''
    boxes_3d, qboxes_3d: (cx, cy, cz, l, w, h, r) n, 7

    '''
    assert boxes_3d.shape == qboxes_3d.shape
    bbox_corner_3d = boxes3d2corners(boxes_3d)  # n, 8, 3
    qbbox_corner_3d = boxes3d2corners(qboxes_3d)

    o = box_ops_cc.rbbox_iou_3d_pair(bbox_corner_3d, qbbox_corner_3d)

    return o


def cube_nms_np(dets, nms_thresh, top_k=300):
    '''
    :param dets: [[cx, cy, cz, l, w, h, ry, score]]
    :param thresh: retain overlap < thresh
    :return: indices to keep
    '''
    if dets.shape[0] == 0:
        return []
    if dets.shape[0] == 1:
        return [0]

    assert dets.shape[1] == 8

    scores = dets[:, 7]

    order = scores.argsort()[::-1]
    order = order[:top_k]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        box1 = dets[i, :7][np.newaxis, :]
        boxes2 = dets[:, :7][order[1:]]

        ovr = rbbox_iou_3d(box1, boxes2)
        ovr = ovr[0]

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def bev_nms_np(dets, nms_thresh, top_k=300):
    '''
    :param dets: [[cx, cz, l, w, ry, score]]
    :param thresh: retain overlap < thresh
    :return: indices to keep
    '''
    if dets.shape[0] == 0:
        return []
    if dets.shape[0] == 1:
        return [0]

    assert dets.shape[1] == 6

    scores = dets[:, 5]

    order = scores.argsort()[::-1]
    order = order[:top_k]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        box1 = dets[i, :5][np.newaxis, :]
        boxes2 = dets[:, :5][order[1:]]

        ovr = rbbox_iou(box1, boxes2)
        ovr = ovr[0]

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


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

    box1_bev = box1[:, [0, 2, 3, 4, 6]]
    box2_bev = box2[:, [0, 2, 3, 4, 6]]

    # o = rbbox_iou(box1_bev, box2_bev)
    # print(o)
    import time

    box1 = np.tile(box1, (1000, 1))
    box2 = np.tile(box2, (1000, 1))
    tic = time.time()
    o = rbbox_iou_3d_pair(box1, box2)
    print(time.time() - tic)

    print(o)
