import numpy as np
import torch


def size_decode(offset, class_mean_size, size_class_label):

    offset_select = torch.gather(offset, 1, size_class_label.view(-1, 1, 1).expand(-1, -1, 3))
    offset_select = offset_select.squeeze(1)

    ex = class_mean_size[size_class_label]

    return offset_select * ex + ex


def size_encode(gt, class_mean_size, size_class_label):
    ex = class_mean_size[size_class_label]
    return (gt - ex) / ex


def center_decode(ex, offset):
    return ex + offset


def center_encode(gt, ex):
    return gt - ex


def angle_decode(ex_res, ex_class_id, num_bins=12, to_label_format=True):

    ex_res_select = torch.gather(ex_res, 1, ex_class_id.unsqueeze(1))
    ex_res_select = ex_res_select.squeeze(1)

    angle_per_class = 2 * np.pi / float(num_bins)

    angle = ex_class_id.float() * angle_per_class + ex_res_select * (angle_per_class / 2)

    if to_label_format:
        flag = angle > np.pi
        angle[flag] = angle[flag] - 2 * np.pi

    return angle

# def angle_encode(gt_angle, num_bins=12):
#     gt_angle = gt_angle % (2 * np.pi)
#     angle_per_class = 2 * np.pi / float(num_bins)

#     gt_class_id = torch.round(gt_angle / angle_per_class).long()
#     gt_res = gt_angle - gt_class_id.float() * angle_per_class

#     gt_res /= angle_per_class
#     print(gt_class_id.min().item(), gt_class_id.max().item())
#     return gt_class_id, gt_res


def angle_encode(gt_angle, num_bins=12):
    gt_angle = gt_angle % (2 * np.pi)
    assert ((gt_angle >= 0) & (gt_angle <= 2 * np.pi)).all()

    angle_per_class = 2 * np.pi / float(num_bins)
    shifted_angle = (gt_angle + angle_per_class / 2) % (2 * np.pi)
    gt_class_id = torch.floor(shifted_angle / angle_per_class).long()
    gt_res = shifted_angle - (gt_class_id.float() * angle_per_class + angle_per_class / 2)

    gt_res /= (angle_per_class / 2)
    return gt_class_id, gt_res
