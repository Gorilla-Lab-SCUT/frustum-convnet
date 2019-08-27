import os
import sys
import warnings

import numpy as np
import torch


def huber_loss(error, delta, weight=None):
    delta = torch.ones_like(error) * delta
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    # Note condisder batch mean
    if weight is not None:
        losses *= weight

    return losses.mean()


def smooth_l1_loss(input, target, sigma=1.0, size_average=True):
    '''
    input: B, *
    target: B, *

    '''
    # smooth_l1_loss with sigma
    """
            (sigma * x)^2/2  if x<1/sigma^2
    f(x)=
            |x| - 1/(2*sigma^2) otherwise
    """
    assert input.shape == target.shape

    diff = torch.abs(input - target)

    mask = (diff < (1. / sigma**2)).detach().type_as(diff)

    output = mask * torch.pow(sigma * diff, 2) / 2.0 + (1 - mask) * (diff - 1.0 / (2.0 * sigma**2.0))
    loss = output.sum()
    if size_average:
        loss = loss / input.shape[0]

    return loss


def get_box3d_corners_helper(centers, headings, sizes):

    N = centers.shape[0]
    l = sizes[:, 0]  # (N)
    w = sizes[:, 1]  # (N)
    h = sizes[:, 2]  # (N)
    x_corners = torch.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], 1)  # (N,8)
    y_corners = torch.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], 1)  # (N,8)
    z_corners = torch.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], 1)  # (N,8)
    corners = torch.stack([x_corners, y_corners, z_corners], 1)  # (N,3,8)

    c = torch.cos(headings)
    s = torch.sin(headings)
    ones = headings.new_ones(N)
    zeros = headings.new_zeros(N)
    row1 = torch.stack([c, zeros, s], 1)  # (N,3)
    row2 = torch.stack([zeros, ones, zeros], 1)
    row3 = torch.stack([-s, zeros, c], 1)
    R = torch.stack([row1, row2, row3], 1)  # (N,3,3)

    # (N,3,3) * ((N,3,8))
    corners_3d = torch.bmm(R, corners)  # (N,3,8)
    corners_3d = corners_3d + centers.unsqueeze(2)  # (N,3,8)
    corners_3d = torch.transpose(corners_3d, 1, 2).contiguous()  # (N,8,3)
    return corners_3d
