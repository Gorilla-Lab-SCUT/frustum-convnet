from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import math
import time

import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F


def init_params(m, method='constant'):
    """
    method: xavier_uniform, kaiming_normal, constant
    """
    if isinstance(m, list):
        for im in m:
            init_params(im, method)
    else:
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data)
        elif method == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        elif isinstance(method, (int, float)):
            m.weight.data.fill_(method)
        else:
            raise ValueError("unknown method.")
        if m.bias is not None:
            m.bias.data.zero_()


def Conv1d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.Conv1d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm1d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.Conv1d(i_c, o_c, k, s, p), nn.ReLU(True))


def Conv2d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.Conv2d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm2d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.Conv2d(i_c, o_c, k, s, p), nn.ReLU(True))


def Conv3d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.Conv3d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm3d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.Conv3d(i_c, o_c, k, s, p), nn.ReLU(True))


def DeConv1d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.ConvTranspose1d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm1d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.ConvTranspose1d(i_c, o_c, k, s, p), nn.ReLU(True))


def DeConv2d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.ConvTranspose2d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm2d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.ConvTranspose2d(i_c, o_c, k, s, p), nn.ReLU(True))


def DeConv3d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.ConvTranspose3d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm3d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.ConvTranspose3d(i_c, o_c, k, s, p), nn.ReLU(True))


def get_accuracy(output, target, ignore=None):

    assert output.shape[0] == target.shape[0]
    if ignore is not None:
        assert isinstance(ignore, int)
        keep = (target != ignore).nonzero().view(-1)
        output = output[keep]
        target = target[keep]

    pred = torch.argmax(output, -1)

    correct = (pred.view(-1) == target.view(-1)).float().sum()
    acc = correct * (1.0 / target.view(-1).shape[0])

    return acc

def scatter_nd(x, y, shape):
    '''
    Scatter updates into a new (initially zero) tensor according to indices.
    x (b, feat, k)
    y index LongTensor (b, k, 3)
    shape (n1, n2, n3)
    out (b, feat, n1, n2, n3)
    '''
    assert len(shape) == y.shape[-1]
    assert x.dim() == 3 and x.shape[-1] == y.shape[1]
    # assert y.shape[-1] == 3
    assert x.is_contiguous()

    batch_size = x.shape[0]
    num_feats = x.shape[1]
    output = x.new_zeros(batch_size, num_feats, *shape)

    # flat index to one dimension
    stride = torch.LongTensor(output[0][0].stride()).type_as(y.data)

    index = torch.sum(stride * y, dim=-1)  # b, k

    output = output.view(batch_size, num_feats, -1)
    output.scatter_(2, index.unsqueeze(1).expand(-1, num_feats, -1).contiguous(), x)

    return output.view(batch_size, num_feats, *shape)


def scatter_add_nd(x, y, shape):
    '''
    Scatter updates into a new (initially zero) tensor according to indices.
    x (b, feat, k)
    y index LongTensor (b, k, 3)
    shape (n1, n2, n3)
    out (b, feat, n1, n2, n3)
    '''
    assert len(shape) == y.shape[-1]
    assert x.dim() == 3 and x.shape[-1] == y.shape[1]
    # assert y.shape[-1] == 3
    assert x.is_contiguous()

    batch_size = x.shape[0]
    num_feats = x.shape[1]
    output = x.new_zeros(batch_size, num_feats, *shape)

    # flat index to one dimension
    stride = torch.LongTensor(output[0][0].stride()).type_as(y.data)

    index = torch.sum(stride * y, dim=-1)  # b, k

    output = output.view(batch_size, num_feats, -1)
    output.scatter_add_(2, index.unsqueeze(1).expand(-1, num_feats, -1).contiguous(), x)

    return output.view(batch_size, num_feats, *shape)


def gather_nd(x, y):
    '''
    Scatter updates into a new (initially zero) tensor according to indices.
    x (b, feat, k)
    y index LongTensor (b, k, 3)
    shape (n1, n2, n3)
    out (b, feat, n1, n2, n3)
    '''
    assert x.is_contiguous()

    batch_size = x.shape[0]
    num_feats = x.shape[1]

    # flat index to one dimension
    stride = torch.LongTensor(x[0][0].stride()).type_as(y.data)

    index = torch.sum(stride * y, dim=-1)  # b, k

    x = x.view(batch_size, num_feats, -1)
    return torch.gather(x, 2, index.unsqueeze(1).expand(-1, num_feats, -1).contiguous())


def scatter_1d(x, y, shape):
    assert len(shape) == 1
    output = x.new_zeros(x.shape[0], x.shape[1], *(shape))
    output.scatter_(2, y.unsqueeze(1).expand(-1, x.shape[1], -1), x)

    return output


def sigmoid_focal_loss(prob, target, alpha=0.25, gamma=2, grad_scale=None, weights=None):
    pt = target * prob + (1 - target) * (1 - prob)

    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = -alpha_t * (1 - pt) ** gamma * torch.log(pt + 1e-14)

    if weights is not None:
        assert weights.shape == loss.shape
        loss *= weights
    if grad_scale is None:
        grad_scale = 1.0 / loss.shape[0]

    loss = loss.sum() * grad_scale
    return loss


def softmax_focal_loss(prob, target, alpha=0.25, gamma=2, grad_scale=None, weights=None):
    alpha_t = (1 - alpha) * (target == 0).float() + alpha * (target >= 1).float()

    prob_t = prob[range(len(target)), target]
    # alpha_t = alpha_t[range(len(target)), target]
    loss = -alpha_t * (1 - prob_t) ** gamma * torch.log(prob_t + 1e-14)

    if weights is not None:
        assert weights.shape == loss.shape
        loss *= weights

    if grad_scale is None:
        grad_scale = 1.0 / loss.shape[0]

    loss = loss.sum() * grad_scale

    return loss


def softmax_focal_loss_ignore(prob, target, alpha=0.25, gamma=2, ignore_idx=-1):
    keep = (target != ignore_idx).nonzero().view(-1)
    num_fg = (target > 0).data.sum()

    target = target[keep]
    prob = prob[keep, :]

    alpha_t = (1 - alpha) * (target == 0).float() + alpha * (target >= 1).float()

    prob_t = prob[range(len(target)), target]
    # alpha_t = alpha_t[range(len(target)), target]
    loss = -alpha_t * (1 - prob_t) ** gamma * torch.log(prob_t + 1e-14)

    loss = loss.sum() / (num_fg + 1e-14)

    return loss


def separable_conv2d(in_channels, out_channels, k, s=(1, 1), depth_multiplier=1):
    #
    conv = [nn.Conv2d(in_channels, in_channels * depth_multiplier, k, groups=in_channels)]

    if out_channels is not None:
        conv += [nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, 1)]

    conv = nn.Sequential(*conv)

    return conv


class XConv(nn.Module):
    def __init__(self, K, C, depth_multiplier=1, with_X_transformation=True):

        super(XConv, self).__init__()

        # transform K*K
        self.conv_t0 = \
            nn.Sequential(
                nn.Conv2d(3, K * K, (1, K)),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(K * K),
            )

        self.conv_t1 = \
            nn.Sequential(
                separable_conv2d(K, None, (1, K), (1, 1), K),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(K * K),
            )

        self.conv_t2 = \
            nn.Sequential(
                separable_conv2d(K, None, (1, K), (1, 1), K),
                nn.BatchNorm2d(K * K),

            )

        self.separable_conv2d = \
            nn.Sequential(
                separable_conv2d(C, None, (1, K), (1, 1), depth_multiplier),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(C),

            )

        self.with_X_transformation = with_X_transformation

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, nn_pts_local, nn_fts_input):
        """
        pts: nn_pts_local (N, 3, P, K)
        nn_fts_input : (N, C, P, K) 

        """
        N, C0, P, K = nn_pts_local.shape
        assert C0 == 3

        if self.with_X_transformation:

            X_0 = self.conv_t0(nn_pts_local)  # N, K*K, P, 1
            X_0 = X_0.view(N, K, K, P).transpose(2, 3).contiguous()  # N, K, P, K
            X_1 = self.conv_t1(X_0)
            X_1 = X_1.view(N, K, K, P).transpose(2, 3).contiguous()  # N, K, P, K
            X_2 = self.conv_t2(X_1)
            X_2 = X_1.view(N, K, K, P).permute(0, 3, 1, 2).contiguous()  # N, K, P, K

            X = X_2.view(N * P, K, K)  #
            # C1 = prev_C + C_pts_fts
            # N,C1,P,K -> N,P,C1,K -> N, P, K, C1
            nn_fts_input = nn_fts_input.permute(0, 2, 3, 1).contiguous().view(N * P, K, -1)
            # (N * P, K, K)  (N * P, K, C1)  -> (N * P, K, C1)
            fts_X = torch.bmm(X, nn_fts_input)
            # N, P, K, C1 -> N, C1, K, P -> N, C1, P, K
            fts_X = fts_X.view(N, P, K, -1).permute(0, 3, 1, 2).contiguous()

        else:
            fts_X = nn_fts_input

        fts = self.separable_conv2d(fts_X)  # (N, C, P, 1)

        # return fts.squeeze(-1)  # (N, C, P)
        return fts
