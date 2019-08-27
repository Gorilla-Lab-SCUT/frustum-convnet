import math
from torch import nn
from torch.autograd import Function
import torch

from . import query_depth_point_cuda


class _query_depth_point(Function):

    @staticmethod
    def forward(ctx, dis_z, nsample, xyz1, xyz2):
        '''
        Input:
            dis_z: float32, depth distance search distance
            nsample: int32, number of points selected in each ball region
            xyz1: (batch_size, 3, ndataset) float32 array, input points
            xyz2: (batch_size, 3, npoint) float32 array, query points
        Output:
            idx: (batch_size, npoint, nsample) int32 array, indices to input points
            pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
        '''
        assert xyz1.is_cuda and xyz1.size(1) == 3
        assert xyz2.is_cuda and xyz2.size(1) == 3
        assert xyz1.size(0) == xyz2.size(0)
        assert xyz1.is_contiguous()
        assert xyz2.is_contiguous()

        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()

        b = xyz1.size(0)
        n = xyz1.size(1)
        m = xyz2.size(1)

        idx = xyz1.new(b, m, nsample).long().zero_()
        pts_cnt = xyz1.new(b, m).int().zero_()

        query_depth_point_cuda.forward(b, n, m, dis_z, nsample, xyz1, xyz2, idx, pts_cnt)
        return idx, pts_cnt

    @staticmethod
    def backward(ctx, grad_output):
        return (None,) * 6


class QueryDepthPoint(nn.Module):
    def __init__(self, dis_z, nsample):
        super(QueryDepthPoint, self).__init__()
        self.dis_z = dis_z
        self.nsample = nsample

    def forward(self, xyz1, xyz2):
        return _query_depth_point.apply(self.dis_z, self.nsample, xyz1, xyz2)
