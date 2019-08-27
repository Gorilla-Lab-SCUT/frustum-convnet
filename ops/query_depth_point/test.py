import sys
import torch
import torch.nn as nn

sys.path.insert(0, '../')
from query_depth_point.query_depth_point import QueryDepthPoint

xyz1 = torch.rand(2, 3, 50) * 2 - 1

xyz2 = torch.index_select(xyz1, 2, torch.arange(10).long())

batch_size = xyz1.shape[0]
mask = torch.zeros(batch_size, xyz2.shape[2], xyz1.shape[2])
for i in range(batch_size):
    for j in range(xyz2.shape[2]):
        x1, y1, z1 = xyz2[i, :, j]
        inside = (torch.abs(xyz1[i, 2] - z1) < 0.2)
        mask[i, j] = inside

xyz1 = xyz1.float().cuda()
xyz2 = xyz2.float().cuda()

net = QueryDepthPoint(0.2, 4).cuda()

idx, pts_idx = net(xyz1, xyz2)
print(idx[0])
print(pts_idx)
print(torch.nonzero(mask[0]))
