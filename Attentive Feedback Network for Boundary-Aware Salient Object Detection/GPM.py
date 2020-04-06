# -*- coding: utf-8 -*-
# @Time    : 2019/7/10 下午8:37
# @Author  : Lart Pang
# @FileName: GPM.py
# @Home    : https://www.yuque.com/lart/architecture/mutli
# @GitHub  : https://github.com/lartpang

# https://drive.google.com/open?id=1EVZR8cNGUv3zb7JtR1fxbXZ8lp5mbgWe
import torch
from torch import nn


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            ), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.basicconv(x)


class GPM(nn.Module):
    def __init__(self, in_C, out_C, n=(2, 4, 7)):
        super(GPM, self).__init__()
        self.n = n
        channal_list = []
        self.cnn_list = nn.ModuleList()
        for i in self.n:
            mid_C = i * i * in_C
            channal_list.append(mid_C)
            self.cnn_list.append(BasicConv2d(mid_C, mid_C, 3, 1, 1))

        self.fuse = BasicConv2d(3 * in_C, out_C, 1)

    def forward(self, in_feat):
        assert all([in_feat.size(2) % n == 0 for n in self.n])

        feats = []
        for idx, n in enumerate(self.n):
            chunk_feats = [y for x in in_feat.chunk(n, 2) for y in x.chunk(n, 3)]
            chunk_feats = torch.cat(chunk_feats, dim=1)
            chunk_feats = self.cnn_list[idx](chunk_feats)

            total_feat = []
            for x in chunk_feats.chunk(n, 1):

                row_feat = []
                for y in x.chunk(n, 1):
                    row_feat.append(y)

                row_feat = torch.cat(row_feat, dim=3)
                total_feat.append(row_feat)

            total_feat = torch.cat(total_feat, dim=2)
            feats.append(total_feat)

        return self.fuse(torch.cat(feats, dim=1))


if __name__ == '__main__':
    a = torch.rand((4, 32, 28, 28)).cuda()
    gpm = GPM(32, 32).cuda()
    print(gpm(a).size())
