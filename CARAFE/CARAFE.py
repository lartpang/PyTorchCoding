# -*- coding: utf-8 -*-
# @Time    : 2019/8/2 下午3:23
# @Author  : Lart Pang
# @FileName: CARAFE.py
# @Project : Paper_Code
# @GitHub  : https://github.com/lartpang

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(inC, inC // 4, 1)
        self.encoder = nn.Conv2d(inC // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(inC, outC, 1)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # N, Cm, H, W
        kernel_tensor = self.encoder(kernel_tensor)  # N, S^2 * Kup^2, H, W
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # N, Kup^2, S*H, S*W
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # N, Kup^2, S*H, S*W
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # N, H, W, Kup^2, S^2

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1)
        in_tensor = in_tensor.unfold(3, self.kernel_size, step=1)
        in_tensor = in_tensor.reshape(N, C, H, W, -1)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # N, H, W, C, Kup^2

        out_tensor = torch.matmul(in_tensor, kernel_tensor)  # N, H, W, C, S^2
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        return out_tensor


if __name__ == '__main__':
    a = torch.rand(4, 20, 10, 10)
    sub = CARAFE(20)
    print(sub(a).size())
