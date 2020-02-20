import torch
import torch.nn as nn
import torch.nn.functional as F


class DCM(nn.Module):
    def __init__(self, in_C, out_C):
        super(DCM, self).__init__()
        self.ks = [1, 3, 5]
        self.mid_C = in_C // 4
        
        self.ger_kernel_branches = nn.ModuleList()
        for k in self.ks:
            self.ger_kernel_branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(k),
                    nn.Conv2d(in_C, self.mid_C, kernel_size=1)
                )
            )

        self.trans_branches = nn.ModuleList()
        self.fuse_inside_branches = nn.ModuleList()
        for i in range(len(self.ks)):
            self.trans_branches.append(
                nn.Conv2d(in_C, self.mid_C, kernel_size=1)
            )
            self.fuse_inside_branches.append(
                nn.Conv2d(self.mid_C, self.mid_C, 1)
            )
        
        self.fuse_outside = nn.Conv2d(len(self.ks) * self.mid_C + in_C, out_C, 1)

    def forward(self, x, y):
        """
        x: 被卷积的特征
        y: 用来生成卷积核
        """
        feats_branches = [x]
        for i in range(len(self.ks)):
            kernel = self.ger_kernel_branches[i](y)
            kernel_single = kernel.split(1, dim=0)
            x_inside = self.trans_branches[i](x)
            x_inside_single = x_inside.split(1, dim=0)
            feat_single = []
            for kernel_single_item, x_inside_single_item \
                in zip(kernel_single, x_inside_single): 
                feat_inside_single = self.fuse_inside_branches[i](
                        F.conv2d(
                            x_inside_single_item,
                            weight=kernel_single_item.transpose(0, 1),
                            bias=None,
                            stride=1,
                            padding=self.ks[i]//2,
                            dilation=1,
                            groups=self.mid_C
                        )
                    )
                feat_single.append(feat_inside_single)
            feat_single = torch.cat(feat_single, dim=0)
            feats_branches.append(feat_single)
        return self.fuse_outside(torch.cat(feats_branches, dim=1))

x = torch.randn(4, 2048, 20, 20)
y = torch.randn(4, 2048, 20, 20)
dcm = DCM(in_C=2048, out_C=20)
print(dcm(x, y).size())
