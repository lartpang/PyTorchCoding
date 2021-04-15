# -*- coding: utf-8 -*-
# @Time    : 2021/4/15
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvAttention(nn.Module):
  """
  https://github.com/mlpc-ucsd/CoaT
  
  @misc{xu2021coscale,
      title={Co-Scale Conv-Attentional Image Transformers}, 
      author={Weijian Xu and Yifan Xu and Tyler Chang and Zhuowen Tu},
      year={2021},
      eprint={2104.06399},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
  """
    def __init__(self, embedding_dim=64, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.pos_embed_dim = embedding_dim
        self.rel_pos_embed = nn.Parameter(torch.zeros(embedding_dim, window_size, window_size))
        nn.init.trunc_normal_(self.rel_pos_embed, std=0.02)

        self.qkv_conv = nn.Conv2d(embedding_dim, 3 * embedding_dim, 1, bias=False)

        # Conditional Positional Encodings for Vision Transformers
        self.cpe = nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1)

        self.out = nn.Conv2d(embedding_dim, embedding_dim, 1)

    def forward(self, src):
        """
        :param query: B,C,H,W
        :param key: B,C,H,W
        :param value: B,C,H,W
        :return: B,C,H,W
        """
        _, C, H, W = src.shape
        
        # Convolutional Position Encoding
        src = self.cpe(src) + src
        
        # Linear Projection
        qkv = self.qkv_conv(src)
        q, k, v = qkv.chunk(3, dim=1)

        # Factorized Attention
        content_lambda = torch.einsum("bin, bon -> bio", k.flatten(-2).softmax(-1), v.flatten(-2))
        content_output = torch.einsum("bin, bio -> bon", q.flatten(-2) * (C // 3) ** -0.5, content_lambda)
        content_output = content_output.unflatten(dim=-1, sizes=(H, W))
        
        # Convolutional Relative Position Encoding
        position_lambda = F.conv2d(
            v,
            weight=rearrange(self.rel_pos_embed, "D Mx My -> D 1 Mx My"),
            padding=self.window_size // 2,
            groups=self.pos_embed_dim,
        )
        position_output = q * position_lambda
        
        # Output Feature Map
        result = content_output + position_output
        return self.out(result)


if __name__ == "__main__":
    src = torch.randn(1, 64, 32, 32)

    conv_attention = ConvAttention()
    print(conv_attention(src=src).shape)
