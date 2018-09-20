# EE838 Special Topics on Image Engineering <Image Restoration and Quality Enhancement>
# Semester: Fall 2018, School of EE, KAIST
# Student: Tung M. Luu
# ----------------------------- Homework Assignment 1 -----------------------------
# ---- Implementation and Verification of Single Image Super-Resolution (SISR) ----

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, n_features, kernel_size, stride=1, bias=True, act=None):
        super(ResidualBlock, self).__init__()
        body = []
        body.append(nn.Conv2d(in_channels=n_features,
                              out_channels=n_features,
                              kernel_size=kernel_size,
                              padding=(kernel_size // 2),
                              stride=stride,
                              bias=bias))
        if act is not None:
            body.append(act)
            body.append(nn.Conv2d(in_channels=n_features,
                                  out_channels=n_features,
                                  kernel_size=kernel_size,
                                  padding=(kernel_size // 2),
                                  stride=stride,
                                  bias=bias))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)  # Feed to residual block
        x = res + x     # local skip connection
        return x


class Model_SR(nn.Module):
    def __init__(self, channel_in, opt_head, opt_body, opt_tail):
        super(Model_SR, self).__init__()
        self.channel_in = channel_in
        self.opt_head = opt_head  # keys: 'out_channels', 'kernel_size', 'stride'
        self.opt_body = opt_body    # keys: 'n_resblocks', 'n_features', 'kernel_size', 'stride', 'bias'
        self.opt_tail = opt_tail  # keys: 'out_channels', 'kernel_size', 'stride', 'upscale_factor', 'final_kernel_size',
                                  #       'final_out_channels', 'final_stride'

        # ------ Head module ------
        head = []
        head.append(nn.Conv2d(in_channels=self.channel_in,
                              out_channels=self.opt_head['out_channels'],
                              kernel_size=self.opt_head['kernel_size'],
                              padding=(self.opt_head['kernel_size'] // 2),
                              stride=self.opt_head['stride']))
        head.append(nn.ReLU())

        # ------ Body module ------
        body = []
        for i in range(self.opt_body['n_resblocks']):
            body.append(ResidualBlock(n_features=self.opt_body['n_features'],
                                      kernel_size=self.opt_body['kernel_size'],
                                      stride=self.opt_body['stride'],
                                      bias=self.opt_body['bias'],
                                      act=nn.ReLU()))
        body.append(nn.Conv2d(in_channels=self.opt_body['n_features'],
                              out_channels=64,
                              kernel_size=3,
                              padding=(3//2),
                              stride=1))

        # ------ Tail module ------
        tail = []
        tail.append(nn.Conv2d(in_channels=64,
                              out_channels=self.opt_tail['out_channels'],
                              kernel_size=self.opt_tail['kernel_size'],
                              padding=(self.opt_tail['kernel_size'] // 2),
                              stride=self.opt_tail['stride']))
        tail.append(nn.PixelShuffle(self.opt_tail['upscale_factor']))
        tail.append(nn.ReLU())
        tail.append(nn.Conv2d(in_channels=64,
                              out_channels=self.opt_tail['final_out_channels'],
                              kernel_size=self.opt_tail['final_kernel_size'],
                              padding=(self.opt_tail['final_kernel_size'] // 2),
                              stride=self.opt_tail['final_stride']))

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x   # global skip connection
        x = self.tail(res)
        return x