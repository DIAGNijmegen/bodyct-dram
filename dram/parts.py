import torch.nn as nn
import torch
import numpy as np
import functools
import torch.nn.functional as F
import math

from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, args=None):
        return x

def normal_wrapper(normal_method, in_ch, in_ch_div=2):
    if normal_method is "bn":
        return nn.BatchNorm3d(in_ch)
    elif normal_method is "bnt":
        # this should be used when batch_size=1
        return nn.BatchNorm3d(in_ch, affine=True, track_running_stats=False)
    elif normal_method is "bntna":
        # this should be used when batch_size=1
        return nn.BatchNorm3d(in_ch, affine=False, track_running_stats=False)
    elif normal_method is "ln":
        return nn.GroupNorm(1, in_ch)
    elif normal_method is "lnna":
        return nn.GroupNorm(1, in_ch, affine=False)
    elif normal_method is "in":
        return nn.GroupNorm(in_ch, in_ch)
    elif normal_method is "sbn":
        return nn.SyncBatchNorm(in_ch)
    else:
        return Identity()

def crop_concat_5d(t1, t2):
    """"Channel-wise cropping for 5-d tensors in NCDHW format,
    assuming t1 is smaller than t2 in all DHW dimension. """
    assert (t1.dim() == t2.dim() == 5)
    assert (t1.shape[-1] <= t2.shape[-1])
    slices = (slice(None, None), slice(None, None)) \
             + tuple(
        [slice(int(np.ceil((b - a) / 2)), a + int(np.ceil((b - a) / 2))) for a, b in zip(t1.shape[2:], t2.shape[2:])])
    x = torch.cat([t1, t2[slices]], dim=1)
    return x

def act_wrapper(act_method, num_parameters=1, init=0.25):
    if act_method is "relu":
        return nn.ReLU(inplace=True)
    elif act_method is "prelu":
        return nn.PReLU(num_parameters, init)
    else:
        raise NotImplementedError


def checkpoint_wrapper(module, segments, *tensors):
    if segments > 0:
        # if type(module) in [nn.Sequential, nn.ModuleList, list]:
        #     return checkpoint_sequential(module, segments, *tensors)
        # else:
        return checkpoint(module, *tensors)
    else:
        return module(*tensors)

class ConvBlock5d(nn.Module):

    def __init__(self, in_chs, base_chs, checkpoint_segments, conv_ksize,
                 conv_bias, conv_pad, dropout=0.1, conv_strides=1,
                 norm_method='bn', act_methpd='relu', lite=False,
                 **kwargs):
        super(ConvBlock5d, self).__init__()
        if not isinstance(conv_ksize, (tuple, list)):
            conv_ksize = [conv_ksize] * len(in_chs)

        if not isinstance(conv_pad, (tuple, list)):
            conv_pad = [conv_pad] * len(in_chs)

        if not isinstance(conv_strides, (tuple, list)):
            conv_strides = [conv_strides] * len(in_chs)

        if lite:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx],
                              bias=conv_bias, stride=conv_stride),
                    act_wrapper(act_methpd),
                ) for idx, (in_ch, base_ch, conv_stride) in enumerate(zip(in_chs, base_chs, conv_strides))
            ])
        else:
            if dropout > 0:
                print("use dropout in convs!")
                self.conv_blocks = nn.Sequential(*[
                    nn.Sequential(
                        nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx],
                                  bias=conv_bias, stride=conv_stride),
                        normal_wrapper(norm_method, base_ch),
                        act_wrapper(act_methpd),
                        nn.Dropout(dropout),
                    ) for idx, (in_ch, base_ch, conv_stride) in enumerate(zip(in_chs, base_chs, conv_strides))
                ])
            else:
                self.conv_blocks = nn.Sequential(*[
                    nn.Sequential(
                        nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx],
                                  bias=conv_bias, stride=conv_stride),
                        normal_wrapper(norm_method, base_ch),
                        act_wrapper(act_methpd),
                    ) for idx, (in_ch, base_ch, conv_stride) in enumerate(zip(in_chs, base_chs, conv_strides))
                ])

    def forward(self, x, args=None):
        return self.conv_blocks(x)


class UpsampleConvBlock5d(nn.Module):

    def __init__(self, in_chs, base_chs, checkpoint_segments, scale_factor,
                 conv_ksize, conv_bias, conv_pad, dropout=0.1,
                 norm_method='bn', act_methpd='relu', **kwargs):
        super(UpsampleConvBlock5d, self).__init__()
        self.checkpoint_segments = checkpoint_segments
        self.scale_factor = scale_factor
        if not isinstance(conv_ksize, (tuple, list)):
            conv_ksize = [conv_ksize] * len(in_chs)

        if not isinstance(conv_pad, (tuple, list)):
            conv_pad = [conv_pad] * len(in_chs)

        if dropout > 0:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx], bias=conv_bias),
                    normal_wrapper(norm_method, base_ch),
                    act_wrapper(act_methpd),
                    nn.Dropout(dropout)
                ) for idx, (in_ch, base_ch) in enumerate(zip(in_chs, base_chs))
            ])
        else:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx], bias=conv_bias),
                    normal_wrapper(norm_method, base_ch),
                    act_wrapper(act_methpd),
                ) for idx, (in_ch, base_ch) in enumerate(zip(in_chs, base_chs))
            ])

        self.merge_func = kwargs.get('merge_func', crop_concat_5d)
        self.upsample = nn.Upsample(size=None, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)

    def forward(self, inputs, cats, args=None):
        up_inputs = self.upsample(inputs)
        x = crop_concat_5d(up_inputs, cats)
        x = self.conv_blocks(x)
        return x

class ConvPoolBlock5d(nn.Module):

    def __init__(self, in_ch_list, base_ch_list, checkpoint_segments,
                 conv_ksize, conv_bias, conv_pad,
                 pool_ksize, pool_strides, pool_pad, dropout=0.1,
                 conv_strdes=1, norm_method='bn', act_method="relu",
                 **kwargs):
        super(ConvPoolBlock5d, self).__init__()
        self.checkpoint_segments = checkpoint_segments
        if not isinstance(conv_ksize, (tuple, list)):
            conv_ksize = [conv_ksize] * len(in_ch_list)

        if not isinstance(conv_pad, (tuple, list)):
            conv_pad = [conv_pad] * len(in_ch_list)

        if not isinstance(conv_strdes, (tuple, list)):
            conv_strdes = [conv_strdes] * len(in_ch_list)

        if dropout > 0:
            self.conv_blocks = nn.Sequential(
                *[nn.Sequential(nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], stride=conv_strdes[idx],
                                          padding=conv_pad[idx], bias=conv_bias),
                                normal_wrapper(norm_method, base_ch),
                                act_wrapper(act_method),
                                nn.Dropout(dropout))
                  for idx, (in_ch, base_ch) in enumerate(zip(in_ch_list, base_ch_list))])
        else:
            self.conv_blocks = nn.Sequential(
                *[nn.Sequential(nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], stride=conv_strdes[idx],
                                          padding=conv_pad[idx], bias=conv_bias),
                                normal_wrapper(norm_method, base_ch),
                                act_wrapper(act_method)
                                )
                  for idx, (in_ch, base_ch) in enumerate(zip(in_ch_list, base_ch_list))])
        self.maxpool = nn.MaxPool3d(kernel_size=pool_ksize, stride=pool_strides, padding=pool_pad)

    def forward(self, x, args=None):
        y = self.conv_blocks(x)
        pooled = self.maxpool(y)
        return y, pooled