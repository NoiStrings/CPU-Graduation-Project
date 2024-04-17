# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qC5A7I1bJPn5LVUnjdpTyQWjS7JHxsoF
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from einops import rearrange
from easydict import EasyDict

config = EasyDict()
config.angRes = 9
config.dispMin = -4
config.dispMax = 4
config.device = 'WTF'


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.angRes = config.angRes
        self.device = config.device
        self.feature_extracion = Feature_Extraction(config)
        self.mask_generator = Mask_Generator(config)
        self.cost_constructor = Cost_Constructor(config, channels_in = 8, channels_out = 512)
        self.cost_aggregator = Cost_Aggregator(config, channels_in = 512)
    def forward(self, lf, dispGT = None):
        # lf.shape = (b c (u h) (v w)), c = 1
        # dispGT.shape = (b c h w), c = 2
        lf = rearrange(lf, "b c (u h) (v w) -> b c u v h w",
                       u = self.angRes, v = self.angRes)
        _, _, u, v, h, w = lf.shape

        feature_map = self.feature_extracion(lf)
        # feature_map.shape = (b c (u v) h w), c = 8

        if dispGT is not None:
            mask = self.mask_generator(lf, dispGT)
            # mask.shape = (b c h w), c = 9 * 9
            cost = self.cost_constructor(feature_map, mask)
            # cost.shape = (b c d h w), c = 512, d = num of candidate disp = 9
            dispPred = self.cost_aggregator(cost)
            # dispPred.shape = (b c h w), c = 1
        else:
            init_mask = torch.ones(1, u * v, h, w).to(self.device)
            init_cost = self.cost_constructor(feature_map, init_mask)
            dispInit = self.cost_aggregator(init_cost)
            mask = self.mask_generator(lf, dispInit)
            cost = self.cost_constructor(feature_map, mask)
            dispPred = self.cost_aggregator(cost)
        return dispPred

class Feature_Extraction(nn.Module):
    def __init__(self):
        super(Feature_Extraction, self).__init__()
        self.initial_extraction = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size = (1,3,3),
                      stride = 1, padding = (0, 1, 1), bias = False),
            nn.BatchNorm3d(16)
            )
        self.deep_extraction = nn.Sequential(
            ResBlock(16), ResBlock(16),
            ResBlock(16), ResBlock(16),
            ResBlock(16), ResBlock(16),
            nn.Conv3d(16, 16, kernel_size = (1, 3, 3),
                      stride = 1, padding = (0, 1, 1), bias = False),
            nn.BatchNorm3d(16), nn.LeakyReLU(0.1, inplace = True),
            nn.Conv3d(16, 8, kernel_size = (1, 3, 3),
                      stride = 1, padding = (0, 1, 1), bias = False),
            nn.BatchNorm3d(8), nn.LeakyReLU(0.1, inplace = True)
            )
        self.final_conv = nn.Conv3d(8, 8, kernel_size = (1, 3, 3),
                                    stride = 1, padding = (0, 1, 1), bias = False)
    def forward(self, lf):
        # lf.shape = (b c u v h w), c = 1
        lf = rearrange(lf, "b c u v h w -> b c (u v) h w")
        feature_init = self.initial_extraction(lf)
        feature_deep = self.deep_extraction(feature_init)
        feature_out = self.final_conv(feature_deep)
        # feature_out.shape = (b c (u v) h w), c = 8
        return feature_out

class ResBlock(nn.Module):
    def __init__(self, numChannels):
        super(ResBlock, self).__init__()
        self.numChannels = numChannels
        self.conv = nn.Sequential(
            nn.Conv3d(numChannels, numChannels, kernel_size = (1, 3, 3),
                      stride = 1, padding = (0, 1, 1), bias = False),
            nn.BatchNorm3d(numChannels),
            nn.LeakyReLU(0.1, inplace = True)
            )
    def forward(self, feature_in):
        # feature_in.shape = (b c (u v) h w), c = 16
        feature_out = self.conv(feature_in)
        # feature_out.shape = feature_in.shape
        return feature_in + feature_out

class Mask_Generator:
    def __init__(self, config):
        self.device = config.device
    def __call__(self, lf, disp):
        # lf.shape = (b c u v h w), c = 1
        # disp.shape = (b c h w), c = 2
        _, _, u, v, h, w = lf.shape
        x_base = torch.linspace(0, 1, w).repeat(h, 1).to(self.device)
        y_base = torch.linspace(0, 1, h).repeat(w, 1).transpose(0, 1).to(self.device)
        center_u = (u - 1) // 2
        center_v = (v - 1) // 2
        view_center = lf[:, :, center_u, center_v, :, :]
        views_residual = []
        for i_u in range(u):
            for i_v in range(v):
                view = lf[:, :, i_u, i_v, :, :]
                if i_u == center_u and i_v == center_v:
                    view_warped = view
                else:
                    du = i_u - center_u
                    dv = i_v - center_v
                    view_warped = self.ViewWarp(lf, disp, view, x_base ,y_base, du, dv)
                view_residual = abs(view_warped - view_center)
                view_residual = (view_residual - view_residual.min()) / (view_residual.max() - view_residual.min())
                views_residual.append(view_residual)
        mask = torch.cat(views_residual, dim=1)
        mask = (1 - mask) ** 2
        # mask.shape = (b c h w), c = 9 * 9
        return mask
    def ViewWarp(self, lf, disp, view, x_base ,y_base, du, dv):
        # lf.shape = (b c u v h w), c = 1
        # disp.shape = (b c h w), c = 2
        # view.shape = (b c h w), c = 1
        _, _, u, v, h, w = lf.shape
        x_base = x_base.unsqueeze(0)
        y_base = y_base.unsqueeze(0)
        x_shifted = x_base + dv * disp[:, 0, :, :] / w
        y_shifted = y_base + du * disp[:, 0, :, :] / h
        sample_axis = torch.stack((x_shifted, y_shifted), dim = 3)
        view_warped = F.grid_sample(view, 2 * sample_axis - 1, mode='bilinear', padding_mode='zeros')
        # view_warped.shape = (b c h w), c = 1
        return view_warped

class Cost_Constructor(nn.Module):
    def __init__(self, config, channels_in, channels_out):
        super(Cost_Constructor, self).__init__()
        self.dispMax = config.dispMax
        self.dispMin = config.dispMin
        self.angRes = config.angRes
        self.modulator = Feature_Modulator(kernel_size = self.angRes,
                                           channels_in = channels_in, channels_out = channels_out)
    def forward(self, feature_map, mask):
        # feature_map.shape = (b c (u v) h w), c = 8
        # mask.shape = (b c h w), c = 9 * 9
        bdr = (self.angRes // 2) * self.dispMax
        getPad = nn.ZeroPad2d((bdr, bdr, bdr, bdr))
        feature_map = getPad(feature_map)
        _, _, _, h_padded, w_padded = feature_map.shape
        feature_map = rearrange(feature_map, "b c (u v) h w -> b c (u h) (v w)",
                                    u = self.angRes, v = self.angRes)
        mask_avg = torch.mean(mask, dim=1)
        costs = []
        for disp_i in range(self.dispMin, self.dispMax + 1):
            self.modulator.dilation_rate = h_padded - disp_i
            if disp_i != self.dispMin:
                crop = (self.angRes // 2) * (disp_i - self.dispMin)
                feature_map = feature_map[:, :, crop: -crop, crop: -crop]
            cost_i = self.modulator(feature_map, mask)
            # cost_i.shape = (b c h w), c = 512
            costs.append(cost_i // mask_avg.unsqueeze(1).repeat(1, cost_i.shape[1], 1, 1))
        cost = torch.stack(costs, dim = 2)
        # cost.shape = (b c d h w), c = 512, d = num of candidate disp = dispMax - dispMin + 1 = 9
        return cost

class Feature_Modulator(nn.Module):
    def __init__(self, kernel_size, channels_in, channels_out, dilation_rate = 1):
        super(Feature_Modulator, self).__init__()
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.maskUnfold = nn.Unfold(kernel_size=1, stride=1, dilation=1, padding=0)
        self.getPatches = nn.Unfold(kernel_size=self.kernel_size, stride=1, dilation=self.dilation_rate)
        self.finalConv = nn.Conv2d(in_channels = channels_in * kernel_size * kernel_size,
                                   out_channels = channels_out,
                                   kernel_size=1, stride=1, padding=0, bias=False, groups=channels_in)
    def forward(self, feature_map, mask):
        # feature_map.shape = (b c (u hp) (v wp)), c = 8, hp = h_padded, wp = w_padded
        # mask.shape = (b c h w), c = 9 * 9
        angular_patches = self.getPatches(feature_map)
        mask_unfold = self.maskUnfold(mask)
        # angular_patches.shape = (b (c k^2) l), c = 8, k = kernel_size = 9, l = num of areas covered by kernel = 512 * 512
        # mask_unfold.shape = (b (c k^2) l), c = 9 * 9, k = 1, l = 512 * 512
        angular_patches_modulated = angular_patches * mask_unfold.repeat(1, feature_map.shape[1], 1)
        # angular_patches_modulated.shape = (b c l), c = 81 * 8, l = 512 * 512
        patchFold = nn.Fold(output_size=(mask.shape[2], mask.shape[3]), kernel_size=1, stride=1)
        feature_map_modulated = patchFold(angular_patches_modulated)
        # feature_map_modulated.shape = (b c h w), c = 81 * 8, h = w = 512
        cost = self.finalConv(feature_map_modulated)
        # cost.shape = (b c h w), c = 512
        return cost

class Cost_Aggregator(nn.Module):
    def __init__(self, config, channels_in):
        super(Cost_Aggregator, self).__init__()
        self.device = config.device
        self.dispMin = config.dispMin
        self.dispMax = config.dispMax
        self.dispRange = self.dispMax - self.dispMin + 1
        self.conv_pre = nn.Sequential(
            nn.Conv3d(in_channels = channels_in, out_channels = 160,
                      kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm3d(160),
            nn.LeakyReLU(0.1, inplace = True)
            )
        self.conv_1 = nn.Sequential(
            nn.Conv3d(160, 160, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm3d(160),
            nn.LeakyReLU(0.1, inplace = True)
            )
        self.conv_2 = nn.Sequential(
            nn.Conv3d(160, 160, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm3d(160),
            nn.LeakyReLU(0.1, inplace = True)
            )
        self.conv_3 = nn.Sequential(
            ResBlock_Att(160, self.dispRange),
            ResBlock_Att(160, self.dispRange)
            )
        self.conv_4 = nn.Sequential(
            nn.Conv3d(160, 160, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm3d(160),
            nn.LeakyReLU(0.1, inplace = True)
            )
        self.conv_final = nn.Sequential(
            nn.Conv3d(in_channels = 160, out_channels = 1,
                      kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.Softmax(dim = 1)
            )
    def forward(self, cost):
        # cost.shape = (b c d h w), c = 512, d = 9
        temp = self.conv_pre(cost)
        temp = self.conv_1(temp)
        temp = self.conv_2(temp)
        temp = self.conv_3(temp)
        temp = self.conv_4(temp)
        attMap = self.conv_final(temp)
        # AttMap.shape = (b c d h w), c = 1, d = 9
        attMap = attMap.squeeze() # dim c is squeezed
        # AttMap.shape = (b c h w), c = d = num of candidate disp = 9
        disp = torch.zeros(attMap.shape).to(self.device)
        for disp_i in range(self.dispMin, self.dispMax + 1):
            disp[:, disp_i, :, :] = attMap[:, disp_i, :, :] * disp_i
        dispMap = torch.sum(disp, dim=1, keepdim=True)
        # dispMap.shape = (b c h w), c = 1
        return dispMap

class ResBlock_Att(nn.Module):
    def __init__(self, channels_in, dispRange):
        super(ResBlock_Att, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels_in, channels_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm3d(channels_in),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv3d(channels_in, channels_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm3d(channels_in)
            )
        self.channelAttLayer = ChannelAttLayer(channels_in, dispRange)
    def forward(self, cost_in):
        # cost_in.shape = (b c d h w), c = 160, d = 9
        temp = self.conv(cost_in)
        cost_out = self.channelAttLayer(temp)
        # cost_out.shape = (b c d h w), c = 160, d = 9
        return cost_out + cost_in

class ChannelAttLayer(nn.Module):
    def __init__(self, channels_in, dispRange):
        super(ChannelAttLayer, self).__init__()
        self.pooling = nn.AdaptiveAvgPool3d((dispRange, 1, 1))
        self.getChannelAtt = nn.Sequential(
            nn.Conv3d(channels_in, 16, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv3d(16, channels_in, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm3d(channels_in),
            nn.Sigmoid()
            )
    def forward(self, cost_in):
        # cost_in.shape = (b c d h w), c = 160, d = 9
        temp = self.pooling(cost_in)
        channelAtt = self.getChannelAtt(temp)
        # channelAtt.shape = (b c d hp wp), c = 160, d = 9, hp = h_pooled = 1, wp = w_pooled = 1
        cost_out = cost_in * channelAtt
        # cost_out.shape = (b c d h w), c = 160, d = 9, h = w = 512
        return cost_out