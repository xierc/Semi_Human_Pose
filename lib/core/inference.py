# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from utils.transforms import transform_preds

def get_max_preds_tensor(batch_heatmaps):
    '''
    get predictions from score maps
    '''

    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    maxvals, idx = torch.max(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((batch_size, num_joints))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = idx.repeat( [1, 1, 2]).float()

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    return preds, maxvals

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def soft_argmax(heatmaps):
    heatmaps = torch.from_numpy(heatmaps).cuda()
    assert isinstance(heatmaps, torch.Tensor)
    batch_size = heatmaps.size(0)
    num_joints = heatmaps.size(1)
    height = heatmaps.size(2)
    width = heatmaps.size(3)
    heatmaps = heatmaps.reshape((-1, num_joints, height*width))
    heatmaps = F.softmax(heatmaps, 2)
    maxv,_ = torch.max(heatmaps, 2, keepdim=True)
    heatmaps = heatmaps.reshape((-1, num_joints, height, width))

    accu_x = heatmaps.sum(dim=(2))
    accu_y = heatmaps.sum(dim=(3))

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1,width+1).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1,height+1).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
 
    accu_x = accu_x.sum(dim=2, keepdim=True) -1
    accu_y = accu_y.sum(dim=2, keepdim=True) -1

    coord_out = torch.cat((accu_x, accu_y), dim=2)

    return coord_out.cpu().numpy(),maxv.cpu().numpy()
    
def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
            + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx],[dy]])
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i,j])
            dr = np.zeros((height + 2 * border, width + 2 * border))
            dr[border: -border, border: -border] = hm[i,j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[i,j] = dr[border: -border, border: -border].copy()
            hm[i,j] *= origin_max / np.max(hm[i,j])
    return hm

def get_final_preds(config, batch_heatmaps, center, scale, rotation, hm_type='gaussian'):
    if hm_type == 'gaussian':
        coords, maxvals = get_max_preds(batch_heatmaps)
    else:
        coords, maxvals = soft_argmax(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    ## Dark Distribution Aware Representation of Keypoint
    if config.TEST.POST_PROCESS and config.TEST.POST_TYPE == 'dark':
        config.TEST.BLUR_KERNEL = 11
        hm = batch_heatmaps
        hm = gaussian_blur(hm, config.TEST.BLUR_KERNEL)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n,p] = taylor(hm[n][p], coords[n][p])

    ##  post-processing
    if config.TEST.POST_PROCESS and config.TEST.POST_TYPE == 'shift':
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # if flip:
    #     coords[:, :, 0] = coords[:, :, 0] + 0.75

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i], rotation[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals
