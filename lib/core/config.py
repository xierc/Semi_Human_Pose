# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

config.LOCAL_RANK = 0

# Dual Networks
## Affine Transformation
config.UNSUP_TRANSFORM = True
## RandAugment
config.CONS_RAND_AUG = False
config.RAND_MAGNITUDE = 10
## Joint Cutout
config.MASK_JOINT_NUM = 0

config.RAW_INPUT = False

config.SAVE_INTER = 0

config.OUTPUT_DIR = ''
config.LOG_DIR = ''
config.DATA_DIR = ''
config.GPUS = '0'
config.WORKERS = 4
config.PRINT_FREQ = 20

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = False
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# pose_resnet related params
POSE_RESNET = edict()
POSE_RESNET.NUM_LAYERS = 50
POSE_RESNET.DECONV_WITH_BIAS = False
POSE_RESNET.NUM_DECONV_LAYERS = 3
POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
POSE_RESNET.FINAL_CONV_KERNEL = 1
POSE_RESNET.TARGET_TYPE = 'gaussian'
POSE_RESNET.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
POSE_RESNET.SIGMA = 2


# pose_multi_resoluton_net related params
POSE_HIGH_RESOLUTION_NET = edict()
POSE_HIGH_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
POSE_HIGH_RESOLUTION_NET.STEM_INPLANES = 64
POSE_HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL = 1

POSE_HIGH_RESOLUTION_NET.STAGE2 = edict()
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [32, 64]
POSE_HIGH_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'

POSE_HIGH_RESOLUTION_NET.STAGE3 = edict()
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [32, 64, 128]
POSE_HIGH_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'

POSE_HIGH_RESOLUTION_NET.STAGE4 = edict()
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
POSE_HIGH_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'

MODEL_EXTRAS = {
    'pose_resnet': POSE_RESNET,
    'pose_high_resolution_net': POSE_HIGH_RESOLUTION_NET,
}

# common params for NETWORK
config.MODEL = edict()
# Only Used for SSL
config.MODEL.BACKBONE = 'resnet'
config.MODEL.PRM = False

config.MODEL.NAME = 'pose_resnet'
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED = ''
config.MODEL.TCH_PRETRAINED = ''
config.MODEL.SEG_PRETRAINED = ''
config.MODEL.NUM_JOINTS = 16
config.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
config.MODEL.GROUPS = 1
config.MODEL.WIDTH_PER_GROUP = 64
config.MODEL.EXTRA = MODEL_EXTRAS[config.MODEL.NAME]

config.MODEL.STYLE = 'pytorch'

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.PSEUDO_ANNO = ''
config.DATASET.TRAIN_DATASET = 'mpii'
config.DATASET.TEST_DATASET = 'mpii'
config.DATASET.TRAIN_SET = 'train'
config.DATASET.TRAIN_UNSUP_SET = ''
config.DATASET.TEST_SET = 'valid'
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.HYBRID_JOINTS_TYPE = ''
config.DATASET.SELECT_DATA = False
config.DATASET.TRAIN_LEN = -1
config.DATASET.AI_ROOT = ''
config.DATASET.BOX_THRE = [0, 0]
config.DATASET.SCORE_THRE = [0]
config.DATASET.COLOR_RGB = False
config.DATASET.PROB_HALF_BODY = 0.0
config.DATASET.NUM_JOINTS_HALF_BODY = 8
# training data augmentation
config.DATASET.FLIP = True
config.DATASET.SCALE_FACTOR = 0.25
config.DATASET.ROT_FACTOR = 30

# train
config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.TRAIN.RESUME = True
config.TRAIN.CHECKPOINT = ''

config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True

# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE = 32
# Test Model Epoch
config.TEST.INTERVAL = 1
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True
config.TEST.POST_TYPE = 'shift' # or dark

config.TEST.USE_GT_BBOX = False
# nms
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MODEL_FILE = ''
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0
config.TEST.SAVE_RESULT = True

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = False
config.DEBUG.SAVE_BATCH_IMAGES_GT = False
config.DEBUG.SAVE_BATCH_IMAGES_PRED = False
config.DEBUG.SAVE_HEATMAPS_GT = False
config.DEBUG.SAVE_HEATMAPS_PRED = False


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    if config.DATA_DIR:
        config.DATASET.ROOT = os.path.join(
                config.DATA_DIR, config.DATASET.ROOT)

        if config.DATASET.AI_ROOT:
            config.DATASET.AI_ROOT = os.path.join(
                config.DATA_DIR, config.DATASET.ROOT)

        if config.TEST.COCO_BBOX_FILE:
            config.TEST.COCO_BBOX_FILE = os.path.join(
                config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

        if config.TEST.MODEL_FILE:
            config.TEST.MODEL_FILE = os.path.join(
                config.DATA_DIR, config.TEST.MODEL_FILE)

        if config.MODEL.PRETRAINED:
            config.MODEL.PRETRAINED = os.path.join(
                config.DATA_DIR, config.MODEL.PRETRAINED)

        if config.MODEL.TCH_PRETRAINED:
            config.MODEL.TCH_PRETRAINED = os.path.join(
                config.DATA_DIR, config.MODEL.TCH_PRETRAINED)

        if config.MODEL.SEG_PRETRAINED:
            config.MODEL.SEG_PRETRAINED = os.path.join(
                config.DATA_DIR, config.MODEL.SEG_PRETRAINED)

        if config.DATASET.PSEUDO_ANNO:
            config.DATASET.PSEUDO_ANNO = os.path.join(
                config.DATA_DIR, config.DATASET.PSEUDO_ANNO)
        # to mount 
        config.OUTPUT_DIR = os.path.join(
                config.DATA_DIR, config.OUTPUT_DIR)

def get_model_name(cfg):
    name = cfg.MODEL.NAME
    full_name = cfg.MODEL.NAME
    extra = cfg.MODEL.EXTRA
    if name in ['pose_resnet', 'pose_dual','pose_cons','pose_hrnet']:
        num_layers = extra.NUM_LAYERS  if "NUM_LAYERS" in extra else extra.STAGE2.NUM_CHANNELS[0]
        name = '{model}_{num_layers}'.format(
            model=name,
            num_layers=num_layers)
        deconv_suffix = ''.join(
            'd{}'.format(num_filters)
            for num_filters in extra.NUM_DECONV_FILTERS) if 'NUM_DECONV_FILTERS' in extra else ''
        full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
            height=cfg.MODEL.IMAGE_SIZE[1],
            width=cfg.MODEL.IMAGE_SIZE[0],
            name=name,
            deconv_suffix=deconv_suffix)
    else:
        raise ValueError('Unkown model: {}'.format(cfg.MODEL))

    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
