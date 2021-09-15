# ------------------------------------------------------------------------------
# Written by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import copy
from core.config import config
from dataset.JointsDataset import JointsDataset
from dataset.coco import COCODataset
from utils.rand_augment import RandAugment
import logging
logger = logging.getLogger(__name__)

class Mixed_COCO_COCO_Dataset(JointsDataset):

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        transform_sup = transform  
        self.coco = COCODataset(cfg, root, image_set, is_train, transform_sup)

        # Set Length as Full
        cfg.DATASET.TRAIN_LEN = 0
        # If set unsup set specially
        if cfg.DATASET.TRAIN_UNSUP_SET:
            image_set = cfg.DATASET.TRAIN_UNSUP_SET

        if cfg.RAW_INPUT:
            self.ai = COCODataset(cfg, root, image_set, False, transform)
        else:
            self.ai = COCODataset(cfg, root, image_set, is_train, transform)

        np.random.seed(1314)

        self.ai_size = len(self.ai.db)
        self.coco_size = len(self.coco.db)
        self.sets = [self.coco, self.ai]
        self.sizes = [self.coco_size,self.ai_size]

        self.group_size = max(self.sizes)
        self.cfg = cfg
        self.shuffle_ind()

        if cfg.CONS_RAND_AUG :
            N,M = 2,cfg.RAND_MAGNITUDE
            transform_aug = copy.deepcopy(transform)
            if  transform.transforms[0].__class__.__name__ != 'RandAugment':
                transform_aug.transforms.insert(0, RandAugment(N, M))
            self.aug_ai = COCODataset(cfg, root, image_set, is_train, transform_aug)
            self.sets.append(self.aug_ai)

        logger.info('=> Total load {} unlabeled samples'.format(len(self.ai.db)))

    def __len__(self):
        return self.group_size

    def shuffle_ind(self):
        self.data_ind = np.random.choice((self.sizes)[0], self.group_size)
        return None

    def __getitem__(self, idx):

        # Set the COCO joints as joints vis of AI
        ai_joint_ind = [] 
        mapping = self.sets[0].u2a_mapping
        for i in range(len(mapping)):
            if mapping[i] != '*':
                ai_joint_ind.append(int(i))

        # Pre define the rotation angle, scale and flip
        rf = self.sets[0].rotation_factor
        rot = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
            if random.random() <= 0.6 else 0
        sf = self.sets[0].scale_factor
        scale = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        flip = random.random() <= 0.5

        input, target, weight, meta = [], [], [], []

        for k, data in enumerate(self.sets):   
            # loop the small dataset to construct pair
            if k==0:
                t_idx = self.data_ind[idx]
            else:
                t_idx = idx

            # Get the same transform for unsup and unsup_aug
            if k==0:
                i, t, w, m = data.__getitem__(t_idx)
            else:
                i, t, w, m = data.__getitem__(t_idx, rot_angle = rot, scale = scale, flip=flip)
                
            input.append(i)

            # For unsup dataset, set full vis  
            if k != 0:
                w.zero_()
                w[ai_joint_ind, :] = 1
                m['joints_vis'] = meta[0]['joints_vis'].copy()
                m['joints_vis'][ai_joint_ind, :] = 1

            if k<=1:
                target.append(t)
                weight.append(w)
                meta.append(m)


        return input, target, weight, meta

