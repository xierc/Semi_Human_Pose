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
from dataset.mpii import MPIIDataset
from dataset.ai_challenger import AIDataset
from utils.rand_augment import RandAugment

class Mixed_MPII_AI_Dataset(JointsDataset):

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        # Labeled set
        self.mpii = MPIIDataset(cfg, root, image_set, is_train, transform)
        if cfg.RAW_INPUT:
            self.ai = AIDataset(cfg, root, image_set, False, transform)
        else:
            self.ai = AIDataset(cfg, root, image_set, is_train, transform)
        np.random.seed(1314)

        self.mpii_size = len(self.mpii.db)
        self.ai_size = len(self.ai.db)
        
        self.sets = [self.mpii, self.ai]
        self.sizes = [self.mpii_size, self.ai_size]

        self.group_size = max(self.sizes)
        self.cfg = cfg
        if cfg.CONS_RAND_AUG:
            N,M = 2,cfg.RAND_MAGNITUDE
            transform_aug = copy.deepcopy(transform)
            transform_aug.transforms.insert(0, RandAugment(N, M))
            self.aug_ai = AIDataset(cfg, root, image_set, is_train, transform_aug)
            self.sets.append(self.aug_ai)
            self.set_dic={1:2}

    def __len__(self):
        return self.group_size

    def __getitem__(self, idx):

        # Set the MPII joints as joint_vis of AI
        ai_joint_ind = [] 
        mapping = self.sets[0].u2a_mapping
        for i in range(len(mapping)):
            if mapping[i] != '*':
                ai_joint_ind.append(int(i))

        # Pre define the rotation angle and scale
        rf = self.sets[0].rotation_factor
        rot = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
            if random.random() <= 0.6 else 0

        sf = self.sets[0].scale_factor
        scale = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        flip = random.random() <= 0.5

        input, target, weight, meta = [], [], [], []
        for k, data in enumerate(self.sets[:2]):   
            # loop the small dataset to construct pair
            # Assume that labeled set is smaller
            if k==0:
                t_idx = np.random.choice(self.sizes[k], 1)[0]
            else:
                t_idx = idx

            # Get the same transform for unsup and unsup_aug
            if k==0:
                i, t, w, m = data.__getitem__(t_idx)
            else:
                i, t, w, m = data.__getitem__(t_idx, rot_angle = rot, scale = scale, flip=flip)

            input.append(i)
            
            if self.cfg.CONS_RAND_AUG and k in self.set_dic:
                ind = self.set_dic[k]
                aug_i, _, _, _  = self.sets[ind].__getitem__(t_idx, rot_angle = rot, scale = scale, flip=flip)
                input.append(aug_i)

            # Set all joint of labeled set are visible in unlabel set
            if k == 1 :
                w.zero_()
                w[ai_joint_ind, :] = 1
                m['joints_vis'] = meta[0]['joints_vis'].copy()
                m['joints_vis'][ai_joint_ind, :] = 1

            target.append(t)
            weight.append(w)
            meta.append(m)

        return input, target, weight, meta

