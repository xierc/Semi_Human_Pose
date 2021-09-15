# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import os
import json_tricks as json

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class MPIIDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        print('db')
        self.actual_joints = {
            0: 'rank',
            1: 'rkne',
            2: 'rhip',
            3: 'lhip',
            4: 'lkne',
            5: 'lank',
            6: 'root',
            7: 'thorax',
            8: 'upper neck',
            9: 'head top',
            10: 'rwri',
            11: 'relb',
            12: 'rsho',
            13: 'lsho',
            14: 'lelb',
            15: 'lwri'
        }
        # self.num_joints = 16
        # self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.flip_pairs = [[1,4],[2,5],[3,6],[14,17],[15,18],[16,19],[20,21],[22,23]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)
            
        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

        logger.info('=> load {} samples'.format(len(self.db)))
        

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(self.root,
                                 'mpii',
                                 'annot',
                                 self.image_set+'.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        # Load Box file
        # box_f  = os.path.join(self.root,
        #                          'mpii',
        #                          'annot',
        #                          'MPHB_bbox.npy') 
        # box_dict = np.load(box_f, allow_pickle=True).item()

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((16, 3), dtype=np.float)
            joints_3d_vis = np.zeros((16,  3), dtype=np.float)

            bbox =  np.array([[0, 0],[0, 0],[0, 0],[0, 0]])
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                # assert len(joints) == self.num_joints, \
                #     'joint num diff: {} vs {}'.format(len(joints),
                #                                       self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            # Generate boundingbox for MPII
            # if image_name in box_dict.keys():
            #     # Use MPHB dataset
            #     boxes = box_dict[image_name].reshape(-1,2,2)
            #     boxes_center = np.mean( boxes ,1)
            #     box_dis = np.linalg.norm(boxes_center - c,axis = 1) 
            #     box_id = np.argmin(box_dis)
            #     bbox = boxes[box_id]

            #     x1,y1,x2,y2 = bbox.reshape(-1)
            #     bbox =  np.array([[x1, y1],[x1, y2],[x2, y2],[x2, y1]])
            # else:

                # Use joint to generate box
                joint = joints_3d[:,:2]
                joint = joint[joint[:,0]>0]

                x1,y1 = np.min(joint, 0)
                x2,y2 = np.max(joint, 0)
                # Scale to save person
                center = np.mean([[x1,y1],[x2,y2]],0)
                w = (x2-x1) * 1.2
                h = (y2-y1) * 1.1

                x1,y1 = center - np.array([w/2 , h/2])
                x2,y2 = center + np.array([w/2 , h/2])

                bbox =  np.array([[x1, y1],[x1, y2],[x2, y2],[x2, y1]])

            image_dir = 'images.zip@/images' if self.data_format == 'zip' else 'images'
            
            gt_db.append({
                'image': os.path.join(self.root,'mpii', image_dir, image_name),
                'center': c,
                'scale': s,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                'bbox': bbox,
                })
        print('box')
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds[:, :, 0:2] = preds[:, :, 0:2] + 1.0

        # mapping
        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))
        # Union to dataset
        preds = preds[:, su]
        
        if output_dir:
            if args[2]:
                pred_file = os.path.join(output_dir, args[2])
            else:
                pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        preds = preds[:, :, 0:2]

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'mpii',
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio)),
            ('Mean@0.2', np.sum(pckAll[20, :] * jnt_ratio)),
            ('Mean@0.3', np.sum(pckAll[30, :] * jnt_ratio)),
            ('Mean@0.4', np.sum(pckAll[40, :] * jnt_ratio)),
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

    def compute(self, preds, gt_dict):
        SC_BIAS = 0.6
        threshold = 0.5
        
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']
        pos_pred_src = np.transpose(preds, [1, 2, 0])

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                        jnt_visible)
        return less_than_threshold,jnt_count,scaled_uv_err,jnt_visible

    def combine_evaluate(self, cfg, preds1, preds2, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'mpii',
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))

        # Mapping
        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))
        # Union to dataset
        preds1 = preds1[:, :, 0:2] + 1.0
        preds1 = preds1[:, su]

        preds2 = preds2[:, :, 0:2] + 1.0
        preds2 = preds2[:, su]

        gt_dict = loadmat(gt_file)
        less_than_threshold1,jnt_count,scaled_uv_err1,jnt_visible = self.compute(preds1, gt_dict)
        less_than_threshold2,jnt_count,scaled_uv_err2,jnt_visible = self.compute(preds2, gt_dict)
        threshold = 0.5
        less_than_threshold1 = np.multiply((scaled_uv_err1 <= threshold),
                                        jnt_visible)
        less_than_threshold2 = np.multiply((scaled_uv_err2 <= threshold),
                                        jnt_visible)
        less_than_threshold = less_than_threshold1|less_than_threshold2
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold1 = np.multiply(scaled_uv_err1 <= threshold,
                                            jnt_visible)
            less_than_threshold2 = np.multiply(scaled_uv_err2 <= threshold,
                                            jnt_visible)
            less_than_threshold = less_than_threshold1|less_than_threshold2
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                    jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        dataset_joints = gt_dict['dataset_joints']
        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio)),
            ('Mean@0.2', np.sum(pckAll[20, :] * jnt_ratio)),
            ('Mean@0.3', np.sum(pckAll[30, :] * jnt_ratio)),
            ('Mean@0.4', np.sum(pckAll[40, :] * jnt_ratio)),
        ]
        name_value = OrderedDict(name_value)
        return name_value, name_value['Mean']