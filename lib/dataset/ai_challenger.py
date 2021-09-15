# ------------------------------------------------------------------------------
# Written by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import json_tricks as json
import time
import pprint

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset
from nms.nms import oks_nms

logger = logging.getLogger(__name__)




class AIDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.actual_joints = {
            0: 'rsho',
            1: 'relb',
            2: 'rwri',
            3: 'lsho',
            4: 'lelb',
            5: 'lwri',
            6: 'rhip',
            7: 'rkne',
            8: 'rank',
            9: 'lhip',
            10: 'lkne',
            11: 'lank',
            12: 'head top',
            13: 'neck',
        }
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        
        
        # self.num_joints = 14
        self.actual_num_joints = len(self.actual_joints)
        # self.flip_pairs = [[0,3],[1,4],[2,5],[6,9],[7,10],[8,11]]
        self.flip_pairs = [[1,4],[2,5],[3,6],[14,17],[15,18],[16,19],[20,21],[22,23]]
        self.parent_ids = []

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

        print('=> load {} samples'.format(len(self.db)))
        
    def _get_db(self):
        # create train/val split
        file_name = os.path.join(self.root,'ai_challenger',self.image_set,
                                 'keypoint_'+self.image_set+'_annotation.json')
        self.anno_name = file_name
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        image_dir = 'images.zip@/images' if self.data_format == 'zip' else 'images'
        image_root = os.path.join(self.root,'ai_challenger',self.image_set,image_dir) 
        # image_root = os.path.join(self.root,'ai_challenger',self.image_set,'images')
        
        gt_db = []
        min_x1 = 1000
        min_y1 = 1000
        for img_anno in anno:
            image_name = os.path.join(image_root, img_anno['image_id'] +'.jpg')
            for key in img_anno['keypoint_annotations'].keys():
                # Box to center and scale
                bbox = img_anno['human_annotations'][key]
                x1, y1, x2, y2 = bbox

#                 x1 = np.max((0, x1))
#                 y1 = np.max((0, y1))
#                 x2 = np.min((width - 1, x2 ))
#                 y2 = np.min((height - 1, y2 ))
#                 if  x2 >= x1 and y2 >= y1:
                clean_bbox = [x1, y1, x2-x1, y2-y1]
                center, scale = self._box2cs(clean_bbox[:4])
                
                joints_3d = np.zeros((self.actual_num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.actual_num_joints,  3), dtype=np.float)
                if self.image_set != 'test':
                    joints = np.array(img_anno['keypoint_annotations'][key]).reshape(-1,3)
                    joints_vis = np.array(joints[:,2]<3)
        
                    assert len(joints) == self.actual_num_joints, \
                        'joint num diff: {} vs {}'.format(len(joints),
                                                          self.actual_num_joints)

                    joints_3d[:, 0:2] = joints[:, 0:2]
                    joints_3d_vis[:, 0] = joints_vis[:]
                    joints_3d_vis[:, 1] = joints_vis[:]

                    # joints_3d_vis[6,:] = 0
                    # joints_3d_vis[9,:] = 0
                # image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
                gt_db.append({
                    'image': image_name,
                    'center': center,
                    'scale': scale,
                    'bbox': np.array([[x1, y1],[x1, y2],[x2, y2],[x2, y1]]),
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    })

        return gt_db
    
    
    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale
    
    # PCKH
    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        #
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

        AP_dict, AP = self.evaluate_AP(cfg, preds, output_dir, all_boxes, img_path)

        # 
        preds = preds[:, :, 0:2] 

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 1
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'ai_challenger',
                               'validation',
                               'gt_validation.npy')
        gt_dict = np.load(gt_file, allow_pickle=True).item()
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']
        jnt_labeled = gt_dict['jnt_labeled']

        pos_pred_src = preds

        head = np.where(dataset_joints == 'head')[0][0]
        neck = np.where(dataset_joints == 'neck')[0][0]
        lsho = np.where(dataset_joints == 'lsho')[0][0]
        lelb = np.where(dataset_joints == 'lelb')[0][0]
        lwri = np.where(dataset_joints == 'lwri')[0][0]
        lhip = np.where(dataset_joints == 'lhip')[0][0]
        lkne = np.where(dataset_joints == 'lkne')[0][0]
        lank = np.where(dataset_joints == 'lank')[0][0]

        rsho = np.where(dataset_joints == 'rsho')[0][0]
        relb = np.where(dataset_joints == 'relb')[0][0]
        rwri = np.where(dataset_joints == 'rwri')[0][0]
        rkne = np.where(dataset_joints == 'rkne')[0][0]
        rank = np.where(dataset_joints == 'rank')[0][0]
        rhip = np.where(dataset_joints == 'rhip')[0][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=2)
        headsizes = headboxes_src[:, 1, :] - headboxes_src[:, 0, :]
        headsizes = np.linalg.norm(headsizes, axis=1)
        headsizes[ ~(jnt_labeled[:,-1]&jnt_labeled[:,-2]) ] = 95

        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((uv_err.shape[1], 1 )))
        scale = np.swapaxes(scale,0,1)
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=0)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                            jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=0), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 14))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                                jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=0),
                                        jnt_count)

        # PCKh = np.ma.array(PCKh, mask=False)
        # PCKh.mask[6:8] = True

        # jnt_count = np.ma.array(jnt_count, mask=False)
        # jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Neck', PCKh[neck]),
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
            ('AP', AP)
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

    # need double check this API and classes field
    def evaluate_AP(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % self.image_set)

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': img_path[idx][-44:-4]
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        
        ai_kpts = defaultdict(dict)
        for kpt in _kpts:
            # ai challenger
            if not ai_kpts[kpt['image']] :
                ai_kpts[kpt['image']]['keypoint_annotations'] = {}
            h_num = len(ai_kpts[kpt['image']]['keypoint_annotations'])
            ai_kpts[kpt['image']]['image_id'] = kpt['image']
            ai_kpts[kpt['image']]['keypoint_annotations']['human'+str(h_num+1)] = list(kpt['keypoints'].reshape(-1))
            # coco
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        actual_num_joints = self.actual_num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        # ai challenger
        ai_oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, actual_num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score
            # keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))],
            #                oks_thre)
            keep = range(len(img_kpts))
            
            if len(keep) == 0:
                ai_oks_nmsed_kpts.append(ai_kpts[img])
            else:
                ai_oks_nmsed_kpts.append( {'image_id':  ai_kpts[img]['image_id'], \
                                       'keypoint_annotations':
                                       {'human'+str(_keep+1):ai_kpts[img]['keypoint_annotations']['human'+str(_keep+1)] \
                                        for _keep in keep}}   )     
                
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

#         self._write_coco_keypoint_results(
#             oks_nmsed_kpts, res_file)
        
        if 'test' not in self.image_set:
            
            # Initialize return_dict
            return_dict = dict()
            return_dict['error'] = None
            return_dict['warning'] = []
            return_dict['score'] = None
            
            # Load annotation JSON file
            start_time = time.time()
            annotations = self.load_annotations(anno_file=self.anno_name,
                                           return_dict=return_dict)
            print('Complete reading annotation JSON file in %.2f seconds.' %(time.time() - start_time))

            # Load prediction JSON file
            start_time = time.time()
            predictions = self.load_predictions(preds=ai_oks_nmsed_kpts,
                                           return_dict=return_dict)
            print('Complete reading prediction JSON file in %.2f seconds.' %(time.time() - start_time))
            
            # Keypoint evaluation
            start_time = time.time()
            return_dict = self.keypoint_eval(predictions=predictions,
                                        annotations=annotations,
                                        return_dict=return_dict)
            print('Complete evaluation in %.2f seconds.' %(time.time() - start_time))

            # Print return_dict and final score
            pprint.pprint(return_dict)
            print('Score: ', '%.8f' % return_dict['score'])

            return {'AP': return_dict['score']}, return_dict['score']
        else:
            return {'Null': 0}, 0
        
        
    def load_annotations(self, anno_file, return_dict):
        """Convert annotation JSON file."""

        annotations = dict()
        annotations['image_ids'] = set([])
        annotations['annos'] = dict()
        annotations['delta'] = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                                           0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                                           0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])
        try:
            annos = json.load(open(anno_file, 'r'))
        except Exception:
            return_dict['error'] = 'Annotation file does not exist or is an invalid JSON file.'
            exit(return_dict['error'])

        for anno in annos:
            annotations['image_ids'].add(anno['image_id'])
            annotations['annos'][anno['image_id']] = dict()
            annotations['annos'][anno['image_id']]['human_annos'] = anno['human_annotations']
            annotations['annos'][anno['image_id']]['keypoint_annos'] = anno['keypoint_annotations']

        return annotations


    def load_predictions(self, preds, return_dict):
        """Convert prediction JSON file."""

        predictions = dict()
        predictions['image_ids'] = []
        predictions['annos'] = dict()
        id_set = set([])

        for pred in preds:
            if 'image_id' not in list(pred.keys()):
                return_dict['warning'].append('There is an invalid annotation info, \
                    likely missing key \'image_id\'.')
                continue
            if 'keypoint_annotations' not in list(pred.keys()):
                return_dict['warning'].append(pred['image_id']+\
                    ' does not have key \'keypoint_annotations\'.')
                continue
            image_id = pred['image_id'].split('.')[0]
            if image_id in id_set:
                return_dict['warning'].append(pred['image_id']+\
                    ' is duplicated in prediction JSON file.')
            else:
                id_set.add(image_id)
            predictions['image_ids'].append(image_id)
            predictions['annos'][pred['image_id']] = dict()
            predictions['annos'][pred['image_id']]['keypoint_annos'] = pred['keypoint_annotations']

        return predictions
    
    
    def compute_oks(self, anno, predict, delta):
        """Compute oks matrix (size gtN*pN)."""

        anno_count = len(list(anno['keypoint_annos'].keys()))
        predict_count = len(list(predict.keys()))
        oks = np.zeros((anno_count, predict_count))
        if predict_count == 0:
            return oks.T

        # for every human keypoint annotation
        for i in range(anno_count):
            anno_key = list(anno['keypoint_annos'].keys())[i]
            anno_keypoints = np.reshape(anno['keypoint_annos'][anno_key], (14, 3))
            visible = anno_keypoints[:, 2] == 1
            bbox = anno['human_annos'][anno_key]
            scale = np.float32((bbox[3]-bbox[1])*(bbox[2]-bbox[0]))
            if np.sum(visible) == 0:
                for j in range(predict_count):
                    oks[i, j] = 0
            else:
                # for every predicted human
                for j in range(predict_count):
                    predict_key = list(predict.keys())[j]
                    predict_keypoints = np.reshape(predict[predict_key], (14, 3))
                    dis = np.sum((anno_keypoints[visible, :2] \
                        - predict_keypoints[visible, :2])**2, axis=1)
                    oks[i, j] = np.mean(np.exp(-dis/2/delta[visible]**2/(scale+1)))
        return oks


    def keypoint_eval(self, predictions, annotations, return_dict):
        """Evaluate predicted_file and return mAP."""

        oks_all = np.zeros((0))
        oks_num = 0

        # Construct set to speed up id searching.
        prediction_id_set = set(predictions['image_ids'])

        # for every annotation in our test/validation set
        for image_id in annotations['image_ids']:
            # if the image in the predictions, then compute oks
            if image_id in prediction_id_set:
                oks = self.compute_oks(anno=annotations['annos'][image_id], \
                                  predict=predictions['annos'][image_id]['keypoint_annos'], \
                                  delta=annotations['delta'])
                # view pairs with max OKSs as match ones, add to oks_all
                oks_all = np.concatenate((oks_all, np.max(oks, axis=1)), axis=0)
                # accumulate total num by max(gtN,pN)
                oks_num += np.max(oks.shape)
            else:
                # otherwise report warning
                return_dict['warning'].append(image_id+' is not in the prediction JSON file.')
                # number of humen in ground truth annotations
                gt_n = len(list(annotations['annos'][image_id]['human_annos'].keys()))
                # fill 0 in oks scores
                oks_all = np.concatenate((oks_all, np.zeros((gt_n))), axis=0)
                # accumulate total num by ground truth number
                oks_num += gt_n

        # compute mAP by APs under different oks thresholds
        average_precision = []
        for threshold in np.linspace(0.5, 0.95, 10):
            average_precision.append(np.sum(oks_all > threshold)/np.float32(oks_num))
        return_dict['score'] = np.mean(average_precision)

        return return_dict