import os
import random

import cv2
import json
import math
import numpy as np
import torch.utils.data as data
import pydicom as pd
from image import get_border, get_affine_transform, affine_transform
from image import draw_umich_gaussian, gaussian_radius


class KidneyStonesAxisDataset(data.Dataset):
    def __init__(self, split, split_ratio=1.0, img_size=256):
        super(KidneyStonesAxisDataset, self).__init__()
        self.num_classes = 4
        self.class_name = ['stone']
        self.valid_ids = [0]
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

        self.data_rng = np.random.RandomState(123)
        # self.mean = COCO_MEAN
        # self.std = COCO_STD

        self.split = split
        self.data_dir = 'data'
        self.img_dir = 'data'
        if split == 'val':
            self.annot_path = '../Data/data/labels_axis_val.json'
        else:
            self.annot_path = '../Data/data/labels_axis_train.json'

        self.max_objs = 128
        self.padding = 127  # 31 for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {'h': img_size, 'w': img_size}
        self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7

        print('==> initializing kidney stones %s data.' % split)
        # self.coco = coco.COCO(self.annot_path)
        self.image_ids = []
        with open(self.annot_path, 'r') as f:
            json_annot = json.load(f)
            for image in json_annot['images']:
                image_id = image['id']
                try:
                    self.image_ids.index(image_id)
                except ValueError:
                    self.image_ids.append(image['id'])

        if 0 < split_ratio < 1:
            split_size = int(np.clip(split_ratio * len(self.image_ids), 1, len(self.image_ids)))
            self.image_ids = self.image_ids[:split_size]

        random.shuffle(self.image_ids)
        self.num_samples = len(self.image_ids)

        print('Loaded %d %s samples' % (self.num_samples, split))

    def get_annotations(self, img_id):
        annotations = []
        with open(self.annot_path, 'r') as f:
            json_annot = json.load(f)
            for a in json_annot['annotations']:
                if a['image_id'] == img_id:
                    annotations.append(a)
        return annotations

    def get_image_path(self, img_id):
        with open(self.annot_path, 'r') as f:
            json_annot = json.load(f)
            for i in json_annot['images']:
                if i['id'] == img_id:
                    return i['file_name']
        return None

    def __len__(self):
        return len(self.image_ids)
        # return 1

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_file = self.get_image_path(img_id)
        img_path = '../Data/' + img_file
        annotations = self.get_annotations(img_id)
        labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations])
        #  points - top extreme point, bottom extreme point
        points = np.array([anno['points'] for anno in annotations], dtype=np.float32)

        gt_points = []
        for p in points:
            gt_points.append(p)

        file = pd.read_file(img_path)
        img = np.asarray(file.pixel_array)
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
        scale = max(height, width) * 1.0

        flipped = False
        if self.split == 'train':
            scale = scale * np.random.choice(self.rand_scales)
            w_border = get_border(128, width)
            h_border = get_border(128, height)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)

            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1]
                center[0] = width - center[0] - 1
        # if self.split == 'val':
        #     scale = np.array([scale, scale], dtype=np.int32)

        trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])
        img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))

        img = img.astype(np.float32)
        img = np.stack((img, img, img), axis=2)

        img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]
        trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])

        hm_t = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap top point
        hm_b = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap bottom point

        reg_t = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression top point
        reg_b = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression bottom point

        ind_t = np.zeros((self.max_objs), dtype=np.int64)
        ind_b = np.zeros((self.max_objs), dtype=np.int64)

        ind_masks = np.zeros(self.max_objs, dtype=np.uint8)
        reg_mask = np.zeros(self.max_objs, dtype=np.uint8)

        num_objs = len(annotations)
        for k in range(num_objs):
            ann = annotations[k]
            # bbox = self._coco_box_to_bbox(ann['bbox'])
            # tlbr
            pts = np.array(ann['points'], dtype=np.float32).reshape(4, 2)
            # cls_id = int(self.cat_ids[ann['category_id']] - 1) # bug
            cls_id = int(self.cat_ids[ann['category_id']])
            hm_id = 0 if self.opt.agnostic_ex else cls_id
            if flipped:
                pts[:, 0] = width - pts[:, 0] - 1
                pts[1], pts[3] = pts[3].copy(), pts[1].copy()
            for j in range(4):
                pts[j] = affine_transform(pts[j], trans_fmap)
            pts = np.clip(pts, 0, self.opt.output_res - 1)
            h, w = pts[2, 1] - pts[0, 1], pts[3, 0] - pts[1, 0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                pt_int = pts.astype(np.int32)
                draw_umich_gaussian(hm_t[hm_id], pt_int[0], radius)
                draw_umich_gaussian(hm_b[hm_id], pt_int[2], radius)
                reg_t[k] = pts[0] - pt_int[0]
                reg_b[k] = pts[2] - pt_int[2]
                ind_t[k] = pt_int[0, 1] * [self.fmap_size['w'], self.fmap_size['h']] + pt_int[0, 0]
                ind_b[k] = pt_int[2, 1] * [self.fmap_size['w'], self.fmap_size['h']] + pt_int[2, 0]
                reg_mask[k] = 1

        # detections = np.array(detections, dtype=np.float32) \
        #   if len(detections) > 0 else np.zeros((1, 6), dtype=np.float32)

        img_id = img_id.split('.')[-1]

        # return {'image': img,
        #         'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
        #         'c': center, 's': scale, 'img_id': img_id, 'fmap_w': self.fmap_size['w'],
        #         'fmap_h': self.fmap_size['h'], 'gt_boxes': gt_boxes, 'img_file': img_file.split('/')[-1]}

        ret = {'image': img, 'hm_t': hm_t, 'hm_b': hm_b, 'fmap_w': self.fmap_size['w'],
                'fmap_h': self.fmap_size['h'], 'gt_points': gt_points, 'img_file': img_file.split('/')[-1],
               'c': center, 's': scale, 'img_id': img_id,}
        if self.opt.reg_offset:
            ret.update({'reg_mask': reg_mask,
                        'reg_t': reg_t, 'reg_b': reg_b,
                        'ind_t': ind_t, 'ind_b': ind_b})
        return ret
