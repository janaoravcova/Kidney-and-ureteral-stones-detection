import os
import random

import cv2
import json
import math
import numpy as np
import torch.utils.data as data
import pydicom as pd
from scipy.ndimage import gaussian_filter

from image import get_border, get_affine_transform, affine_transform
from image import draw_umich_gaussian, gaussian_radius


class KidneyStonesDataset(data.Dataset):
    def __init__(self, split, split_ratio=1.0, img_size=256, spatial=False):
        super(KidneyStonesDataset, self).__init__()
        self.num_classes = 1
        self.class_name = ['stone']
        self.valid_ids = [0]
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

        self.data_rng = np.random.RandomState(123)

        self.split = split
        self.data_dir = 'data'
        self.img_dir = 'data'
        if split == 'val':
            self.annot_path = '../Data/data/labels_final_val.json'
            self.ignore_regions = self.gather_ignore_regions('../Data/data/ignore_regions_val.txt')
        else:
            self.annot_path = '../Data/data/labels_final_train.json'
            self.ignore_regions = self.gather_ignore_regions('../Data/data/ignore_regions_train.txt')

        self.max_objs = 50
        self.padding = 127  # 31 for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {'h': img_size, 'w': img_size}
        self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7

        print('==> initializing kidney stones %s data.' % split)
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

        self.num_samples = len(self.image_ids)

        print('Loaded %d %s samples' % (self.num_samples, split))

    def get_annotations(self, img_id):
        annotations = []
        with open(self.annot_path, 'r') as f:
            json_annot = json.load(f)
            for a in json_annot['annotations']:
                if a['image_id'] == img_id:
                    # w, h = a['bbox'][2], a['bbox'][3]
                    # if w > 3 or h > 3:
                    annotations.append(a)
        return annotations

    def get_image_path(self, img_id):
        with open(self.annot_path, 'r') as f:
            json_annot = json.load(f)
            for i in json_annot['images']:
                if i['id'] == img_id:
                    return i['file_name']
        return None

    def get_prev_image_path(self, img_id):
        if self.split == 'val':
            prefix_id = 13
        else:
            prefix_id = 8
        with open(self.annot_path, 'r') as f:
            json_annot = json.load(f)
            for i in json_annot['images']:
                if i['id'] == img_id:
                    return 'data/' + i['prev_file_name'][prefix_id:]
        return None

    def get_next_image_path(self, img_id):
        if self.split == 'val':
            prefix_id = 13
        else:
            prefix_id = 8
        with open(self.annot_path, 'r') as f:
            json_annot = json.load(f)
            for i in json_annot['images']:
                if i['id'] == img_id:
                    return 'data/' + i['next_file_name'][prefix_id:]
        return None

    def shuffle_data(self):
        random.shuffle(self.image_ids)

    def __len__(self):
        return len(self.image_ids)
        # return 1

    def window_image(self, dicom_image, window_center, window_width):

        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        window_image = dicom_image.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max

        return window_image

    def gather_ignore_regions(self, path):
        ignore_regions_directory = {}
        with open(path, 'r') as file:
            lines = file.readlines()
            data_array = np.array([line.split(' ') for line in lines])
            frame_ids = data_array[:, 0]
            center = np.stack((data_array[:, 1], data_array[:, 2]), axis=1)
            radius = data_array[:, 3]
            for i, frame_id in enumerate(frame_ids):
                regions = ignore_regions_directory.get(frame_id)
                if regions is None:
                    ignore_regions_directory[frame_id] = []
                ignore_regions_directory[frame_id].append([[int(center[i][0]), int(center[i][1])], int(radius[i])])
        return ignore_regions_directory

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_file = self.get_image_path(img_id)
        img_path = '../Data/' + img_file
        annotations = self.get_annotations(img_id)
        labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations])
        bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32)
        if len(bboxes) == 0:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array([[0, 1, 2, 3]])
        bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy
        gt_boxes = []
        for (x_0, y_0, bbox_width, bbox_height) in [anno['bbox'] for anno in annotations]:
            x_1 = x_0 + bbox_width
            y_1 = y_0 + bbox_height
            gt_boxes.append((x_0, y_0, x_1, y_1))

        file = pd.read_file(img_path)
        scan_slice = file.pixel_array.astype(np.int16)
        scan_slice[scan_slice == -2000] = 0
        intercept = file.RescaleIntercept
        slope = file.RescaleSlope

        if slope != 1:
            scan_slice = slope * scan_slice.astype(np.float64)
            scan_slice = scan_slice.astype(np.int16)

        scan_slice += np.int16(intercept)
        img = scan_slice

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

        trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])
        img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))
        img = img.astype(np.float32)
        # window/level augmentation - apply random
        if self.split == 'train':
            if np.random.random() < 0.5:
                random_level = random.randint(400, 600)
                random_width = random.randint(1100, 2400)
                img = self.window_image(img, random_level, random_width)
        img = gaussian_filter(img, sigma=0.7)
        img = (img[:, :] - (img.min())) / (img.max() - (img.min()))
        trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])

        hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        ignore_regions_per_frame = self.ignore_regions.get(file.SOPInstanceUID)
        hmap_ignore_areas = np.zeros((1, 1, self.fmap_size['w'], self.fmap_size['w']), dtype=np.float32)
        if ignore_regions_per_frame is not None:
            ignore_regions_per_frame = np.array(ignore_regions_per_frame)
            for area in ignore_regions_per_frame:
                center_obj = np.array(area[0]).astype(np.int32)
                center_obj = affine_transform(center_obj, trans_fmap)
                obj_size = max(1, int((area[1] / file.PixelSpacing[0]) / 4))
                # print("Object size {}".format(obj_size))
                hmap_ignore_areas[0, 0, int(center_obj[1]) - obj_size: int(center_obj[1]) + obj_size,
                int(center_obj[0]) - obj_size: int(center_obj[0]) + obj_size] = 1

        detections = []
        for k, (bbox, label) in enumerate(zip(bboxes, labels)):
            # detections.append(bbox)

            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_fmap)
            bbox[2:] = affine_transform(bbox[2:], trans_fmap)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
                draw_umich_gaussian(hmap[label], obj_c_int, radius)
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int  # discretization error
                inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                ind_masks[k] = 1

        img_id = img_id.split('.')[-1]
        return {'image': img,
                'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
                'c': center, 's': scale, 'img_id': img_id, 'fmap_w': self.fmap_size['w'],
                'fmap_h': self.fmap_size['h'], 'gt_boxes': gt_boxes, 'img_file': img_file.split('/')[-1],
                'ignore_mask': hmap_ignore_areas}
