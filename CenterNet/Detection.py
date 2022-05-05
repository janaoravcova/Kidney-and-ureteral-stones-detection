import json
import os
import sys
from math import floor
from original_hourglass import get_small_hourglass_net
import cv2
import argparse
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.ndimage import gaussian_filter
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import torch
import torch.utils.data
from utils import load_model
from image import transform_preds, get_affine_transform, gaussian_radius, draw_umich_gaussian, affine_transform
from post_process import ctdet_decode
import pydicom as pd

Focus_classes = ['stone']

# Training settings
parser = argparse.ArgumentParser(description='centernet')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--ckpt_detector', type=str, default='model-1.0.pth')
parser.add_argument('--ckpt_classifier', type=str, default='./Classifier/epoch_45.pth.tar')
parser.add_argument('--arch', type=str, default='small_hourglass')

cfg = parser.parse_args()
os.chdir(cfg.root_dir)


def overlap_ratio(bbox1, bbox2, thresh):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    # calculate area of two bbox
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    # calculate area of intersection
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h
    if (s1 + s2 - a1) <= 0:
        return False
    if a1 / (s1 + s2 - a1) > thresh:
        return True
    return False


def max_bbox(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    return [min(xmin1, xmin2), min(ymin1, ymin2), max(xmax1, xmax2), max(ymax1, ymax2)]


def absorb(dets, thresh=0.5):
    keep = []
    keep_conf = []
    for bbox, conf in dets:
        absorbed = 0
        for index, exist_bbox in enumerate(keep):
            if overlap_ratio(bbox, exist_bbox, thresh):
                keep[index] = max_bbox(bbox, exist_bbox)
                keep_conf[index] = max(conf, keep_conf[index])
                absorbed = 1
        if absorbed == 0:
            keep.append(bbox)
            keep_conf.append(conf)

    return keep, keep_conf


def calculate_tp_fn_ra(gt, pred, iou_thresh=0.3):
    tp = 0
    fp = 0
    for gt_box in gt:
        for pred_item in pred:
            pred_box = [pred_item[0], pred_item[1], pred_item[2], pred_item[3]]
            if overlap_ratio(gt_box, pred_box, iou_thresh):
                tp += 1
    fn = len(gt) - tp
    fp = len(pred) - tp
    return tp, fn, fp


def get_annotations(img_id, json_annot):
    annotations = []
    for a in json_annot['annotations']:
        if a['image_id'] == img_id:
            annotations.append(a)
    return annotations


def get_previous_id(frame_id, look_back_count):
    return int(frame_id.split(".")[-1]) - look_back_count


def get_next_id(frame_id, look_forward_count):
    return int(frame_id.split(".")[-1]) + look_forward_count


def evaluate_on_tracklets(gt, pred_bb_with_id):
    boxes_groups = group_boxes(pred_bb_with_id)
    groups = []
    fn_tr, tp_tr, fp_tr = 0, 0, 0
    for key in sorted(boxes_groups.keys()):
        for obj in pred_bb_with_id:
            if obj['id'] == key:
                groups.append({
                    "id": key,
                    "frame_id": obj['frame_id'],
                    "track_id": boxes_groups[key],
                    "box": obj['box']
                })
    gt_with_track_id = fit_gt_in_groups(groups, gt)
    # print("GT {}".format(gt))
    real_groups = {}
    for det in groups:
        track_id = det['track_id']
        values = real_groups.get(track_id)
        if values is None:
            values = []
        values.append({
            "frame_id": det['frame_id'],
            "det_id": det['id'],
            "track_id": track_id
        })
        real_groups[track_id] = values

    tp_tr = 0
    tp_tr_small = 0
    found_track_ids = []
    real_sizes = []
    for gt_box in gt:
        for key in real_groups.keys():
            group = real_groups[key]
            if group[0]['track_id'] == gt_box['track_id']:
                if gt_box['len'] >= 3:
                    tp_tr += 1
                    found_track_ids.append(gt_box['track_id'])
                    real_sizes.append(gt_box['len'])
                else:
                    tp_tr_small += 1
        if gt_box['track_id'] == -1:
            print(gt_box)
            if gt_box['len'] >= 3:
                fn_tr += 1
    return tp_tr, len(real_groups.keys()) - tp_tr - tp_tr_small, fn_tr, real_groups, found_track_ids, real_sizes


def group_boxes(boxes):
    boxes_groups = {}
    track_id_counter = 0
    for a in boxes:
        for b in boxes:
            if overlap_ratio(a['box'], b['box'], 0.3):

                if get_previous_id(a['frame_id'], 5) <= int(b['frame_id'].split(".")[-1]) <= get_next_id(a['frame_id'],
                                                                                                         5) or \
                        get_previous_id(b['frame_id'], 5) <= int(a['frame_id'].split(".")[-1]) <= get_next_id(
                    b['frame_id'], 5):
                    track_id = None
                    a_track_id = boxes_groups.get(a['id'])
                    b_track_id = boxes_groups.get(b['id'])
                    if a_track_id is not None and b_track_id is not None:
                        if a_track_id == b_track_id:
                            continue
                        else:
                            track_id = track_id_counter
                            track_id_counter += 1
                            boxes_groups[a['id']] = track_id
                            boxes_groups[b['id']] = track_id
                            for key in boxes_groups.keys():
                                if boxes_groups.get(key) == a_track_id or boxes_groups.get(key) == b_track_id:
                                    boxes_groups[key] = track_id
                            continue

                    if a_track_id is not None:
                        boxes_groups[b['id']] = a_track_id
                        continue
                    if b_track_id is not None:
                        boxes_groups[a['id']] = b_track_id
                        continue
                    if a_track_id is None and b_track_id is None:
                        track_id = track_id_counter
                        track_id_counter += 1
                        boxes_groups[a['id']] = track_id
                        boxes_groups[b['id']] = track_id

    return boxes_groups


def check_if_fits(gt_box, group):
    if get_previous_id(group['frame_id'], 2) <= int(gt_box['frame_id'].split(".")[-1]) <= get_next_id(group['frame_id'],
                                                                                                      2) \
            and overlap_ratio(gt_box['box'], group['box'], 0.3):
        return group['track_id']
    return None


def fit_gt_in_groups(boxes_groups, boxes_gt):
    for gt_box in boxes_gt:
        for group in boxes_groups:
            track_id = check_if_fits(gt_box, group)
            if track_id is None:
                continue
            gt_box['track_id'] = track_id
    return boxes_gt


def check_if_object_in_gt_group(obj, gt):
    for gt_obj in gt:
        if obj['track_id'] == gt_obj['track_id']:
            return True
    return False


def evaluation(model, original_folder, target_folder, visualization=True):
    max_per_image = 10
    test_gt_file = '../Data/data/labels_bbox_modified_test.json'
    with open(test_gt_file, 'r') as f:
        json_object = json.load(f)
    tp = 0
    fn = 0
    fp = 0

    tp_per_score = {0.05: 0, 0.10: 0, 0.15: 0, 0.20: 0, 0.25: 0, 0.3: 0,
                    0.35: 0, 0.40: 0, 0.45: 0, 0.5: 0, 0.55: 0,
                    0.60: 0, 0.65: 0, 0.70: 0, 0.75: 0, 0.80: 0, 0.85: 0, 0.90: 0, 0.95: 0}
    fp_per_score = {0.05: 0, 0.10: 0, 0.15: 0, 0.20: 0, 0.25: 0, 0.3: 0,
                    0.35: 0, 0.40: 0, 0.45: 0, 0.5: 0, 0.55: 0,
                    0.60: 0, 0.65: 0, 0.70: 0, 0.75: 0, 0.80: 0, 0.85: 0, 0.90: 0, 0.95: 0}
    fn_per_score = {0.05: 0, 0.10: 0, 0.15: 0, 0.20: 0, 0.25: 0, 0.3: 0,
                    0.35: 0, 0.40: 0, 0.45: 0, 0.5: 0, 0.55: 0,
                    0.60: 0, 0.65: 0, 0.70: 0, 0.75: 0, 0.80: 0, 0.85: 0, 0.90: 0, 0.95: 0
                    }
    tp_per_score_tr = {0.05: 0, 0.10: 0, 0.15: 0, 0.20: 0, 0.25: 0, 0.3: 0,
                       0.35: 0, 0.40: 0, 0.45: 0, 0.5: 0, 0.55: 0,
                       0.60: 0, 0.65: 0, 0.70: 0, 0.75: 0, 0.80: 0, 0.85: 0, 0.90: 0, 0.95: 0}
    fp_per_score_tr = {0.05: 0, 0.10: 0, 0.15: 0, 0.20: 0, 0.25: 0, 0.3: 0,
                       0.35: 0, 0.40: 0, 0.45: 0, 0.5: 0, 0.55: 0,
                       0.60: 0, 0.65: 0, 0.70: 0, 0.75: 0, 0.80: 0, 0.85: 0, 0.90: 0, 0.95: 0}
    fn_per_score_tr = {0.05: 0, 0.10: 0, 0.15: 0, 0.20: 0, 0.25: 0, 0.3: 0,
                       0.35: 0, 0.40: 0, 0.45: 0, 0.5: 0, 0.55: 0,
                       0.60: 0, 0.65: 0, 0.70: 0, 0.75: 0, 0.80: 0, 0.85: 0, 0.90: 0, 0.95: 0
                       }

    found_stones_stats = []
    for img_folder in os.listdir(original_folder):
        print(img_folder)
        img_folder_path = os.path.join(original_folder, img_folder)
        pred_bb_with_id = []
        pred_bb_with_id_tracklets = {0.05: [], 0.10: [], 0.15: [], 0.20: [], 0.25: [], 0.3: [],
                                     0.35: [], 0.40: [], 0.45: [], 0.5: [], 0.55: [],
                                     0.60: [], 0.65: [], 0.70: [], 0.75: [], 0.80: [], 0.85: [], 0.90: [], 0.95: []}
        gt = []

        hu_volume = []
        frame_ids = []
        pixel_spacing = []
        for dcm_file in os.listdir(img_folder_path):
            if not dcm_file.lower().endswith('.dcm'):
                continue
            img_path = os.path.join(img_folder_path, dcm_file)
            try:
                file = pd.read_file(img_path)
                image = np.asarray(file.pixel_array).astype(np.int16)
                pixel_spacing = file.PixelSpacing
            except Exception:
                continue

            try:
                image[image == -2000] = 0
                intercept = file.RescaleIntercept
                slope = file.RescaleSlope

                if slope != 1:
                    image = slope * image.astype(np.float64)
                    image = image.astype(np.int16)
                image += np.int16(intercept)

            except Exception:
                pass
            hu_volume.append(image)
            frame_ids.append(file.SOPInstanceUID)
            gt_per_image = get_annotations(file.SOPInstanceUID, json_object)

            gt_boxes = []
            for (x_0, y_0, bbox_width, bbox_height), l in zip([anno['bbox'] for anno in gt_per_image],
                                                              [anno['len'] for anno in gt_per_image]):
                x_1 = x_0 + bbox_width
                y_1 = y_0 + bbox_height
                gt_boxes.append((x_0, y_0, x_1, y_1, l))

            image = image.astype(np.float32)
            height, width = image.shape[0:2]
            imgs = {}
            scale = 1
            new_height = int(height * scale)
            new_width = int(width * scale)

            img_height, img_width = 512, 512
            center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            scaled_size = max(height, width) * 1.0
            scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)

            img = cv2.resize(image, (new_width, new_height))
            trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
            img = cv2.warpAffine(img, trans_img, (img_width, img_height))

            img = gaussian_filter(img, sigma=0.7)
            img = (img[:, :] - (img.min())) / (img.max() - (img.min()))
            img = img[np.newaxis, :, :]
            imgs[scale] = {'image': torch.from_numpy(img).float(),
                           'center': np.array(center),
                           'scale': np.array(scaled_size),
                           'fmap_h': np.array(img_height // 4),
                           'fmap_w': np.array(img_width // 4)}

            # Prediction on input image
            with torch.no_grad():
                detections = []
                for scale in imgs:
                    imgs[scale]['image'] = imgs[scale]['image'].cuda()

                    output = model(imgs[scale]['image'])[-1]
                    hmap, regs, w_h_ = output['hm'], output['reg'], output['wh']
                    dets = ctdet_decode(hmap, regs, w_h_, K=100)
                    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                    top_preds = {}
                    dets[:, :2] = transform_preds(dets[:, 0:2],
                                                  imgs[scale]['center'],
                                                  imgs[scale]['scale'],
                                                  (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
                    dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                                   imgs[scale]['center'],
                                                   imgs[scale]['scale'],
                                                   (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
                    cls = dets[:, -1]
                    for j in range(1):
                        inds = (cls == j)
                        top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                        top_preds[j + 1][:, :4] /= scale

                    detections.append(top_preds)

                bbox_and_scores = {}
                for j in range(1):
                    temp = np.concatenate([d[j + 1] for d in detections], axis=0)
                    bbox_and_scores[j] = temp

                scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1)])
                if len(scores) > max_per_image:
                    kth = len(scores) - max_per_image
                    thresh = np.partition(scores, kth)[kth]
                    for j in range(1):
                        keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                        bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

                fig = plt.figure(0)
                plt.imshow(image, cmap='gray')

                predicts = []
                gt_bb = []
                predicts_per_score = {0.05: [], 0.10: [], 0.15: [], 0.20: [], 0.25: [], 0.3: [],
                                      0.35: [], 0.40: [], 0.45: [], 0.5: [], 0.55: [],
                                      0.60: [], 0.65: [], 0.70: [], 0.75: [], 0.80: [], 0.85: [], 0.90: [], 0.95: []}
                for lab in bbox_and_scores:
                    for boxes in bbox_and_scores[lab]:
                        x1, y1, x2, y2, score = boxes
                        for score_threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60,
                                                0.65, 0.70,
                                                0.75, 0.80, 0.85, 0.9, 0.95]:
                            if score > score_threshold:
                                predicts_per_score[score_threshold].append(([x1, y1, x2, y2], score))

                        if score >= 0.3:
                            predicts.append(([x1, y1, x2, y2], score))

                pred_bb_with_conf = []
                pred_bb_per_score = {0.05: [], 0.10: [], 0.15: [], 0.20: [], 0.25: [], 0.3: [],
                                     0.35: [], 0.40: [], 0.45: [], 0.5: [], 0.55: [],
                                     0.60: [], 0.65: [], 0.70: [], 0.75: [], 0.80: [], 0.85: [], 0.90: [], 0.95: []}
                for sc in predicts_per_score.keys():
                    pred_bb, pred_conf = absorb(predicts_per_score[sc])
                    for k, bbox in enumerate(pred_bb):
                        x1, y1, x2, y2 = bbox
                        pred_bb_per_score[sc].append([x1, y1, x2, y2, pred_conf[k]])
                        pred_bb_with_id_tracklets[sc].append({
                            "id": len(pred_bb_with_id_tracklets[sc]) + 1,
                            "frame_id": file.SOPInstanceUID,
                            "box": [int(x1), int(y1), int(x2), int(y2)]})

                pred_bb, pred_conf = absorb(predicts)  # Post-Process to merge bboxs
                for k, bbox in enumerate(pred_bb):
                    x1, y1, x2, y2 = bbox
                    pred_bb_with_conf.append([x1, y1, x2, y2, pred_conf[k]])
                    pred_bb_with_id.append({
                        "id": len(pred_bb_with_id) + 1,
                        "frame_id": file.SOPInstanceUID,
                        "box": bbox
                    })

                    plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0.5, edgecolor='r',
                                                  facecolor='none'))  # Plot bbox

                for boxes in gt_boxes:
                    x1, y1, x2, y2, l = boxes
                    gt_bb.append([x1, y1, x2, y2])
                    gt.append({
                        "frame_id": file.SOPInstanceUID,
                        "box": [x1, y1, x2, y2],
                        "track_id": -1,
                        "len": l
                    })
                    gt_per_image.append([x1, y1, x2, y2, 0, 0, 0])
                    m_x = (x1 + x2) / 2
                    m_y = (y1 + y2) / 2
                    plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0.5, edgecolor='green',
                                                  facecolor='none'))

                for sc in pred_bb_per_score.keys():
                    tp_per_img, fn_per_img, fp_per_image = calculate_tp_fn_ra(gt_bb, pred_bb_per_score[sc])
                    tp_per_score[sc] += tp_per_img
                    fp_per_score[sc] += fp_per_image
                    fn_per_score[sc] += fn_per_img

                tp_per_img, fn_per_img, fp_per_image = calculate_tp_fn_ra(gt_bb, pred_bb_with_conf)
                tp += tp_per_img
                fn += fn_per_img
                fp += fp_per_image

                fig.patch.set_visible(False)
                plt.axis('off')
                if len(gt_boxes) > 0 or len(predicts) > 0:
                    save_path = os.path.join(target_folder, img_folder)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    save_name = os.path.join(save_path, dcm_file[:-3] + 'png')
                    plt.savefig(save_name, dpi=300, transparent=True)  # Save visualization result
                plt.close('all')

        boxes_for_fpr = []
        for sc in pred_bb_with_id_tracklets.keys():
            pred = pred_bb_with_id_tracklets[sc]
            tp_tr, fp_tr, fn_tr, group_objects, found_track_ids, real_sizes = evaluate_on_tracklets(copy.deepcopy(gt),
                                                                                                    pred)
            fn_per_score_tr[sc] += fn_tr
            tp_per_score_tr[sc] += tp_tr
            fp_per_score_tr[sc] += fp_tr
            print("Per score {} tp={}, fp={}, fn={}".format(sc, tp_tr, fp_tr, fn_tr))
            score_string = str(sc * 100)
            if sc == 0.45:
                per_patient_results = []
                for track_id in group_objects.keys():
                    group_dets = group_objects[track_id]
                    try:
                        found_index = found_track_ids.index(track_id)
                        real_size = real_sizes[found_index]
                    except ValueError:
                        continue
                    middle_slice_index = (floor(len(group_dets) / 2))
                    middle_slice_det = group_dets[middle_slice_index]
                    original_det = pred[middle_slice_det['det_id'] - 1]
                    x1, y1, x2, y2 = original_det['box']
                    m_x = (x1 + x2) / 2
                    m_y = (y1 + y2) / 2
                    frame_id = original_det['frame_id']

                    per_patient_results.append({

                        "frame_id": frame_id,
                        "position": [int(m_x), int(m_y)],
                        "len": real_size
                    })
                    boxes_for_fpr.append(' '.join([img_folder, frame_id, str(int(m_x)), str(int(m_y))]) + '\n')
                found_stones_stats.append({
                    "patient": img_folder,
                    "results": per_patient_results
                })
        if len(found_stones_stats) > 0:
            with open('results_stone_cn.json', 'w') as file_obj:
                json.dump(found_stones_stats, file_obj, indent=4)

        sensitivities = [(tp_per_score[sc] / (tp_per_score[sc] + fn_per_score[sc])) for sc in tp_per_score.keys()]
        specifities = [(fp_per_score[sc] / (len(os.listdir(original_folder)))) for sc in tp_per_score.keys()]

        sensitivities_tr = []
        for sc in tp_per_score.keys():
            if tp_per_score_tr[sc] == 0 and fn_per_score_tr[sc] == 0:
                sensitivities_tr.append(0)
            else:
                sensitivities_tr.append((tp_per_score_tr[sc] / (tp_per_score_tr[sc] + fn_per_score_tr[sc])))
        # sensitivities_tr = [(tp_per_score_tr[sc] / (tp_per_score_tr[sc] + fn_per_score_tr[sc])) for sc in tp_per_score.keys()]
        specifities_tr = [(fp_per_score_tr[sc] / (len(os.listdir(original_folder)))) for sc in tp_per_score.keys()]
        print(sensitivities)
        print(specifities)
        print(sensitivities_tr)
        print(specifities_tr)

        plt.plot(specifities, sensitivities, 'orange')
        plt.ylabel("citlivosť")
        plt.xlabel("FPS na pacienta")
        plt.show()
        plt.savefig('roc_mw.png', dpi=300, transparent=True)
        plt.close('all')
        plt.plot(specifities_tr, sensitivities_tr, 'blue')
        plt.ylabel("citlivosť")
        plt.xlabel("FPS na pacienta")
        plt.show()
        plt.savefig('roc_mw_tr.png', dpi=300, transparent=True)
        plt.close('all')


def main():
    cfg.device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False
    print('Creating model...')

    if 'hourglass' in cfg.arch:
        model = get_small_hourglass_net()
    else:
        raise NotImplementedError

    model = load_model(model, 'model-single-slice-ign-lrg-lr.pth')
    model = model.to(cfg.device)
    model.eval()
    evaluation(model, '../Data/test/', '../Data/test_results')


if __name__ == '__main__':
    main()
