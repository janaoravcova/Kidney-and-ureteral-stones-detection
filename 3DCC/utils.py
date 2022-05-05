import hashlib
import os
import random

import numpy as np
import pandas
import scipy
import sklearn
import pydicom as pd
from pandas import DataFrame
from scipy.ndimage import gaussian_filter, rotate
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table
from skimage import measure
from skimage.morphology import flood

import annotation_parser
import simple_itk_methods


def is_series_axial(series_desc):
    axial_series_desc = [1, 0, 0, 0, 1, 0]
    for j in range(6):
        if series_desc[j] != axial_series_desc[j]:
            return False
    return True


def load_scan(path):
    slices = []
    for file in os.listdir(path):
        if not file.lower().endswith('dcm'):
            continue
        scan_slice = pd.read_file(os.path.join(path, file))
        try:
            if not is_series_axial(scan_slice.ImageOrientationPatient):
                continue
        except AttributeError:
            continue

        slices.append({
            "data": scan_slice,
            "file_name": os.path.join(path, file),
            "frame_id": scan_slice.SOPInstanceUID
        })

    # skip files with no SliceLocation (eg scout views)
    slices_filtered = []
    for scan_slice in slices:
        if hasattr(scan_slice['data'], 'SliceLocation'):
            slices_filtered.append(scan_slice)

    # ensure they are in the correct order
    slices_sorted = sorted(slices_filtered, key=lambda s: s['data'].SliceLocation)
    slices = [s['data'] for s in slices_sorted]
    file_names = [s['file_name'] for s in slices_sorted]
    frame_ids = [s['frame_id'] for s in slices_sorted]

    return slices, file_names, frame_ids


def prepare_slice(original_slice):
    try:
        slice_data = original_slice.pixel_array
    except ValueError:
        return None, None
    scan_slice = slice_data.astype(np.int16)
    # body_mask = bed_remover.remove_bed_from_axial_ct(scan_slice)
    scan_slice[scan_slice == -2000] = 0
    intercept = original_slice.RescaleIntercept
    slope = original_slice.RescaleSlope

    if slope != 1:
        scan_slice = slope * scan_slice.astype(np.float64)
        scan_slice = scan_slice.astype(np.int16)

    scan_slice += np.int16(intercept)

    metadata = {
        "frame_id": original_slice.SOPInstanceUID,
        "pixel_spacing": original_slice.PixelSpacing,
        "slice_thickness": original_slice.SliceThickness,
        "slice_location": original_slice.SliceLocation,
        "image_size": [original_slice.Columns, original_slice.Rows],
    }
    return scan_slice, metadata


def binarize_slice(original_slice):
    filtered_image = gaussian_filter(original_slice, sigma=0.7)
    binary_image = np.where(filtered_image > 150, 1, 0)
    binary_image = np.array(binary_image, dtype=np.int16)
    return scipy.ndimage.binary_fill_holes(binary_image).astype(int)


def is_sphere_like_volume(bbox):
    w, h, d = bbox[4], bbox[5], bbox[3]
    epsilon = 20
    if (h - epsilon <= w <= h + epsilon) and (d - epsilon <= w <= d + epsilon) and \
            (w - epsilon <= h <= w + epsilon) and (d - epsilon <= h <= d + epsilon) and \
            (h - epsilon <= d <= h + epsilon) and (w - epsilon <= d <= w + epsilon):
        return True
    return False


def find_3d_cc(orig_binary_volume, orig_hu_volume, spacing, min=None):
    binary_volume = sitk.GetImageFromArray(orig_binary_volume)
    binary_volume.SetSpacing(spacing)
    cc_filter = sitk.ConnectedComponentImageFilter()
    multi_label_image = cc_filter.Execute(binary_volume)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(multi_label_image)
    # simple_itk_methods.myshow3d(sitk.LabelToRGB(multi_label_image), yslices=range(50, 500, 20), zslices=range(50, 500, 20),
    #                             dpi=30)
    # simple_itk_methods.myshow(sitk.LabelToRGB(multi_label_image[375, :, :]))
    # simple_itk_methods.myshow(sitk.LabelToRGB(multi_label_image[408, :, :]))
    # simple_itk_methods.myshow(sitk.LabelToRGB(multi_label_image[328, :, :]))
    candidates_labels = []
    for label in range(1, label_shape_filter.GetNumberOfLabels() + 1):
        voxel_volume = label_shape_filter.GetNumberOfPixelsOnBorder(label) + label_shape_filter.GetNumberOfPixels(label)
        real_volume = voxel_volume * spacing[0] * spacing[1] * spacing[2]
        bbox = label_shape_filter.GetBoundingBox(label)
        centroid = binary_volume.TransformPhysicalPointToIndex(label_shape_filter.GetCentroid(label))
        if min is None:
            min = 3
        if (min <= real_volume <= 4200) and centroid[2] < 400:
            candidates_labels.append(label)

    candidates_objects = []
    for label in candidates_labels:
        centroid = binary_volume.TransformPhysicalPointToIndex(label_shape_filter.GetCentroid(label))  # (z, x, y)
        bbox = label_shape_filter.GetBoundingBox(label)
        candidates_objects.append({'centroid': centroid, 'bbox': bbox})
    return candidates_objects


def label_candidates(candidates_objects, labels, frame_ids, w, h):
    is_positive_present = False
    for obj in candidates_objects:
        obj['label'] = 0
        for label in labels:
            z_index = frame_ids.index(label['frame_id'])
            m = [round((label['a'][0] * w + label['b'][0] * h) / 2),
                 round((label['a'][1] * w + label['b'][1] * h) / 2)]
            c = [m[0], m[1], z_index]
            bbox = obj['bbox']
            xmin, xmax = bbox[1], bbox[1] + bbox[4]
            ymin, ymax = bbox[2], bbox[2] + bbox[5]
            zmin, zmax = bbox[0], bbox[0] + bbox[3]
            if (xmin < c[0] < xmax) and (ymin < c[1] < ymax) and (zmin <= c[2] <= zmax):
                if label['l'] >= 3:
                    obj['label'] = 1
                    is_positive_present = True
                if label['l'] < 3:
                    obj['label'] = 2
    return candidates_objects, is_positive_present


def rotate_volume(stack):
    rotated_stacks = []
    for i in range(4):
        random_angle = random.randint(-20, 20)
        rotated_stack = []
        for j in range(len(stack)):
            rotated_patch = rotate(stack[j], random_angle, reshape=False)
            rotated_stack.append(rotated_patch)
        rotated_stacks.append(rotated_stack)
    return rotated_stacks


def convert_to_greyscale(hu_volume):
    min_value = 0
    for i in range(len(hu_volume)):
        for j in range(len(hu_volume[i])):
            for k in range(len(hu_volume[i][j])):
                if hu_volume[i][j][k] > 1000:
                    hu_volume[i][j][k] = 1000
                if min_value > hu_volume[i][j][k]:
                    min_value = hu_volume[i][j][k]
    max_value = 1000
    for i in range(len(hu_volume)):
        for j in range(len(hu_volume[i])):
            for k in range(len(hu_volume[i][j])):
                hu_volume[i][j][k] = (hu_volume[i][j][k] - min_value) / (max_value - min_value)
    return hu_volume


def get_stone_length(binary_image):
    all_labels = measure.label(binary_image)
    properties = ['major_axis_length']

    if all_labels.size == 0:
        return 0

    try:
        df = DataFrame(regionprops_table(all_labels, binary_image, properties=properties))
    except IndexError:
        return 0

    if not df.empty:
        if len(df) > 1:
            print("whaat")
        for index, row in df.iterrows():
            return row['major_axis_length']


# return area, average HU, longest axis
def analyze_stone(patch):
    binary_patch = np.zeros(patch.shape)
    filtered_patch = gaussian_filter(patch, sigma=0.7)

    binary_patch[filtered_patch > 150] = 1
    binary_patch = flood(binary_patch, (20, 20), connectivity=1)
    stone_pixels_num = binary_patch.sum()
    patch[binary_patch == 0] = 0
    hu_sum = patch.sum()
    mean_hu = hu_sum/stone_pixels_num
    max_hu = patch.max()
    area_px = stone_pixels_num

    length = get_stone_length(binary_patch)
    # print("Mean HU {}, max HU {}, area {}px, length {}".format(mean_hu, max_hu, area_px, length))
    return mean_hu, max_hu, area_px, length


def find_stone_candidates(path, labels, folder_name, mode='train', min=None):
    w, h, d = 20, 20, 7

    scan, _, frame_ids = load_scan(path)
    # return None, frame_ids
    binary_volume = []
    hu_volume = []
    for scan_slice in scan:
        hu_scan, metadata = prepare_slice(scan_slice)
        if hu_scan is None:
            continue
        binary_scan = binarize_slice(hu_scan)
        binary_volume.append(binary_scan)
        hu_volume.append(hu_scan)

    binary_volume = np.stack(binary_volume, axis=2)
    candidates_objects = find_3d_cc(binary_volume[:, :, :], hu_volume,
                                    [float(metadata['pixel_spacing'][0]),
                                     float(metadata['pixel_spacing'][0]),
                                     float(metadata['slice_thickness'])], min=min)

    if mode == 'detection':
        volumes_to_classify = []
        stats = []
        centers = []
        for i, candidate in enumerate(candidates_objects):
            key_patch_c = candidate['centroid']
            if (key_patch_c[0] - d < 0) or (key_patch_c[0] + d >= len(hu_volume)) or \
                    (key_patch_c[2] - w < 0) or (key_patch_c[2] + w > 512) or \
                    (key_patch_c[1] - h < 0) or (key_patch_c[1] + h > 512):
                continue

            volume = [[y[key_patch_c[1] - h: key_patch_c[1] + h] for y in x[key_patch_c[2] - w: key_patch_c[2] + w]] for
                      x in hu_volume[key_patch_c[0] - d: key_patch_c[0] + d]]
            volumes_to_classify.append(volume)
            mean_hu, max_hu, area_px, length = analyze_stone(np.array(volume[d]))
            stats.append([mean_hu, max_hu, area_px*metadata['pixel_spacing'][0]*metadata['pixel_spacing'][1],
                          length*metadata['pixel_spacing'][0]])
            centers.append(key_patch_c) # z, x, y
        return volumes_to_classify, frame_ids, stats, centers

    labeled_candidates, is_any_positive = label_candidates(candidates_objects, labels, frame_ids,
                                                           binary_scan.shape[0], binary_scan.shape[1])
    positives = [obj for obj in labeled_candidates if obj['label']]
    print("Found {}/{} stones".format(len(positives), len(labels)))
    print("Found {} other candidates".format(len(labeled_candidates) - len(positives)))
    if mode == 'test':
        labels = []
        volumes_to_classify = []
        stats = []
        centers = []
        for i, candidate in enumerate(candidates_objects):
            # print("BBOX size ({},{},{})".format(candidate['bbox'][4], candidate['bbox'][5], candidate['bbox'][3]))
            key_patch_c = candidate['centroid']
            if (key_patch_c[0] - d < 0) or (key_patch_c[0] + d >= len(hu_volume)) or \
                    (key_patch_c[2] - w < 0) or (key_patch_c[2] + w > 512) or \
                    (key_patch_c[1] - h < 0) or (key_patch_c[1] + h > 512):
                continue

            volume = [[y[key_patch_c[1] - h: key_patch_c[1] + h] for y in x[key_patch_c[2] - w: key_patch_c[2] + w]] for
                      x in hu_volume[key_patch_c[0] - d: key_patch_c[0] + d]]
            volumes_to_classify.append(volume)
            labels.append(candidate['label'])
            mean_hu, max_hu, area_px, length = analyze_stone(np.array(volume[d]))
            stats.append([mean_hu, max_hu, area_px*metadata['pixel_spacing'][0]*metadata['pixel_spacing'][1],
                          length*metadata['pixel_spacing'][0]])
            centers.append(key_patch_c) # z, x, y
        return volumes_to_classify, frame_ids, labels, stats, centers

    if not is_any_positive:
        return []

    negatives = [obj for obj in labeled_candidates if not obj['label']]
    max_negative_count = min(len(negatives), len(positives) * 5)
    random_negative = random.sample(negatives, max_negative_count)
    labeled_candidates = positives
    labeled_candidates.extend(random_negative)
    new_labels = []
    label_index = 0

    for candidate in labeled_candidates:
        key_patch_c = candidate['centroid']
        hu_volume = np.array(hu_volume)
        if (key_patch_c[0] - d < 0) or (key_patch_c[0] + d >= len(hu_volume)) or \
                (key_patch_c[2] - w < 0) or (key_patch_c[2] + w > hu_scan.shape[0]) or \
                (key_patch_c[1] - h < 0) or (key_patch_c[1] + h > hu_scan.shape[1]):
            continue
        volume = hu_volume[key_patch_c[0] - d: key_patch_c[0] + d, key_patch_c[2] - w: key_patch_c[2] + w,
                 key_patch_c[1] - h: key_patch_c[1] + h]

        np.save(folder_name + ('/p_' if candidate['label'] else '/n_') + str(label_index), volume)
        new_labels.append(" ".join([folder_name + ('/p_' if candidate['label'] else '/n_') + str(label_index)
                                    + ".npy", str(candidate['label']),
                                    str(candidate['centroid'][1]),
                                    str(candidate['centroid'][2]),
                                    str(candidate['centroid'][0])]) + "\n")
        label_index += 1
        if candidate['label']:
            for rot_volume in rotate_volume(volume):
                np.save(folder_name + ('/p_' if candidate['label'] else '/n_') + str(label_index), rot_volume)
                new_labels.append(" ".join([folder_name + ('/p_' if candidate['label'] else '/n_') + str(label_index)
                                            + ".npy", str(candidate['label']),
                                            str(candidate['centroid'][1]),
                                            str(candidate['centroid'][2]),
                                            str(candidate['centroid'][0])]) + "\n")
                label_index += 1

    return new_labels


def get_immediate_subdirectories(a_dir):
    subdir = [os.path.join(a_dir, name) for name in os.listdir(a_dir)
              if os.path.isdir(os.path.join(a_dir, name))]
    if len(subdir) == 0:
        return [a_dir]
    return subdir


def main(path, mode='train', min=None):
    all_directories = get_immediate_subdirectories(path)
    for i, directory in enumerate(all_directories):
        print("{}/{} - {}".format(i, len(all_directories), os.path.basename(directory)))
        folder_name = "train/processed_" + os.path.basename(directory)
        os.makedirs(folder_name, exist_ok=True)
        labels = annotation_parser.parse_file(directory)

        if mode == 'detection':
            return find_stone_candidates(directory, labels, folder_name, mode='detection', min=min)
        if mode == 'test':
            volumes, frame_ids, labels, hus, centers = find_stone_candidates(directory, labels, folder_name, mode='test', min=min)
            # with open('labels_test_' + os.path.basename(path) + '.txt', 'a') as file_object:
            #     file_object.writelines(new_labels)
            return volumes, frame_ids, labels, hus, centers
        else:
            new_labels = find_stone_candidates(directory, labels, folder_name)
            with open('labels_train_nobed.txt', 'a') as file_object:
                file_object.writelines(new_labels)

# main('D:\\data')
