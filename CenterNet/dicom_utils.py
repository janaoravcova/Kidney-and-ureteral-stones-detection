import os

import pydicom as pd
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter


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
            "file_name": os.path.join(path, file)
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
    return slices, file_names


def prepare_slice(original_slice, xtart=None, xstop=None, ystart=None, ystop=None):
    try:
        slice_data = original_slice.pixel_array
    except ValueError:
        return None, None
    if xtart is None:
        scan_slice = slice_data.astype(np.int16)
    else:
        scan_slice = slice_data[xtart:xstop, ystart:ystop].astype(np.int16)
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


def binarize_image(image, hu_threshold=250, filter_sigma=0.7):
    filtered_image = gaussian_filter(image, sigma=filter_sigma)
    binary_image = np.where(filtered_image > hu_threshold, 1, 0)
    binary_image = np.array(binary_image, dtype=np.int16)
    binary_image = scipy.ndimage.binary_fill_holes(binary_image).astype(int)
    return binary_image


def get_previous_id(frame_id, i):
    return ".".join(frame_id.split(".")[: -1]) + "." + str(
        int(frame_id.split(".")[-1]) - i)


def get_next_id(frame_id, i):
    return ".".join(frame_id.split(".")[: -1]) + "." + str(
        int(frame_id.split(".")[-1]) + i)
