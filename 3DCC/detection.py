import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import annotation_parser
from model import CTNetModel

from utils import get_immediate_subdirectories, main


def evaluate_for_size_range(data, path, min, max, mode='test'):
    annotations = annotation_parser.parse_file(path, min=min, max=max)
    tps_per_image, fps_per_image, fns_per_image, negs_per_image = [0 for i in range(20)], \
                                                                  [0 for i in range(20)], \
                                                                  [0 for i in range(20)], \
                                                                  [0 for i in range(20)]
    volumes, frame_ids, labels, stats, centers = data
    found_stones_stats = []
    for volume, label, stat, center in zip(volumes, labels, stats, centers):
        input = np.asarray(volume).astype(np.float32)
        input[input > 1000] = 1000
        input = input[:, :, :]
        input = (input[:, :, :] - (-1057)) / (1000 - (-1057))
        with torch.no_grad():
            output = model(torch.tensor(input[np.newaxis, :, :, :]))
            if mode == 'detection':
                candidate_class = torch.where(torch.squeeze(output) >= 0.85, 1, 0)
                if candidate_class.item() == 1:
                    found_stones_stats.append({
                        "frame_id": frame_ids[center[0]],
                        "position": [int(center[1]), int(center[2])],
                        "mean_hu": stat[0],
                        "max_hu": int(stat[1]),
                        "area": stat[2],
                        "size": stat[3]
                    })
            else:
                # gather data for ROC curve
                for i, th in enumerate(np.linspace(0.05, 1, 20)):
                    candidate_class = torch.where(torch.squeeze(output) >= th, 1, 0)
                    if candidate_class.item() == 1:
                        if label == 1:
                            tps_per_image[i] += 1
                            if th == 0.85:
                                found_stones_stats.append({
                                    "frame_id": frame_ids[center[0]],
                                    "position": [int(center[1]), int(center[2])],
                                    "mean_hu": stat[0],
                                    "max_hu": int(stat[1]),
                                    "area": stat[2],
                                    "size": stat[3]
                                })
                        else:
                            if label == 0:
                                fps_per_image[i] += 1
                    else:
                        negs_per_image[i] += 1


    tps_per_image = [tps_per_image[i] if tps_per_image[i] < len(annotations) else \
                         len(annotations) for i in range(len(tps_per_image))]
    fns_per_image = [len(annotations) - tps_per_image[i] for i in range(len(fns_per_image))]
    return negs_per_image, fns_per_image, tps_per_image, fps_per_image, found_stones_stats


def extract_and_detect(path, mode='test'):
    total_extraction_time = 0
    total_evaluation_time = 0
    total_candidates_count = 0
    total_axial_slice_count = 0
    tps_small, fps_small, fns_small, tns_small = [0 for i in range(20)], [0 for i in range(20)], [0 for i in range(20)], [0 for i in range(20)]

    result_list = []
    for directory in get_immediate_subdirectories(path):
        data = main(directory, mode='test')
        negs_per_image, fns_per_image, tps_per_image, fps_per_image, found_stones_stats = evaluate_for_size_range(data, directory, 3, None)
        result_list.append({
            "patient": os.path.basename(directory),
            "results": found_stones_stats})
        tns_small = [tns_small[i] + (negs_per_image[i] - fns_per_image[i]) for i in range(len(fns_per_image))]
        tps_small = [tps_small[i] + tps_per_image[i] for i in range(len(tps_per_image))]
        fns_small = [fns_small[i] + fns_per_image[i] for i in range(len(tps_per_image))]
        fps_small = [fps_small[i] + fps_per_image[i] for i in range(len(tps_per_image))]
        print("TPs {}, FPs {}, FNs {}, TNs {}".format(tps_small, fps_small, fns_small, tns_small))

    if mode == 'test':
        print("Mean extraction time {}  and mean overall evaluation time {}".format(total_extraction_time/23, total_evaluation_time/23))
        print("Mean candidates count {} and mean axial slice count {}".format(total_candidates_count/23, total_axial_slice_count/23))
        sensitivities_small = [(tps_small[i]/(tps_small[i] + fns_small[i])) if (tps_small[i] + fns_small[i]) > 0 else 0 for i in range(20)]
        specifities_small = [(fps_small[i]/23) for i in range(20)]


        plt.figure()
        plt.plot(specifities_small, sensitivities_small)
        plt.ylabel("citlivosť")
        plt.yticks(np.linspace(0.05, 1, 20))
        plt.xlabel("FP na pacienta")
        plt.show()
    print(result_list)
    with open('result_stones.json', 'w') as file_object:
        json.dump(result_list, file_object, indent=4)

DATA_DIR = 'D:/test_scans_sample'
model = CTNetModel()
model.load_state_dict(torch.load('model-simple-he-init-14.pth'))
model.eval()

extract_and_detect(DATA_DIR, mode='detection')
