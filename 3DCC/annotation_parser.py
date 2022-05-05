import ast
import os

import numpy as np

STONE_GUID_STRINGS = ["1e0e311f-fb21-4c6c-ac68-497623831b1d", "0047b403-b78f-430a-9cdc-beaf38b93fda",
                      "fc4429c9-e5b6-474f-bb05-ebef30403bfe", "12c0f663-438b-457c-8147-99db093ae01f"]


# result file is the only file starting with 'Result' string
def find_result_file_in_dir(directory):
    # print(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('Result'):
                # print(file)
                return file
    return None

def parse_file(INPUT_FOLDER, min=None, max=None):

    file = find_result_file_in_dir(INPUT_FOLDER)
    if file is None:
        return []
    annotations = []
    # print(INPUT_FOLDER)
    with open(INPUT_FOLDER + "/" + file) as fp:
        lines = fp.readlines()

        # crop the lines starting with 5th line as the previous contain general information about scan
        lines = lines[4:len(lines)]
        for line in lines:
            tags = line.split(';')
            frame_id = ""
            position = []
            guid = ""
            length = 0
            for tag in tags:
                try:
                    name, value = tag.split(":")
                    if name.startswith(" (0062,0020)"):
                        guid = value.strip()
                    if name.endswith("FL"):
                        position.append(ast.literal_eval(value.strip()))
                    if name.startswith(" (0008, 1155)"):
                        frame_id = value.strip()
                    if name.endswith("LT"):
                        length = float(value.strip().split(' ')[0][1:])
                except ValueError:
                    ...

            try:
                index = STONE_GUID_STRINGS.index(guid)
                order = line[-5:-2].strip()
                # no need to take the short axis
                if order.split('.')[-1] == '1':
                    continue

                if min is not None:
                    if length < min:
                        continue
                if max is not None:
                    if length > max:
                        continue
                # calculate center by taking middle point of major axis
                a, b = position

                annotations.append({
                    "frame_id": frame_id,
                    "a": a,
                    "b": b,
                    "l": length

                })
            except ValueError:
                continue
    return annotations

