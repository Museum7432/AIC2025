import math

from helpers import elastic_client
from elasticsearch import helpers
import os
import numpy as np
from utils import list_file_recursively

# fmt: off
# List of hex colors
hex_color_pallete = [
    "#cce3e1", "#a3bab8", "#859899", "#647982", "#505966", "#363a42", "#212429",
    "#110f1f", "#2f1d42", "#2f3675", "#365680", "#538abd", "#78c2db", "#82eff5",
    "#c4ffff", "#ffffff", "#fcff5c", "#a4eb84", "#55c281", "#408a91", "#75265c",
    "#ab3057", "#ed2139", "#ff6052", "#ff8636", "#fab941", "#f0d787", "#cfa25d",
    "#b0734a", "#8f4d34", "#803636", "#592444"
]
# fmt: on
# hex_color_pallete = [s[1:] for s in hex_color_pallete]


# fmt: off
class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                     'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
                     'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
                     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
                     'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
                     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
                     'hair drier', 'toothbrush']
# fmt: on


class_names += hex_color_pallete

class_names = [s.replace(" ", "_") for s in class_names]

max_len = max([len(s) for s in class_names])


# .e.g: 'bicycle________'
padded_class_names = [
    "".join([s] + ["_"] * (max_len - len(s) + 1)) for s in class_names
]


# .e.g: 'bicycle_______C'
class_indicator = class_n = [s[:-1] + "C" for s in padded_class_names]


def encode_pos(class_idx, row, col):
    # .e.g: 'parking_meter_C parking_meter__rrr|ccccccc'
    # which mean a parking_meter at cell (2, 6)
    return (
        class_indicator[class_idx]
        + " "
        + padded_class_names[class_idx]
        + "".join(["r"] * (row + 1))
        + "|"
        + "".join(["c"] * (col + 1))
    )


def convert_color_code(color_code):
    # a0#ed2139
    # => '#ed2139_______C #ed2139________r|c'

    # for loading from the file
    position_tag, color_class = color_code.split("#")

    color_class = "#" + color_class

    assert color_class in class_names, color_class
    assert len(position_tag) == 2

    # a0
    row, col = position_tag

    row = ord(row) - ord("a")

    col = int(col)

    class_idx = class_names.index(color_class)

    return encode_pos(class_idx, row, col)


def box_cord2cells(
    cordinate,
    max_height=1,
    max_width=1,
    cell_height=7,
    cell_width=7,
    use_percentage_cord=True,
):
    # cordinate should be cordinate of a box
    # map from screen cordinate into cell cordinate
    # and return a list of cells that encapsulate the object

    # cordinate: (xmin, ymin, xmax, ymax)

    xmax, ymax, xmin, ymin = cordinate

    if not use_percentage_cord:
        xmin /= max_width
        xmax /= max_width

        ymin /= max_height
        ymax /= max_height

    cxmin = xmin * cell_width
    cxmax = xmax * cell_width

    cymin = ymin * cell_height
    cymax = ymax * cell_height

    cxmin = math.floor(cxmin)
    cymin = math.floor(cymin)

    cxmax = math.ceil(cxmax)
    cymax = math.ceil(cymax)

    re = []
    for i in range(cxmin, cxmax + 1):
        for j in range(cymin, cymax + 1):
            re.append((j, i))

    return re


class ObjectLocationDB:
    def __init__(
        self, object_loc_base_path, color_code_base_path, remove_old_index=False
    ):
        self.elastic_client = elastic_client

        self.remove_old_index = remove_old_index
        # self.remove_old_index = True

        self.prefix_len = len(padded_class_names[0])

        self.create_index(object_loc_base_path, color_code_base_path)

    def create_index(self, object_loc_base_path, color_code_base_path):
        if self.elastic_client.indices.exists(index="color_obj_loc"):
            print("color_obj_loc index has already exist")

            if not self.remove_old_index:
                return

            print("delete old color_obj_loc index")
            self.elastic_client.indices.delete(index="color_obj_loc", ignore=[400, 404])

        self.elastic_client.indices.create(
            index="color_obj_loc",
            body={
                "settings": {
                    "analysis": {"analyzer": "whitespace"},
                    "similarity": {
                        "default": {
                            "type": "boolean"  # This specifies the use of TF-IDF
                        }
                    },
                },
            },  # we should only split on whitespace
        )

        color_code_relative_path = list_file_recursively(color_code_base_path)

        for color_lp in color_code_relative_path:
            if not color_lp.endswith(".txt"):
                raise ValueError(f"unrecognized color code file extension {color_lp}")

            vid_name = color_lp.split(".")[0]

            documents = []

            obj_loc_lp = os.path.join(color_lp.replace(".txt", ""), "labels")
            
            color_code_path = os.path.join(color_code_base_path, color_lp)

            object_loc_path = os.path.join(object_loc_base_path, obj_loc_lp)

            assert os.path.isdir(object_loc_path), f"dir not found: {object_loc_path}"

            # load the color code
            with open(color_code_path, newline="") as file:
                lines = [line.rstrip() for line in file]

                frames_color_code = [l.split() for l in lines]

                frames_color_code = [
                    [convert_color_code(obj) for obj in f] for f in frames_color_code
                ]

                frames_color_code = [" ".join(f) for f in frames_color_code]

            # load object location

            nun_frames = len(frames_color_code)

            frames_obj_loc = []

            for frame_idx in range(nun_frames):
                frame_path = os.path.join(object_loc_path, f"{str(frame_idx)}.txt")

                if not os.path.isfile(frame_path):
                    frames_obj_loc.append("")
                    continue
                # assert os.path.isfile(frame_path), frame_path

                # all encoded objects in a frame
                objects = ""

                with open(frame_path, newline="") as file:
                    lines = [line.rstrip() for line in file]

                    for l in lines:
                        l = l.split()

                        class_idx = int(l[0])

                        box_cordinate = l[1:]
                        box_cordinate = [float(i) for i in box_cordinate]

                        assert len(box_cordinate) == 4

                        # list of cells that fit the object
                        cells = box_cord2cells(box_cordinate)

                        for row, col in cells:

                            objects = " ".join(
                                [objects, encode_pos(class_idx, row, col)]
                            )

                frames_obj_loc.append(objects)

            assert len(frames_color_code) == len(frames_obj_loc)
            enocded_frames = [i + j for i, j in zip(frames_color_code, frames_obj_loc)]

            docs = [
                {
                    "vid_name": vid_name,
                    "keyframe_id": idx,
                    "text": text,
                }
                for idx, text in enumerate(enocded_frames)
            ]

            helpers.bulk(self.elastic_client, docs, index="color_obj_loc")

    def enocde_pos(self, class_idx, box_cordinate):
        cells = box_cord2cells(box_cordinate)

        objects = ""
        for row, col in cells:
            objects = " ".join([objects, encode_pos(class_idx, row, col)])

        return objects
