import os
import glob
import numpy as np
from tqdm import tqdm
import json
import faiss
from typing import List, Tuple, Dict  # mới thêm OCR


class ObjectCountDB:
    def __init__(self, objcount_base_path=""):

        self.fast_db = self.load_fast_db(objcount_base_path)
        self.slow_db = self.load_slow_db(objcount_base_path)

    def load_slow_db(self, base_path):
        data_base = []
        for name_file_feature in sorted(os.listdir(base_path)):
            vid_name = name_file_feature.split(".")[0]
            features = np.load(os.path.join(base_path, name_file_feature))
            for idx, feat in enumerate(features, 1):
                instance = (vid_name, idx, feat)
                data_base.append(instance)
        return data_base

    def load_fast_db(self, base_path):

        # fmt: off
        class_names=[
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard',
            'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear',
            'hair_drier', 'toothbrush'
            ]
        # fmt: on

        data_base = []
        num_obj_per_cls_max = 20
        for idx, class_name in enumerate(class_names):
            data_base.append([])
            for i in range(num_obj_per_cls_max):
                data_base[idx].append(set())

        for name_file_feature in sorted(os.listdir(base_path)):
            vid_name = name_file_feature.split(".")[0]
            features_vid = np.load(os.path.join(base_path, name_file_feature))
            for id_img, feature_img in enumerate(features_vid, 1):
                for id_cls, num_obj_per_cls in enumerate(feature_img):
                    if num_obj_per_cls != 0:
                        data_base[id_cls][num_obj_per_cls].add((vid_name, id_img))

        return data_base
