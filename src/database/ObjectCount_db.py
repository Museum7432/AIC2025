import os
import glob
import numpy as np
from tqdm import tqdm
import json
import faiss
from typing import List, Tuple, Dict  # mới thêm OCR
from utils import list_file_recursively


class ObjectCountDB:
    def __init__(self, objcount_base_path=""):

        self.fast_db = self.load_fast_db(objcount_base_path)
        self.slow_db = self.load_slow_db(objcount_base_path)

    def load_slow_db(self, base_path):
        data_base = []
        embs_relative_path = list_file_recursively(base_path)
       
        for embs_lp in embs_relative_path:
            if not embs_lp.endswith(".npy"):
                raise ValueError(f"unrecognized embedding file extension {embs_lp}")

            video_path = os.path.join(base_path, embs_lp)
            vid_name = embs_lp.split(".")[0]
            features = np.load(video_path)
            for idx, feat in enumerate(features, 0):
                instance = (vid_name, idx, feat)
                data_base.append(instance)
        return data_base

    def load_fast_db(self, base_path):

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

        data_base = []
        num_obj_per_cls_max = 100
        for idx, class_name in enumerate(class_names):
            data_base.append([])
            for i in range(num_obj_per_cls_max):
                data_base[idx].append(set())
        embs_relative_path = list_file_recursively(base_path)
        
        
     
       
        for embs_lp in tqdm(embs_relative_path):
            if not embs_lp.endswith(".npy"):
                raise ValueError(f"unrecognized embedding file extension {embs_lp}")

            video_path = os.path.join(base_path, embs_lp)
            
            vid_name = embs_lp.split(".")[0]
            features_vid = np.load(video_path)
            for id_img, feature_img in enumerate(features_vid, 0):
                for id_cls, num_obj_per_cls in enumerate(feature_img):
                    if num_obj_per_cls != 0:
                      
                        data_base[id_cls][num_obj_per_cls].add((vid_name, id_img))
                        
        # for i, instance in enumerate(data_base[0]): 
        #     print(f'length of set {i}' , len(instance))

        return data_base
