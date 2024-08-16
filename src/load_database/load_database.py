import os
import glob
import numpy as np
from tqdm import tqdm
import json
import faiss
from typing import List, Tuple, Dict# mới thêm OCR




# add database into faiss indexing
def faiss_indexing(db: list, feature_dimension: int) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(feature_dimension)

    for idx, instance in enumerate(db):
        video_name, idx, feat_vec = instance

        feat_vec /= np.linalg.norm(feat_vec)
        feat_vec = np.float32(feat_vec)

        index.add(feat_vec.reshape(-1, 1).T)
    return index


# Load database from embeddings
def Database(PATH_TO_CLIPFEATURES: str) -> List[Tuple[str, int, np.ndarray],]:
    data_base = []
    for name_file_feature in tqdm(sorted(os.listdir(PATH_TO_CLIPFEATURES))):
        vid_name = name_file_feature.split('.')[0]
        features = np.load(os.path.join(PATH_TO_CLIPFEATURES, name_file_feature))
        for idx, feat in enumerate(features, 1):
            instance = (vid_name, idx, feat)
            data_base.append(instance)
    return data_base

####### Mới thêm OCR
def load_databaseOCR(PATH_TO_DB: str) -> Dict[Dict[str,str],]:
        database = []
        for vid in tqdm(os.listdir(PATH_TO_DB)):
            path_to_file = os.path.join(PATH_TO_DB, vid)
            with open(path_to_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            database.append((vid, data))
        return database

#####Mới thêm 11/8/2024 obj count
def load_databaseObjectCount_Slow(PATH_TO_DB:str)-> Dict[Dict[str,str],]:
    data_base=[]
    for name_file_feature in tqdm(sorted(os.listdir(PATH_TO_DB))):
        vid_name=name_file_feature.split('.')[0]
        features=np.load(os.path.join(PATH_TO_DB,name_file_feature))
        for idx,feat in enumerate(features,1):
            instance=(vid_name,idx,feat)
            data_base.append(instance)
    return data_base
def load_databaseObjectCount_Fast(PATH_TO_DB:str)-> Dict[Dict[str,str],]:
    class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

    data_base=[]
    num_obj_per_cls_max=20
    for idx,class_name in enumerate(class_names):
        data_base.append([])
        for i in range(num_obj_per_cls_max):
            data_base[idx].append(set())
    
    for name_file_feature in tqdm(sorted(os.listdir(PATH_TO_DB))):
        vid_name=name_file_feature.split('.')[0]
        features_vid=np.load(os.path.join(PATH_TO_DB,name_file_feature))
        for id_img,feature_img in enumerate(features_vid,1):
            for id_cls,num_obj_per_cls in enumerate(feature_img):
                if num_obj_per_cls!=0:
                    data_base[id_cls][num_obj_per_cls].add((vid_name,id_img))




    return data_base

def load_databaseASR(PATH_TO_DB: str):
    database=[]
    for vid in tqdm(sorted(os.listdir(PATH_TO_DB))):
        video_path = os.path.join(PATH_TO_DB, vid)
        with open(video_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    database.append((vid[:-5],data))
    return database