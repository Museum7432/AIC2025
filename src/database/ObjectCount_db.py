import os
import glob
import numpy as np
from tqdm import tqdm
import json
import faiss
from typing import List, Tuple, Dict  # mới thêm OCR
from utils import list_file_recursively

from helpers import elastic_client
from elasticsearch import helpers


def get_padded_class_name(class_names):
    max_len = max([len(s) for s in class_names])

    padded_class_names = [
        "".join([s] + ["_"] * (max_len - len(s) + 1)) for s in class_names
    ]

    return padded_class_names


class ObjectCountDB:
    def __init__(self, objcount_base_path="", remove_old_index=False):
        self.remove_old_index = remove_old_index
        # self.remove_old_index = True

        self.elastic_client = elastic_client

        # fmt: off
        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
            'hair drier', 'toothbrush'
        ]
        # fmt: on
        class_names = [s.replace(" ", "_") for s in class_names]

        self.class_names = class_names

        self.padded_class_names = get_padded_class_name(self.class_names)

        # elasticsearch should exclude the class name from
        # fuzzy search
        self.prefix_len = len(self.padded_class_names[0])

        self.create_index(objcount_base_path)

        self.fast_db = self.load_fast_db(objcount_base_path)
        self.slow_db = self.load_slow_db(objcount_base_path)

    def _encode_one_obj_count(self, class_idx, obj_count):
        # .e.g: 3 person: 'person________+ person_________III'
        # when perfoming fuzzy search 3 person and 2 person should have a
        # edit distance of 1
        # in case where the counting does not work
        # it should still match the class name
        return (
            self.padded_class_names[class_idx][:-1]
            + "+ "
            + "".join([self.padded_class_names[class_idx]] + ["I"] * obj_count)
        )

    def _encode_obj_count_vec(self, obj_count_vec):
        """obj_count_vec should be a 1d vector of int"""
        assert len(obj_count_vec) == len(self.class_names)

        result = []
        for idx, count in enumerate(obj_count_vec):
            if count <= 0:
                continue
            result.append(self._encode_one_obj_count(idx, count))

        return " ".join(result)

    def create_index(self, base_path):

        if self.elastic_client.indices.exists(index="obj_count"):
            print("obj_count index has already exist")

            if not self.remove_old_index:
                return

            print("delete old obj_count index")
            self.elastic_client.indices.delete(index="obj_count", ignore=[400, 404])

        self.elastic_client.indices.create(
            index="obj_count",
            body={
                "settings": {"analysis": {"analyzer": "whitespace"}}
            },  # we should only split on whitespace
        )

        obj_count_relative_path = list_file_recursively(base_path)

        for obj_count_file_lp in obj_count_relative_path:

            if not obj_count_file_lp.endswith(".npy"):
                raise ValueError(
                    f"unrecognized obj_count file extension {obj_count_file_lp}"
                )

            video_path = os.path.join(base_path, obj_count_file_lp)

            vid_name = obj_count_file_lp.split(".")[0]

            # (#frames, #classes)
            features = np.load(video_path)

            # convert features into indexable text
            enocded_texts = [self._encode_obj_count_vec(f) for f in features]

            docs = []
            for idx, text in enumerate(enocded_texts):
                document = {
                    "vid_name": vid_name,
                    "keyframe_id": idx,
                    "text": text,
                }
                docs.append(document)

            helpers.bulk(self.elastic_client, docs, index="obj_count")

    def elastic_search_vec(self, obj_count_vec, topk):
        """obj_count_vec should be a 1d vector of int"""

        query = self._encode_obj_count_vec(obj_count_vec)

        es_query = {
            "bool": {
                "should": [
                    {
                        "match": {
                            "text": {
                                "query": query,
                            }
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "fuzziness": 1,
                                "prefix_length": self.prefix_len,  # important, or else it will
                                # perfom fuzzy search on the class name
                                "fuzzy_transpositions": False,
                                # "boost": 0.8,
                            },
                        },
                    },
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "fuzziness": 2,
                                "prefix_length": self.prefix_len,  # important, or else it will
                                # perfom fuzzy search on the class name
                                "fuzzy_transpositions": False,
                                # "boost": 0.5,
                            }
                        },
                    },
                ]
            }
        }

        hits = self.elastic_client.search(
            index="obj_count",
            query=es_query,
            size=topk,
        ).raw["hits"]["hits"]

        results = [
            {
                "video_name": d["_source"]["vid_name"],
                "keyframe_id": d["_source"]["keyframe_id"],
                # "score": score,
                "score": d["_score"],
                "text": d["_source"]["text"],
            }
            for d in hits
        ]

        return results

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
