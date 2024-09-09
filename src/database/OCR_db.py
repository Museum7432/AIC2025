import os
import glob
import numpy as np
from tqdm import tqdm
import json
import faiss
from typing import List, Tuple, Dict  # mới thêm OCR

from helpers import elastic_client, vietnamese_index_settings
from utils import list_file_recursively
from elasticsearch import helpers


class OcrDB:
    def __init__(self, OCR_base_path="./texts_extracted", remove_old_index=False):

        self.remove_old_index = remove_old_index

        self.db = self.load_ocr(OCR_base_path)

        self.elastic_client = elastic_client

        self.create_index(OCR_base_path)

    def create_index(self, base_path):

        if self.elastic_client.indices.exists(index="ocr"):
            print("OCR index has already exist")

            if not self.remove_old_index:
                return

            print("delete old OCR index")
            self.elastic_client.indices.delete(index="ocr", ignore=[400, 404])

        self.elastic_client.indices.create(index="ocr", body=vietnamese_index_settings)

        ocr_relative_path = list_file_recursively(base_path)

        for ocr_file_lp in ocr_relative_path:

            if not ocr_file_lp.endswith(".json"):
                raise ValueError(f"unrecognized ocr file extension {ocr_file_lp}")

            video_path = os.path.join(base_path, ocr_file_lp)

            vid_name = ocr_file_lp.split(".")[0]

            with open(video_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            docs = []

            for k, v in data.items():
                text = v
                text = text.lower()
                text = text.replace("\n", " ")
                text = text.replace("_", " ")

                document = {
                    "vid_name": vid_name,
                    "keyframe_id": int(k.split(".")[0]),
                    "text": text,
                }

                docs.append(document)

            helpers.bulk(self.elastic_client, docs, index="ocr")

    def load_ocr(self, OCR_base_path):
        database = []
        ocr_relative_path = list_file_recursively(OCR_base_path)
        for vid in ocr_relative_path:
            path_to_file = os.path.join(OCR_base_path, vid)

            vid_name = os.path.join(vid.split("_")[0], vid[:-5])

            with open(path_to_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            database.append((vid_name, data))

        return database
