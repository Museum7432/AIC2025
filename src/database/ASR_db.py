import os
import glob
import numpy as np
from tqdm import tqdm
import json
import faiss
from typing import List, Tuple, Dict
from utils import list_file_recursively

from helpers import elastic_client, vietnamese_index_settings


from elasticsearch import helpers


class AsrDB:
    def __init__(self, asr_base_path="", remove_old_index=False):
        self.remove_old_index = remove_old_index

        # self.db = self.load_asr(asr_base_path)

        self.elastic_client = elastic_client

        self.create_index(asr_base_path)

        self.db = self.load_asr(asr_base_path)

    def create_index(self, base_path):
        if self.elastic_client.indices.exists(index="asr"):
            print("ASR index has already exist")

            if not self.remove_old_index:
                return

            print("delete old asr index")
            self.elastic_client.indices.delete(index="asr", ignore=[400, 404])

        self.elastic_client.indices.create(index="asr", body=vietnamese_index_settings)

        asr_relative_path = list_file_recursively(base_path)

        for asr_file_lp in asr_relative_path:

            if not asr_file_lp.endswith(".json"):
                raise ValueError(f"unrecognized asr file extension {asr_file_lp}")

            video_path = os.path.join(base_path, asr_file_lp)

            vid_name = asr_file_lp.split(".")[0]

            with open(video_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            docs = []
            for row in data:
                document = {
                    "vid_name": vid_name,
                    "start": row["start"],
                    "end": row["end"],
                    "frame_id": row["id"],
                    "text": row["text"],
                }

                docs.append(document)

            helpers.bulk(self.elastic_client, docs, index="asr")

    def load_asr(self, base_path):
        database = []
        embs_relative_path = list_file_recursively(base_path)

        for embs_lp in embs_relative_path:
            if not embs_lp.endswith(".json"):
                raise ValueError(f"unrecognized embedding file extension {embs_path}")

            video_path = os.path.join(base_path, embs_lp)
            vid_name = embs_lp.split(".")[0]
            with open(video_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            database.append((vid_name, data))
        return database
