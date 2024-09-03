import os
import glob
import numpy as np
from tqdm import tqdm
import json
import faiss
from typing import List, Tuple, Dict
from utils import list_file_recursively


class AsrDB:
    def __init__(self, asr_base_path=""):

        self.db = self.load_asr(asr_base_path)

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
            database.append((vid_name , data))
        return database
