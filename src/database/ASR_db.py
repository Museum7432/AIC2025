import os
import glob
import numpy as np
from tqdm import tqdm
import json
import faiss
from typing import List, Tuple, Dict


class AsrDB:
    def __init__(self, asr_base_path=""):

        self.db = self.load_asr(asr_base_path)

    def load_asr(self, base_path):
        database = []
        for vid in tqdm(sorted(os.listdir(base_path))):
            video_path = os.path.join(base_path, vid)
            with open(video_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        database.append((vid[:-5], data))
        return database
