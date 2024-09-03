import os
import glob
import numpy as np
from tqdm import tqdm
import json
import faiss
from typing import List, Tuple, Dict# mới thêm OCR


class OcrDB:
    def __init__(self, OCR_base_path="./texts_extracted"):

        database = []
        for vid in os.listdir(OCR_base_path):
            path_to_file = os.path.join(OCR_base_path, vid)
   
            vid_name = os.path.join(vid.split('_')[0] , vid[:-5])
  

            with open(path_to_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            database.append((vid_name, data))
        
        self.db = database


    