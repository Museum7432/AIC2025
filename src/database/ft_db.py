from ftsearch import FTSearch

import torch
import numpy as np
import faiss
from tqdm import tqdm
import json
from typing import List, Tuple, Dict
import os

from utils import list_file_recursively, normalized_np


class FTdb:
    def __init__(self, embs_base_path):
        self.db = None

        self.embs_dim = None

        self.vid_name2seq_id = {}

        embs_relative_path = list_file_recursively(embs_base_path)

        for embs_lp in tqdm(embs_relative_path):
            embs_path = os.path.join(embs_base_path, embs_lp)

            if not embs_lp.endswith(".npy"):
                print(f"unrecognized embedding file extension {embs_path}")
                continue

            video_embs = np.float32(np.load(embs_path))
            # normalize it
            video_embs = normalized_np(video_embs)

            if self.embs_dim is None:
                self.embs_dim = video_embs.shape[-1]
                self._init_db()
            else:
                if self.embs_dim != video_embs.shape[-1]:
                    print(f"mismatch embedding dimension in {embs_path}")
                    continue

            # save video name
            video_name = embs_lp.split(".")[0]

            self.db.add_seq(video_embs, video_name)

            seq_id = self.db.num_seqs() - 1

            self.vid_name2seq_id[video_name] = seq_id

    def _init_db(self):
        if self.db is None:
            self.db = FTSearch(self.embs_dim)

    def get_info(self, vec_idx):
        return self.db.get_info(vec_idx)
    
    def get_frame_embs(self, vid_name, frame_idx):
        seq_id = self.vid_name2seq_id[vid_name]

        start_idx = self.db.get_seq_info(seq_id)["start_idx"]

        vec_idx = start_idx + frame_idx

        return self.db.get_vec(vec_idx)

