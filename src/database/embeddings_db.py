import torch
import numpy as np
import faiss
from tqdm import tqdm
import json
from typing import List, Tuple, Dict
import os

from utils import list_file_recursively


def get_start_end_indices(arr):
    # arr: array of arrays
    arr_lens = [len(a) for a in arr]

    start_indices = np.cumsum([0] + arr_lens[:-1])

    end_indices = np.cumsum(arr_lens) - 1

    return np.vstack([start_indices, end_indices]).T


def get_flatten_index_mapping(arr):
    # arr: array of arrays
    arr_lens = [len(a) for a in arr]

    row_indices = np.arange(len(arr))

    return np.repeat(row_indices, arr_lens)


class EmbeddingsDB:
    """
    Store the embeddings (normalized) in
    concatenated form, as well as the mapping from
    the concatenated array index to the video keyframe index
    the index return by faiss is the same as the the concatenated array index
    """

    # TODO: map the keyframe index to the acture video frame index

    def __init__(self, embs_base_path):

        self.embs_base_path = embs_base_path
        self.embs_dim = None

        # concatenate all videos' embeddings into a single array for
        # faster access time
        # shape: (total_num_of_frames, embs_dim)
        # is a torch tensor
        self.fused_embs = None

        # map the fused array index into the videos_name index (1d np.array of int)
        # (#videos, )
        # TODO: optimize this
        self.videos_name = []
        # (total_num_of_frames,)
        self.idx2vid_idx = None

        # map video index to the start and end frame in the fused array (2d np.array of int)
        # access the videos embedding with array[start:end+1]
        # (#videos, 2)
        self.vid_idx2idx = None

        self._load_embs()

        # embs_dim, vid_idx2idx,... should be set by _load_embs
        assert len(self.videos_name) == len(self.vid_idx2idx)

    def _load_embs(self):

        assert len(self.videos_name) == 0

        frames_embs = []
        start_index = 0

        embs_relative_path = list_file_recursively(self.embs_base_path)

    

        for embs_lp in embs_relative_path:
            if not embs_lp.endswith(".npy"):
                raise ValueError(f"unrecognized embedding file extension {embs_path}")

            embs_path = os.path.join(self.embs_base_path, embs_lp)

            # load frame embeddings
            video_embs = np.load(embs_path)
            video_embs = np.float32(video_embs)

            frames_embs.append(video_embs)

            # save video name
            video_name = embs_lp.split(".")[0]

            self.videos_name.append(video_name)

            if self.embs_dim is None:
                self.embs_dim = video_embs.shape[-1]
            else:
                assert (
                        self.embs_dim == video_embs.shape[-1]
                    ), f"mismatch embedding dimension in {embs_path}"

        self.fused_embs = np.vstack(frames_embs)

        self.vid_idx2idx = get_start_end_indices(frames_embs)
        self.idx2vid_idx = get_flatten_index_mapping(frames_embs)

        # convert to tensor
        self.fused_embs = torch.from_numpy(self.fused_embs)

        # normalize the embs
        self.fused_embs = torch.nn.functional.normalize(self.fused_embs, dim=-1)

    def get_info(self, fused_frame_idx):
        # return video name, frame id
        vid_idx = self._get_vid_id(fused_frame_idx)

        vid_name = self.videos_name[vid_idx]

        _start, _end = self._get_vid_start_end(vid_idx)

        assert fused_frame_idx <= _end
        frame_idx = fused_frame_idx - _start

        return vid_name, frame_idx

    def _get_vid_id(self, fused_frame_idx):
        # faiss index to video id
        return self.idx2vid_idx[fused_frame_idx]

    def get_vid_name(self, fused_frame_idx):
        # faiss index to video name
        vid_idx = self._get_vid_id(fused_frame_idx)
        return self.videos_name[vid_idx]

    def _get_vid_start_end(self, vid_id):
        # video id to start and end faiss index
        return self.vid_idx2idx[vid_id]

    def get_vid_start_end_by_name(self, vid_name):
        # video name to start and end faiss index
        vid_id = self.videos_name.index(vid_name)
        return self._get_vid_start_end(vid_id)

    def get_vid_embs(self, vid_id):
        _start, _end = self._get_vid_start_end(vid_id)
        return self.fused_embs[_start:_end].numpy()

    def _to_fused_index(self, vid_name, frame_idx):

        _start, _end = self.get_vid_start_end_by_name(vid_name)

        return _start + frame_idx

    def get_frame_embs(self, vid_name, frame_idx):
        fused_frame_idx = self._to_fused_index(vid_name, frame_idx)

        return self.fused_embs[fused_frame_idx].numpy()

    def get_vid_embs_by_name(self, vid_name):
        vid_id = self.videos_name.index(vid_name)
        return self.get_vid_embs(vid_id)
