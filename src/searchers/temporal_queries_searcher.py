from typing import Union, List, Dict

import torch
import faiss

import numpy as np

from PIL import Image
import heapq

from database import EmbeddingsDB
from encoders import ClipEncoder, BlipEncoder
from utils import compute_similarity, get_similarity_func_id
import time


@torch.jit.script
def temporal_matching_pytorch(
    queries: torch.Tensor, frames: torch.Tensor, metric_id: int = 16
):
    """
    match queries to frames within the videos
    metric_id of 16 is dot product
    """

    # (#queries, #frame)
    pairwise_sim = compute_similarity(queries, frames, metric_id=metric_id)

    num_queries, num_frames = pairwise_sim.shape

    # match_first is not supported since flip is quite slow in pytorch

    score = pairwise_sim[0]

    traces = torch.zeros(
        (num_queries - 1, num_frames), device=pairwise_sim.device, dtype=torch.int
    )

    # score will be shifted 1 to the right
    # every loop
    shifted_len = 0

    for i in range(1, num_queries):
        cummax_values, cummax_indices = torch.cummax(score, dim=0)

        # cummax_indices = cumargmax(score)
        # roll the cummax so that the same frame cannot be selected twice
        cummax_indices = cummax_indices[:-1]

        # shifted_len = num_frames - len(cummax_indices)
        shifted_len += 1

        # cummax_values = score[cummax_indices]
        cummax_values = cummax_values[:-1]

        score = cummax_values + pairwise_sim[i][shifted_len:]

        # save the best previous frame index
        traces[i - 1][shifted_len:] = cummax_indices + shifted_len - 1

    score = torch.nn.functional.pad(score, pad=(shifted_len, 0), value=float("-Inf"))

    # the indices of the last queries
    start_index = torch.arange(num_frames, device=pairwise_sim.device, dtype=torch.int)

    steps = [start_index]

    for idx in range(num_queries - 2, -1, -1):
        t = traces[idx]
        steps.insert(0, t[steps[0]])

    steps = torch.vstack(steps).T
    # we should ignore the first num_queries - 1 rows of the steps
    return score, steps


class TemporalSearcher:
    def __init__(
        self,
        embs_db: EmbeddingsDB,
        encoder: Union[ClipEncoder, BlipEncoder],
    ):
        self.db = embs_db
        self.encoder = encoder

        self.fused_embs = embs_db.fused_embs

        self.device = self.fused_embs.device

        assert torch.is_tensor(self.fused_embs)

        # cache the view of each video tensor
        self.videos_embs = []
        for _start, _end in embs_db.vid_idx2idx:
            self.videos_embs.append(
                (self.fused_embs[_start : _end + 1], list(range(_start, _end + 1)))
            )

    def vectors_search(
        self, v_queries, topk=5, queries_weights=None, metric_type="exp_dot", **kwargs
    ):
        # for each video, match each query with its associated frame
        # in a consecutive order

        # match_first: the score of each frame is the maximum score
        # of all sequences of frames that start with that frame
        # if false then the maximum score of all sequences that end with
        # that frame
        start_time = time.time()

        metric_id = get_similarity_func_id(metric_type)

        # v_queries should be a numpy array

        # to tensor
        # (#queries, dim)
        v_queries = torch.from_numpy(v_queries).to(self.device)

        # normalize the query
        v_queries = torch.nn.functional.normalize(v_queries, dim=-1)

        if queries_weights is not None:
            queries_weights = torch.tensor(queries_weights).to(self.device)

        results = []

        for vid_embs, frames_ids in self.videos_embs:
            # vid_embs: (seqlen, dim)
            # frames_ids: (seqlen)

            # (#frame)
            score, matched_ids = temporal_matching_pytorch(
                v_queries, vid_embs, metric_id=metric_id
            )

            # if the highest score within the batch is smaller than the lowest score in result
            if len(results) >= topk and score.max() < results[-1][-1]:
                continue

            score = score.cpu().tolist()
            matched_ids = matched_ids.cpu().tolist()

            for idx, sim, mids in zip(frames_ids, score, matched_ids):
                results.append((idx, mids, sim))

            if len(results) > topk:
                results = heapq.nlargest(topk, results, key=lambda x: x[-1])
                results.sort(reverse=True, key=lambda x: x[-1])

        query_results = []
        for idx, matched_frames, dist in results:
            vid_name, frame_idx = self.db.get_info(idx)

            query_results.append(
                {
                    "score": dist,
                    "keyframe_id": frame_idx.item(),
                    "video_name": vid_name,
                    "matched_frames": matched_frames,
                }
            )

        end_time = time.time()
        print(f"temporal search Runtime: {end_time - start_time} seconds")

        return query_results

    def search_by_texts(self, texts, topk=5, metric_type="exp_dot"):
        # batch search by text
        v_queries = self.encoder.encode_texts(texts, normalization=True)

        return self.vectors_search(v_queries, topk=topk, metric_type=metric_type)

    def search_by_images(self, images, topk=5, metric_type="exp_dot"):
        # batch search by images
        v_queries = self.encoder.encode_images(images, normalization=True)

        return self.vectors_search(v_queries, topk=topk, metric_type=metric_type)
