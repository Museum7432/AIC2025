from typing import Union, List, Dict

import torch
import faiss

import numpy as np

from PIL import Image
import heapq

from database import EmbeddingsDB
from encoders import ClipEncoder, BlipEncoder


def cumargmax(a):
    m = np.maximum.accumulate(a)
    x = np.arange(a.shape[0])
    x[1:] *= m[:-1] < m[1:]
    np.maximum.accumulate(x, axis=0, out=x)
    return x


def temporal_matching(pairwise_sim, match_first=False, return_match_ids=False):
    # pairwise_sim: (#queries, #frame)
    num_queries, num_frames = pairwise_sim.shape

    if torch.is_tensor(pairwise_sim):
        pairwise_sim = pairwise_sim.numpy()

    # match_first: filp both the queries and the frames before matching
    if match_first:
        pairwise_sim = np.flip(pairwise_sim, axis=(0, 1))

    score = None

    traces = []

    for i in range(num_queries):
        if i == 0:
            # the first query
            score = pairwise_sim[0]

        else:
            cummax_indices = cumargmax(score)

            # roll the cummax so that the same frame cannot be selected twice
            cummax_indices = cummax_indices[:-1]

            shifted_len = num_frames - len(cummax_indices)

            cummax_values = score[cummax_indices]

            score = cummax_values + pairwise_sim[i][shifted_len:]

            if return_match_ids:
                # save the best previous frame index
                traces.append(
                    [0] * (shifted_len) + (cummax_indices + shifted_len - 1).tolist()
                )

    shifted_len = num_frames - len(score)

    score = np.pad(score, pad_width=(shifted_len, 0), constant_values=float("-Inf"))
    if match_first:
        score = np.flip(score)

    if not return_match_ids:
        return score, [[]] * len(score)

    matched_ids = [[i] for i in range(num_frames)]

    for t in traces[::-1]:
        for i in range(len(matched_ids)):
            matched_ids[i].append(t[matched_ids[i][-1]])

    matched_ids = [a[::-1] for a in matched_ids]

    if match_first:
        matched_ids = np.array(matched_ids)
        matched_ids = num_frames - 1 - np.flip(matched_ids, axis=(0, 1))
        matched_ids = matched_ids.tolist()

    return score, matched_ids


def temporal_matching_pytorch(pairwise_sim, match_first=False, return_match_ids=False):
    # pairwise_sim: (#queries, #frame)
    num_queries, num_frames = pairwise_sim.shape

    # match_first is not supported since flip is quite slow in pytorch

    score = None

    traces = []

    for i in range(num_queries):
        if i == 0:
            # the first query
            score = pairwise_sim[0]

        else:
            
            cummax_values, cummax_indices = torch.cummax(score, dim=0)

            # cummax_indices = cumargmax(score)

            # roll the cummax so that the same frame cannot be selected twice
            cummax_indices = cummax_indices[:-1]

            shifted_len = num_frames - len(cummax_indices)

            # cummax_values = score[cummax_indices]
            cummax_values = cummax_values[:-1]

            score = cummax_values + pairwise_sim[i][shifted_len:]

            if return_match_ids:
                # save the best previous frame index
                traces.append(
                    [0] * (shifted_len) + (cummax_indices + shifted_len - 1).cpu().tolist()
                )

    shifted_len = num_frames - len(score)

    score = torch.nn.functional.pad(score, pad=(shifted_len, 0), value=float("-Inf"))

    if not return_match_ids:
        return score, [[]] * len(score)

    matched_ids = [[i] for i in range(num_frames)]

    for t in traces[::-1]:
        for i in range(len(matched_ids)):
            matched_ids[i].append(t[matched_ids[i][-1]])

    matched_ids = [a[::-1] for a in matched_ids]

    return score.cpu().tolist(), matched_ids

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

    def vectors_search(
        self,
        v_queries,
        topk=5,
        queries_weights=None,
        match_first=False,
        return_match_ids=True,
        **kwargs
    ):
        # for each video, match each query with its associated frame
        # in a consecutive order

        # match_first: the score of each frame is the maximum score
        # of all sequences of frames that start with that frame
        # if false then the maximum score of all sequences that end with
        # that frame

        # v_queries should be a numpy array

        # to tensor
        # (#queries, dim)
        v_queries = torch.from_numpy(v_queries).to(self.device)

        # normalize the query
        v_queries = torch.nn.functional.normalize(v_queries, dim=-1)

        if queries_weights is not None:
            queries_weights = torch.tensor(queries_weights).to(self.device)

        current_index = 0

        results = []

        for _start, _end in self.db.vid_idx2idx:
            # for each video

            # (seqlen, dim)
            vid_embs = self.fused_embs[_start : _end + 1]

            # (#queries, #frame)
            # pairwise_sim = v_queries @ vid_embs.T
            pairwise_sim = torch.exp(v_queries @ vid_embs.T)
            # Taylor expansion
            # pairwise_sim = v_queries @ vid_embs.T
            # pairwise_sim = 1 + pairwise_sim + pairwise_sim**2 / 2


            if queries_weights is not None:
                pairwise_sim = pairwise_sim * queries_weights[:, None]

            naive_highest_score = pairwise_sim.max(-1).values.sum()

            if len(results) >= topk and results[-1][-1] >= naive_highest_score:
                # if the highest sim in the batch is smaller than the lowest sim
                # found in the topk
                current_index += len(vid_embs)
                continue

            # (#frame)
            # score, matched_ids = temporal_matching(
            #     pairwise_sim, match_first=match_first, return_match_ids=return_match_ids
            # )
            score, matched_ids = temporal_matching_pytorch(
                pairwise_sim, return_match_ids=return_match_ids
            )
            batch_ids = [i + current_index for i in range(len(score))]

            for idx, sim, mids in zip(batch_ids, score, matched_ids):
                results.append((idx, mids, sim))

            if len(results) > topk:
                results = heapq.nlargest(topk, results, key=lambda x: x[-1])
                results.sort(reverse=True, key=lambda x: x[-1])

            current_index += len(score)

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

        return query_results

    def search_by_texts(self, texts, topk=5):
        # batch search by text
        v_queries = self.encoder.encode_texts(texts, normalization=True)

        return self.vectors_search(v_queries, topk=topk)

    def search_by_images(self, images, topk=5):
        # batch search by images
        v_queries = self.encoder.encode_images(images, normalization=True)

        return self.vectors_search(v_queries, topk=topk)
