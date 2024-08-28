from typing import Union, List, Dict

import torch
import faiss

import numpy as np

from PIL import Image
import heapq

from database import EmbeddingsDB
from encoders import ClipEncoder, BlipEncoder

# temporal_matching but return the frame associated with each query
# def get_best_matched_pair(pairwise_sim):
#   # pairwise_sim: (#queries, #frame)

#   num_query, num_frame = pairwise_sim.shape

#   score = np.zeros(num_frame)
#   trace = np.zeros_like(pairwise_sim, dtype="int")

#   for i in range(num_query):
#     trace[i] = cumargmax(score)
#     score = np.maximum.accumulate(score) + pairwise_sim[i]

#   best_score = np.max(score)

#   final_trace = [np.argmax(score)]

#   for t in trace[1:][::-1]:
#     final_trace.append(t[final_trace[-1]])

#   return best_score, final_trace[::-1]


def temporal_matching(pairwise_sim):
    # pairwise_sim: (#queries, #frame)
    num_queries, num_frames = pairwise_sim.shape

    score = None

    traces = []

    for i in range(num_queries):

        if i == 0:
            # the first query
            score = pairwise_sim[0]

            traces.append([j for j in range(num_frames)])

        else:
            best_prev_score = torch.cummax(score, dim=0)

            # roll the cummax so that the same frame cannot be selected twice
            score = best_prev_score.values[:-1] + pairwise_sim[i][i:]

            # save the best previous frame index
            traces.append([0] * (i) + (best_prev_score.indices[:-1].cpu() + i).tolist())

    score = score.cpu().tolist()

    score = [float("Inf")] * num_queries + score

    matched_ids = [[i] for i in range(num_frames)][num_queries:]

    for t in traces[1:][::-1]:
        for i in range(len(matched_ids)):
            matched_ids[i].append(t[matched_ids[i][-1]])

    matched_ids = [[]] * num_queries + matched_ids

    matched_ids = [a[::-1] for a in matched_ids]

    return score, matched_ids


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
        self, v_queries, topk=5, queries_weights=None, return_first=False
    ):
        # for each video, match each query with its associated frame
        # in a consecutive order

        # return_first: the score of each frame is the maximum score
        # of all sequences of frames that start with that frame
        # if false then the maximum score of all sequences that end with
        # that frame

        # v_queries should be a numpy array

        # to tensor
        # (#queries, dim)
        v_queries = torch.from_numpy(v_queries).to(self.device)

        if return_first:
            v_queries = torch.flip(v_queries, dims=0)

        if queries_weights is not None:
            queries_weights = torch.tensor(queries_weights).to(self.device)

        current_index = 0

        results = []

        for _start, _end in self.db.vid_idx2idx:
            # for each video

            # (seqlen, dim)
            vid_embs = self.fused_embs[_start : _end + 1]

            if return_first:
                vid_embs = torch.flip(vid_embs, dims=0)

            # (#queries, #frame)
            pairwise_sim = np.exp(v_queries @ vid_embs.T)

            # (#frame)
            score, matched_ids = temporal_matching(pairwise_sim)

            if return_first:
                score = torch.flip(score, dims=0)

                for i in range(len(matched_ids)):
                    matched_ids[i] = [len(vid_embs) - j for j in matched_ids[i]]

            batch_ids = [i + current_index for i in range(len(score))]

            for idx, sim in zip(batch_ids, score):
                results.append((idx, sim))

            if len(results) > topk:
                results = heapq.nlargest(topk, results, key=lambda x: x[-1])

            current_index += len(score)

        query_results = []
        for idx, dist in results:
            vid_name, frame_idx = self.db.get_info(idx)

            query_results.append(
                {
                    "score": dist,
                    "keyframe_id": frame_idx.item(),
                    "video_name": vid_name,
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
