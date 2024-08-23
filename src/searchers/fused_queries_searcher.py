from typing import Union, List, Dict

import torch
import faiss

import numpy as np

from PIL import Image
import heapq

from database import EmbeddingsDB
from encoders import ClipEncoder, BlipEncoder


def fuesd_queries_distance(frames_embs, queries_embs, queries_weights=None):
    # frames_embs (#frames, #dims)
    # queries_embs (#queries, #dims)

    # the final distance of each frame should be independent of other frames

    # (#queries, #frame)
    pairwise_distance = torch.exp(queries_embs @ frames_embs.T)

    if queries_weights is not None:
        pairwise_distance = pairwise_distance * queries_weights[:, None]

    # (#frame)
    # distance = torch.log(pairwise_distance.mean(0))
    distance = pairwise_distance.mean(0)

    return distance


class FusedSearcher:
    def __init__(
        self,
        embs_db: EmbeddingsDB,
        encoder: Union[ClipEncoder, BlipEncoder],
        batch_size: int = 2048,
    ):

        self.db = embs_db
        self.encoder = encoder

        self.batch_size = batch_size

        self.fused_embs = embs_db.fused_embs

        self.device = self.fused_embs.device

        assert torch.is_tensor(self.fused_embs)

        self.num_sections = len(self.fused_embs) // batch_size + 1

        self.batches = torch.chunk(self.fused_embs, self.num_sections)

    def vectors_search(self, v_queries, topk=5, queries_weights=None):
        # perform linear search
        # return topk instances with the minimum total
        # distance to all queries

        # v_queries should be a numpy array

        # to tensor
        # (#queries, dim)
        v_queries = torch.from_numpy(v_queries).to(self.device)

        if queries_weights is not None:
            queries_weights = torch.tensor(queries_weights).to(self.device)

        current_index = 0

        results = []

        for batch in self.batches:
            distances = fuesd_queries_distance(batch, v_queries, queries_weights).cpu().tolist()

            batch_ids = [i + current_index for i in range(len(batch))]

            for idx, dist in zip(batch_ids, distances):
                results.append((idx, dist))

            if len(results) > topk:
                results = heapq.nlargest(topk, results, key=lambda x: x[-1])

            current_index += len(batch)

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
