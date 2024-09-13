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
    pairwise_sim = torch.exp(queries_embs @ frames_embs.T)

    if queries_weights is not None:
        pairwise_sim = pairwise_sim * queries_weights[:, None]

    # (#frame)
    # sim = torch.log(pairwise_sim.mean(0))
    sim = pairwise_sim.mean(0)

    return sim


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

        fused_embs = embs_db.fused_embs

        self.device = fused_embs.device

        assert torch.is_tensor(fused_embs)

        self.num_sections = len(fused_embs) // batch_size + 1

        self.batches = torch.chunk(fused_embs, self.num_sections)

    def vectors_search(self, v_queries, topk=5, queries_weights=None, **kwargs):
        # perform linear search
        # return topk instances with the minimum total
        # distance to all queries

        # v_queries should be a numpy array
        # and should not be normalized if feature_normalization is on

        # to tensor
        # (#queries, dim)
        v_queries = torch.from_numpy(v_queries).to(self.device)

        # normalize the query
        v_queries = torch.nn.functional.normalize(v_queries, dim=-1)

        if queries_weights is not None:
            queries_weights = torch.tensor(queries_weights).to(self.device)

        current_index = 0

        results = []

        for batch in self.batches:
            score = fuesd_queries_distance(batch, v_queries, queries_weights).cpu()


            # if len(results) > 0:
            #     print(results[-1][-1], score.min())
                # print(results[-1][-1]<= score.min())
            if len(results) >= topk and results[-1][-1] >= score.max():
                # if the highest sim in the batch is smaller than the lowest sim
                # found
                current_index += len(batch)
                continue

            score = score.tolist()

            batch_ids = [i + current_index for i in range(len(batch))]

            current_index += len(batch)

            for idx, dist in zip(batch_ids, score):
                results.append((idx, dist))

            if len(results) > topk:
                results = heapq.nlargest(topk, results, key=lambda x: x[-1])
                results.sort(reverse=True, key=lambda x: x[-1])

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
        v_queries = self.encoder.encode_texts(texts, normalization=False)

        return self.vectors_search(v_queries, topk=topk)

    def search_by_images(self, images, topk=5):
        # batch search by images
        v_queries = self.encoder.encode_images(images, normalization=False)

        return self.vectors_search(v_queries, topk=topk)
