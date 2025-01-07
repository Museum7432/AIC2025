from typing import Union, List, Dict

import torch
import faiss

import numpy as np

from PIL import Image
import heapq

from database import FTdb
from encoders import ClipEncoder
import time


class FTSearcher:
    def __init__(self, embs_db: FTdb, encoder: ClipEncoder):
        self.db = embs_db
        self.encoder = encoder

    def vectors_search(self, v_queries, topk=5):
        # perform linear search
        # return topk instances with the minimum total
        # distance to all queries

        start_time = time.time()

        sims, ids = self.db.db.search(v_queries, topk)

        end_time = time.time()
        print(f"vectors_search Runtime: {end_time - start_time} seconds")

        sims = sims.tolist()
        ids = ids.tolist()

        batch_results = []

        for res, ss in zip(ids, sims):
            # for earch query
            query_results = []

            # topk results of each query
            for re, s in zip(res, ss):

                info = self.db.get_info(re)

                frame_idx = re - info["start_idx"]
                vid_name = info["seq_name"]

                query_results.append(
                    {
                        "score": s,
                        "keyframe_id": frame_idx,
                        "video_name": vid_name,
                    }
                )

            batch_results.append(query_results)

        return batch_results

    def vectors_seq_search(self, v_queries, topk=5, min_item_dist=1, discount_rate=1):

        start_time = time.time()
        sims, ids = self.db.db.seq_search(
            v_queries, topk, min_item_dist=min_item_dist, discount_rate=discount_rate
        )

        end_time = time.time()
        print(f"vectors_seq_search Runtime: {end_time - start_time} seconds")

        sims = sims.tolist()
        ids = ids.tolist()

        query_results = []
        for s, matched_frames in zip(sims, ids):

            info = self.db.get_info(matched_frames[0])

            vid_name = info["seq_name"]

            query_results.append(
                {
                    "score": s,
                    "video_name": vid_name,
                    "matched_frames": [i - info["start_idx"] for i in matched_frames],
                }
            )

        return query_results

    def search_by_texts(self, texts, topk=5):
        # batch search by text
        v_queries = self.encoder.encode_texts(texts)

        return self.vectors_search(v_queries, topk=topk)

    def fuse_search_by_texts(self, texts, topk=5):
        # TODO: move this into the FTsearch library
        # batch search by text
        v_queries = self.encoder.encode_texts(texts)

        v_queries = v_queries.mean(0)[None, :]

        return self.vectors_search(v_queries, topk=topk)[0]

    def search_by_images(self, images, topk=5, min_item_dist=1, discount_rate=1):
        # batch search by images
        v_queries = self.encoder.encode_images(images)

        return self.vectors_search(
            v_queries,
            topk=topk,
            min_item_dist=min_item_dist,
            discount_rate=discount_rate,
        )

    def temporal_search_by_texts(self, texts, topk=5, min_item_dist=1, discount_rate=1):
        v_queries = self.encoder.encode_texts(texts)

        return self.vectors_seq_search(
            v_queries,
            topk=topk,
            min_item_dist=min_item_dist,
            discount_rate=discount_rate,
        )

    def search_by_indexed_image(self, vid_name, frame_idx, topk=5):
        # v_queries = self.db.db.get_vec(vec_idx)[None, :]
        v_queries = self.db.get_frame_embs(vid_name, frame_idx)[None, :]

        return self.vectors_search(v_queries, topk=topk)[0]
