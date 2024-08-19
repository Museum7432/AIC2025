from typing import Union, List, Dict

import torch
import faiss

import numpy as np

from PIL import Image


class FaissSearcher:
    def __init__(self, embs_db, encoder):
        self.db = embs_db
        self.encoder = encoder

        assert (
            embs_db.faiss_index is not None
        ), "the embedding database should have faiss enabled"
        self.faiss_index = embs_db.faiss_index

    def batch_vector_search(self, v_queries, topk=5):
        # perform search by vector (independently)
        # v_queries should be a numpy array

        D, I = self.faiss_index.search(v_queries, topk)

        # D: distance (#queries, topk)
        # I: indices (#queries, topk)

        batch_results = []
        for query_results, distances in zip(I, D):
            # for earch query
            query_results = []

            # topk results of each query
            for re, distance in zip(query_results, distances):

                vid_name, frame_idx = self.db.get_info(re)

                query_results.append(
                    {
                        "score": distance,
                        "keyframe_id": frame_idx,
                        "video_name": vid_name,
                    }
                )

            batch_results.append(query_results)

        return search_result

    def search_by_texts(self, texts, topk=5):
        # batch search by text
        v_queries = self.encoder.encode_texts(texts, normalization=True)

        return self.batch_vector_search(v_queries, topk=topk)


    def search_by_images(self, images, topk=5):
        # batch search by images
        # images should be a list of PIL.Image
        
        v_queries = self.encoder.encode_images(images, normalization=True)

        return self.batch_vector_search(v_queries, topk=topk)
    