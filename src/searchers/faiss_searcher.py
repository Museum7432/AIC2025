from typing import Union, List, Dict

import torch
import faiss

import numpy as np

from PIL import Image

from database import EmbeddingsDB
from encoders import ClipEncoder, BlipEncoder

from utils import normalized_np


class FaissSearcher:
    def __init__(self, embs_db: EmbeddingsDB, encoder: Union[ClipEncoder, BlipEncoder]):
        self.db = embs_db
        self.encoder = encoder

        # build the faiss indexer

        # IndexFlatL2 is a brute-force indexer (.i.e: is no different than linear search)
        # we might want to use cosine similarity instead of L2 (IndexFlatIP)
        self.faiss_index = faiss.IndexFlatIP(embs_db.embs_dim)

        self.faiss_index.add(embs_db.fused_embs)

    def batch_vector_search(self, v_queries, topk=5):
        # perform search by vector (independently)
        # v_queries should be a numpy array

        # normalize the query
        v_queries = normalized_np(v_queries)

        D, I = self.faiss_index.search(v_queries, topk)

        # D: distance (#queries, topk)
        # I: indices (#queries, topk)

        batch_results = []
        for query_results, distances in zip(I, D):
            # for earch query
            q_re = []

            # topk results of each query
            for re, distance in zip(query_results, distances):

                vid_name, frame_idx = self.db.get_info(re)

                q_re.append(
                    {
                        "score": distance.item(),
                        "keyframe_id": frame_idx.item(),
                        "video_name": vid_name,
                    }
                )

            batch_results.append(q_re)

        return batch_results

    def batch_search_by_text(self, texts, topk=5):
        # batch search by text
        v_queries = self.encoder.encode_texts(texts)

        return self.batch_vector_search(v_queries, topk=topk)

    def batch_search_by_image(self, images, topk=5):
        # batch search by images
        # images should be a list of PIL.Image

        v_queries = self.encoder.encode_images(images)

        return self.batch_vector_search(v_queries, topk=topk)

    def search_by_indexed_image(self, video_name, frame_idx, topk=5):
        image_embs = self.db.get_frame_embs(video_name, frame_idx)
        return self.batch_vector_search(image_embs[None, :], topk=topk)[0]
