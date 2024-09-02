import io
import base64

from typing import Union, List, Dict

import torch
import numpy as np

from PIL import Image

from database import EmbeddingsDB
from .fused_queries_searcher import FusedSearcher


def convert_frame_query(query, embs_db: EmbeddingsDB):
    # .e.g: ":f L01_V001 216"
    # video name: L01_V001
    # frame index: 216
    # convert frame query into embeddings
    assert query.startswith("+f ") or query.startswith("-f ")
    query = query[3:]

    vid_name, frame_idx = query.split()

    frame_idx = int(frame_idx)

    return embs_db.get_frame_embs(vid_name, frame_idx)


def convert_image_query(query):
    # convert image query into PIL image
    assert query.startswith("+i ") or query.startswith("-i ")
    query = query[3:]

    # the other part should be base64 encoded image

    pil_img = Image.open(io.BytesIO(base64.b64decode(query)))

    return pil_img


def get_query_type(query):
    if query.startswith("+f ") or query.startswith("-f "):
        return "frame"

    if query.startswith("+i ") or query.startswith("-i "):
        return "image"

    return "text"


class UnifiedSearcher:
    def __init__(self, searcher: FusedSearcher):
        # accepting texts, imgs, frames_id as input
        # performs search with the input searcher

        self.searcher = searcher

        self.encoder = searcher.encoder

        self.embs_db = searcher.db

    def get_queries_embs(self, queries: List[str]):
        # queries should be a list of string

        # place holder
        results = [None] * len(queries)

        # process the text queries
        texts_q = [(i, q) for i, q in enumerate(queries) if get_query_type(q) == "text"]

        texts_embs = self.encoder.encode_texts(
            [q[-1] for q in texts_q]
        )  # feature normalization should be done by the searcher

        for (i, q), q_embs in zip(texts_q, texts_embs):
            results[i] = q_embs

        # process the frame queries
        frames_q = [
            (i, q) for i, q in enumerate(queries) if get_query_type(q) == "frame"
        ]

        frames_embs = [convert_frame_query(q[-1], self.embs_db) for q in frames_q]

        for (i, q), f_embs in zip(frames_q, frames_embs):
            results[i] = f_embs

        # process the images queries

        imgs_q = [(i, q) for i, q in enumerate(queries) if get_query_type(q) == "image"]

        pil_imgs = [convert_image_query(q[-1]) for q in imgs_q]

        imgs_embs = self.encoder.encode_images(pil_imgs)

        for (i, q), i_embs in zip(imgs_q, imgs_embs):
            results[i] = i_embs

        # assert None not in results

        return np.array(results)

    def search(self, queries: List[str], topk=5, **kwargs):

        v_queries = self.get_queries_embs(queries)

        # image query should have a lower weight than text query
        queries_type = [get_query_type(q) for q in queries]

        queries_weights = [2 if t == "text" else 1 for t in queries_type]

        # query start with '-' will be treated as a negative query
        # .i.e: return images that are the farthest to these queries

        queries_weights = [
            -w if q.startswith("-") and t != "text" else w
            for q, w, t in zip(queries, queries_weights, queries_type)
        ]

        if "queries_weights" not in kwargs or kwargs["queries_weights"] is None:
            kwargs["queries_weights"] = queries_weights

        return self.searcher.vectors_search(
            v_queries, topk=topk, **kwargs
        )
