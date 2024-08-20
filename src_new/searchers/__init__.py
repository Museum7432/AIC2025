from .faiss_searcher import FaissSearcher
from .fused_queries_searcher import FusedSearcher
from .ASR_searcher import AsrSearcher
from .ObjectCountSearcher import ObjectCountSearcher
from .OCR_searcher import OcrSearcher

from database import *
from encoders import ClipEncoder, BlipEncoder

from contextlib import asynccontextmanager
from fastapi import FastAPI

Searchers = {}


def load_seacher():
    # load the embeddings
    B32_clip_db = EmbeddingsDB("data/clip-features-vit-b32", build_faiss_index=True)

    # load the model
    B32_encoder = ClipEncoder("ViT-B-32", "openai")

    # create the searcher
    B32_searcher = FaissSearcher(B32_clip_db, B32_encoder)

    return {"B32_searcher": B32_searcher}


@asynccontextmanager
async def SearchersLifespan(app: FastAPI):
    Searchers.update(load_seacher())

    yield

    Searchers.clear()

