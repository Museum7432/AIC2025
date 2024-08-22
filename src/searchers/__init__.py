from contextlib import asynccontextmanager
from fastapi import FastAPI

from .faiss_searcher import FaissSearcher
from .fused_queries_searcher import FusedSearcher
from .ASR_searcher import AsrSearcher
from .ObjectCountSearcher import ObjectCountSearcher
from .OCR_searcher import OcrSearcher
from .unified_searcher import UnifiedSearcher

from database import *
from encoders import ClipEncoder, BlipEncoder

from config import settings

Searchers = {}


def load_seacher():

    re = {}

    if settings.clip_B32_embs_path:
        # load the embeddings
        B32_clip_db = EmbeddingsDB(settings.clip_B32_embs_path, build_faiss_index=True)

        # load the model
        B32_encoder = ClipEncoder("ViT-B-32", "openai", device="cpu")

        # create the searcher
        B32_searcher = FaissSearcher(B32_clip_db, B32_encoder)

        re["B32_searcher"] = B32_searcher
        print("clip B32 loaded!")

    if settings.ocr_path:
        # ocr database
        ocr_db = OcrDB(settings.ocr_path)
        ocr_searcher = OcrSearcher(ocr_db)

        re["ocr_searcher"] = ocr_searcher
        print("OCR loaded!")

    if settings.object_counting_path:
        # object counting database
        obj_db = ObjectCountDB(settings.object_counting_path)
        objcount_searcher = ObjectCountSearcher(obj_db)

        re["objcount_searcher"] = objcount_searcher
        print("object counting loaded!")

    if settings.asr_path:
        asr_db = AsrDB(settings.asr_path)
        asr_searcher = AsrSearcher(asr_db)

        re["asr_searcher"] = asr_searcher
        print("ASR loaded!")

    if settings.blip2_embs_path:
        blip2_db = EmbeddingsDB(settings.blip2_embs_path, build_faiss_index=True)

        blip2_encoder = BlipEncoder("blip2_feature_extractor", "pretrain")

        blip2_searcher = FaissSearcher(blip2_db, blip2_encoder)

        re["blip2_searcher"] = blip2_searcher

        print("BLIP2 loaded!")

    if settings.clip_H_embs_path:
        clip_H_db = EmbeddingsDB(settings.clip_H_embs_path, build_faiss_index=True)

        clip_H_encoder = ClipEncoder("ViT-H-14-378-quickgelu", "dfn5b")

        clip_H_searcher = FaissSearcher(clip_H_db, clip_H_encoder)

        re["clip_H_searcher"] = clip_H_searcher

        print(" ViT-H-14-378-quickgelu!")

    if settings.clip_bigG_embs_path:
        clip_BigG_db = EmbeddingsDB(
            settings.clip_bigG_embs_path, build_faiss_index=True
        )

        clip_BigG_encoder = ClipEncoder("ViT-bigG-14-CLIPA-336", "datacomp1b")

        clip_BigG_searcher = FaissSearcher(clip_BigG_db, clip_BigG_encoder)

        re["clip_BigG_searcher"] = clip_BigG_searcher

        print("ViT-bigG-14-CLIPA-336")

    return re


def model_name_to_searcher(name):
    match name:
        case "Blip2-ViTG":
            return Searchers["blip2_searcher"]
        case "ViT 5b":
            return Searchers["clip_H_searcher"]
        case "ViT-bigG-14":
            return Searchers["clip_BigG_searcher"]
        case "vit-b32":
            return Searchers["B32_searcher"]
        case _:
            raise NotImplementedError()


# will be loaded on app startup
@asynccontextmanager
async def SearchersLifespan(app: FastAPI):
    Searchers.update(load_seacher())

    yield

    # on app shutdown
    Searchers.clear()
