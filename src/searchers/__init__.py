from contextlib import asynccontextmanager
from fastapi import FastAPI

from .faiss_searcher import FaissSearcher
from .fused_queries_searcher import FusedSearcher
from .ASR_searcher import AsrSearcher
from .ObjectCountSearcher import ObjectCountSearcher
from .OCR_searcher import OcrSearcher
from .unified_searcher import UnifiedSearcher

from .temporal_queries_searcher import TemporalSearcher

from .obj_loc_color_searcher import ObjectLocationSearcher

from database import *
from encoders import ClipEncoder

from .ft_searcher import FTSearcher

from config import settings

Searchers = {}


def load_model(
    prefix_name="B32",
    embs_path="data/keyframes_embs_clip_B32",
    clip_model=True,
    model_arch="ViT-B-32",
    pretrain_name="openai",
    device="cpu",
    batch_size=2048,
    jit=True,
):
    # load the embeddings
    embs_db = EmbeddingsDB(embs_path)
    # load the model
    if clip_model:
        encoder = ClipEncoder(model_arch, pretrain_name, device=device, jit=jit)
    else:
        raise RuntimeError("Blip model have been removed")
    # create the searcher
    faiss_searcher = FaissSearcher(embs_db, encoder)

    fused_searcher = UnifiedSearcher(FusedSearcher(embs_db, encoder, batch_size=2048))
    temporal_searcher = UnifiedSearcher(TemporalSearcher(embs_db, encoder))

    print(f"{prefix_name} loaded!")
    return {
        f"{prefix_name}_faiss": faiss_searcher,
        f"{prefix_name}_fused": fused_searcher,
        f"{prefix_name}_temporal": temporal_searcher,
    }


def load_model_FT(
    prefix_name="B32",
    embs_path="data/keyframes_embs_clip_B32",
    clip_model=True,
    model_arch="ViT-B-32",
    pretrain_name="openai",
    device="cpu",
    jit=True,
):
    # load the embeddings
    embs_db = FTdb(embs_path)
    # load the model
    if clip_model:
        encoder = ClipEncoder(model_arch, pretrain_name, device=device, jit=jit)
    else:
        raise RuntimeError("Blip model have been removed")

    # create the searcher
    ft_searcher = FTSearcher(embs_db, encoder)

    print(f"{prefix_name} loaded!")
    return {
        f"{prefix_name}_ft": ft_searcher,
    }


def model_name_map(name):
    match name:
        case "Clip-400M":
            return "S400M"
        case "ViT 5b":
            return "clip_H"
        case "ViT-bigG-2B":
            return "clip_BigG"
        case "vit-b32":
            return "B32"
        case "vit-Med":
            return "clip_Med"
        case _:
            raise NotImplementedError()


def get_faiss_searcher(name):
    name = model_name_map(name)

    return Searchers[f"{name}_faiss"]


def get_fused_searcher(name):
    name = model_name_map(name)
    return Searchers[f"{name}_fused"]


def get_temporal_searcher(name):
    name = model_name_map(name)
    return Searchers[f"{name}_temporal"]


def get_ft_searcher(name):
    name = model_name_map(name)

    return Searchers[f"{name}_ft"]


def load_seacher():
    re = {}

    print("preparing searchers!")

    if settings.clip_B32_embs_path:
        re.update(
            load_model_FT(
                prefix_name="B32",
                embs_path=settings.clip_B32_embs_path,
                clip_model=True,
                model_arch="ViT-B-32",
                pretrain_name="openai",
                device=settings.device,
                jit=True,
            )
        )

    if settings.ocr_path:
        # ocr database
        ocr_db = OcrDB(settings.ocr_path, remove_old_index=settings.remove_old_index)
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
        asr_db = AsrDB(settings.asr_path, remove_old_index=settings.remove_old_index)
        asr_searcher = AsrSearcher(asr_db)

        re["asr_searcher"] = asr_searcher
        print("ASR loaded!")

    if settings.color_code_path:
        objloc_db = ObjectLocationDB(settings.obj_loc_path, settings.color_code_path)

        objloc_searcher = ObjectLocationSearcher(objloc_db)

        re["obj_loc_searcher"] = objloc_searcher

        print("obj location loaded!")

    if settings.blip2_embs_path:
        re.update(
            load_model_FT(
                prefix_name="blip2",
                embs_path=settings.blip2_embs_path,
                clip_model=False,
                model_arch="blip2_feature_extractor",
                pretrain_name="pretrain",
                device=settings.device,
            )
        )

    if settings.clip_S400M_embs_path:
        re.update(
            load_model_FT(
                prefix_name="S400M",
                embs_path=settings.clip_S400M_embs_path,
                clip_model=True,
                model_arch="ViT-SO400M-14-SigLIP-384",
                pretrain_name="webli",
                device=settings.device,
                jit=False,
            )
        )

    if settings.clip_H_embs_path:
        re.update(
            load_model_FT(
                prefix_name="clip_H",
                embs_path=settings.clip_H_embs_path,
                clip_model=True,
                model_arch="ViT-H-14-378-quickgelu",
                pretrain_name="dfn5b",
                device=settings.device,
                jit=True,
            )
        )

    if settings.clip_bigG_embs_path:
        re.update(
            load_model_FT(
                prefix_name="clip_BigG",
                embs_path=settings.clip_bigG_embs_path,
                clip_model=True,
                model_arch="ViT-bigG-14",
                pretrain_name="laion2B-s39B-b160k",
                device=settings.device,
                jit=True,
            )
        )
    
    if settings.clip_Med_embs_path:
        re.update(
            load_model(
                prefix_name="clip_Med",
                embs_path=settings.clip_Med_embs_path,
                clip_model=True,
                model_arch="hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO",
                pretrain_name="",
                device=settings.device,
                batch_size=2048,
                jit=True,
            )
        )

    return re


# will be loaded on app startup
@asynccontextmanager
async def SearchersLifespan(app: FastAPI):
    Searchers.update(load_seacher())

    yield

    # on app shutdown
    Searchers.clear()
