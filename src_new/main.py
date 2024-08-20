import io
import base64
from typing import List, Tuple, Dict
from PIL import Image
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, conlist

from encoders import ClipEncoder
from searchers import AsrSearcher, FaissSearcher, FusedSearcher
from database import EmbeddingsDB, ObjectCountDB



Vit_H_db = EmbeddingsDB("./embeddings/ViT-H-14-378-quickgelu-dfn5b", build_faiss_index=True)

Vit_H_enc = ClipEncoder(model_arch="ViT-B-32", pretrained="openai")

Vit_H_faiss_searcher = FaissSearcher(Vit_H_db, Vit_H_enc)
Vit_H_fused_searcher = FusedSearcher(Vit_H_db, Vit_H_enc)


print(Vit_H_faiss_searcher.batch_search_by_text(["a picture of a dog"]))
