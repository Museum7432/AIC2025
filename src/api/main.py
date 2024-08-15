from __future__ import annotations
import sys
sys.path.append("")
import io
import base64
from typing import List, Tuple, Dict# mới thêm OCR
from src.searchers.OCRSearcher import search_compare_similirity_word_load_fulldatabase_to_ram,search_in_db_v2
import faiss
import pandas as pd
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from api import SemanticSearcher
from src.searchers.BLIPSearcher import searchForBLIP
import os
import glob
import numpy as np
from tqdm import tqdm
import json
from fastapi.middleware.cors import CORSMiddleware# mới thêm 28_7_24
from src.searchers.ObjectCountSearcher import search_obj_count_engine_slow,search_obj_count_engine_fast# mới thêm 11/8/2024
from src.searchers.ASRSearcher import ASR_search_engine # 13/8/2024 ASR
from src.load_database.load_database import faiss_indexing,Database,load_databaseASR,load_databaseObjectCount_Fast,load_databaseObjectCount_Slow,load_databaseOCR


#GPTapi
from openai import OpenAI
api_key = ""
client = OpenAI(api_key = api_key)

# from  api.SemanticSearcher import searchForOpenClip
class Query(BaseModel):
    query: conlist(str, min_items=1, max_items=5) # type: ignore
    k: int = 10
    model: str
    language: str


class SearchResult(BaseModel):
    search_result: List[dict]

#mới thêm OCR
class Query_OCR(BaseModel):
    query: conlist(item_type=str, min_items=1, max_items=5) # type: ignore
    k: int =10

class Query_ObjectCount(BaseModel):
    query: conlist(item_type=str, min_items=1, max_items=5) # type: ignore
    k: int= 10
    mode: str ="slow"
class Query_ASR(BaseModel):
    query: str
    k: int=10


app = FastAPI(title="ELO@AIC Image Semantic Search")
#mới thêm 28_7_24
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn gốc. Bạn có thể giới hạn lại theo nhu cầu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_searcher() -> None:
    
    global dbOCR#biến toàn cục phải khai báo trước khi dùng
    dbOCR = load_databaseOCR("./texts_extracted/")

    global dbObjectCount_slow
    global dbObjectCount_fast
    dbObjectCount_slow = load_databaseObjectCount_Slow("./Object_Counting_vector_np/")
    dbObjectCount_fast = load_databaseObjectCount_Fast("./Object_Counting_vector_np/")

    global dbASR
    dbASR =  load_databaseASR("./ASR_folder/")

    db32 = Database("./embeddings/blip2_feature_extractor-ViTG/")
    db14 = Database("./embeddings/blip2_image_text_matching-coco/")
    # db14_336 = Database("./embeddings/ViT-bigG-14-CLIPA-336-datacomp1b/")

    #load features into databases
    index_32 = faiss_indexing(db32, 768)
    index_14 = faiss_indexing(db14, 768)
    # index_14_336 = faiss_indexing(db14_336 , 1280)

    global searcher32
    global searcher14
    # global searcher14336
    # global searcher14g_La
    # global searcher14G_La

    searcher32 = searchForBLIP("blip2_feature_extractor", "pretrain", index_32, db32)
    searcher14= searchForBLIP("blip2_image_text_matching", "coco", index_14, db14)
    # searcher14336= searchForOpenClip("ViT-bigG-14-CLIPA-336", "datacomp1b", index_14_336, db14_336)

@app.get("/")
def home() -> None:
    return "Welcome to the Image Semantic Search API. Head over http://localhost:8000/docs for more info."


@app.post("/search", response_model=SearchResult)
def search(query_batch: Query) -> SearchResult:
    query = query_batch.query
    #translate Vi2En GPTapi
    if query_batch.language == "Vietnamese":
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant in translating documents from Vietnamese to English, do not translate name, street, organization name, location, character between 2 quotes symbol."},
                {"role": "user",
                 "content": query[0]}
            ]
        )
        query=[response.choices[0].message.content]

    k = query_batch.k
    model = query_batch.model
    if not isinstance(query, list):
        HTTPException(status_code=400, detail="Query must be a list")
    elif query[0].startswith("data:image/"):
        query = [
            Image.open(
                io.BytesIO(base64.b64decode(item.split(",")[1]))
            )
            for item in query
        ]
    elif not isinstance(query[0], str):
        HTTPException(status_code=400, detail="Query must be a list of strings or base64 encoded images")
    if model == "Blip2-Coco":
        result = searcher14(query, k)
    elif model == "Blip2-Coco":
        result = searcher14(query, k)
    elif model == "Blip2-ViTG":
        result = searcher32(query, k)

    return SearchResult(search_result=result)
#mới thêm OCR
@app.post("/search_OCR",response_model=SearchResult)
def search_OCR(query_batch: Query_OCR)-> SearchResult:
    query=query_batch.query
    k=query_batch.k
    # results=search_compare_similirity_word_load_fulldatabase_to_ram(query[0],database=dbOCR,num_img=k)#simple thread
    results=search_in_db_v2(query=query[0],database=dbOCR,num_img=k)#multi thread
    return  SearchResult(search_result=results)
@app.post("/search_ObjectCount", response_model=SearchResult)
def search_ObjectCount(query_batch: Query_ObjectCount) -> SearchResult:
    query=query_batch.query
    k=query_batch.k
    mode=query_batch.mode
    if mode=="slow":
        results=search_obj_count_engine_slow(query[0],db=dbObjectCount_slow,topk=k,measure_method="l2_norm")
    elif mode=="fast":
        results=search_obj_count_engine_fast(query[0],db=dbObjectCount_fast)
    return SearchResult(search_result=results)
@app.post("/search_ASR",response_model=SearchResult)
def search_ASR(query_requets: Query_ASR ) -> SearchResult:
    query=query_requets.query
    k=query_requets.k
    results=ASR_search_engine(query=query,database=dbASR,k=k)
    return SearchResult(search_result=results)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)