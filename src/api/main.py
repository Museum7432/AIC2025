from __future__ import annotations
import io
import base64
from typing import List, Tuple
import faiss
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from src.searchers.BLIPSearcher import searchForBLIP
import os
import numpy as np
from tqdm import tqdm
from  src.searchers.SemanticSearcher import searchForOpenClip
from src.utils.utils import read_config
config = read_config()
class Query(BaseModel):
    query: conlist(str, min_items=1, max_items=5) # type: ignore
    k: int = 10
    model: str

class SearchResult(BaseModel):
    search_result: List[dict]


# add database into faiss indexing
def faiss_indexing(db: list, feature_dimension: int) -> faiss.IndexFlatL2:
  index = faiss.IndexFlatL2(feature_dimension)

  for idx , instance in enumerate(db):

        video_name, idx, feat_vec= instance
        
        feat_vec /= np.linalg.norm(feat_vec)
        feat_vec = np.float32(feat_vec)

       

        index.add(feat_vec.reshape(-1, 1).T)
  return index


# Load database from embeddings
def Database(PATH_TO_CLIPFEATURES: str) -> List[Tuple[str, int, np.ndarray]]:
    data_base = []
    for name_file_feature in tqdm(sorted(os.listdir(PATH_TO_CLIPFEATURES))):
        if name_file_feature.endswith('.npy'):
            vid_name = name_file_feature.split('.')[0]
            features = np.load(os.path.join(PATH_TO_CLIPFEATURES, name_file_feature))
            for idx, feat in enumerate(features, 1):
                instance = (vid_name, idx, feat)
                data_base.append(instance)
    return data_base



app = FastAPI(title="ELO@AIC Image Semantic Search")

@app.on_event("startup")
def load_searcher() -> None:
    EMB_PATH = config['EMB_PATH']
    db32 = Database(EMB_PATH['BLIP2_VITG'])
    db14 = Database(EMB_PATH['BLIP2_COCO'])
    # db14_336 = Database(EMB_CFG['VIT_BIGG'])

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

@app.post("/search", response_model= SearchResult)
def search(query_batch: Query) -> SearchResult:
    query = query_batch.query
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
    elif model == "Blip2-ViTG":
        result = searcher32(query, k)
  
    return SearchResult(search_result = result)


if __name__=="__main__":
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)