from pydantic import BaseModel, conlist
from typing import List, Tuple, Dict, Literal

class SearchResult(BaseModel):
    results: List[dict]


class SingleTextQuery(BaseModel):
    query: str
    topk: int = 10
    model: str = "vit-b32"
    language: str

class MultiQuery(BaseModel):
    query: List[str]
    topk: int = 10
    model: str = "vit-b32"

# class Query_OCR(BaseModel):
#     query: conlist(item_type=str, min_length=1, max_length=5)  # type: ignore
#     topk: int = 10


# class Query_ObjectCount(BaseModel):
#     query: conlist(item_type=str, min_length=1, max_length=5)  # type: ignore
#     topk: int = 10
#     mode: str = "slow"


# class Query_image(BaseModel):
#     video_name: str
#     idx: int
#     model_name: str
#     topk: int = 10


# for ASR
class AsrQuery(BaseModel):
    query: str
    topk: int = 10
    mode: str = "fast"


# for OCR
class OcrQuery(BaseModel):
    query: str
    topk: int = 10


# for object counting
class ObjectCountingQuery(BaseModel):
    query: str
    topk: int = 10
    mode: str = "slow"


class FrameNeighborsQuery(BaseModel):
    video_name: str
    frame_idx: int
    model: str = "vit-b32"
    topk: int = 10
