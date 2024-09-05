from pydantic import BaseModel, conlist
from typing import List, Tuple, Dict, Literal
from typing import Union


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


class TemporalQuery(BaseModel):
    query: List[str]
    topk: int = 10
    model: str = "vit-b32"

    # not required
    queries_weights: Union[List[float], None] = None
    match_first: bool = True
    return_match_ids: bool = True


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
    mode: str = "elastic"


# for OCR
class OcrQuery(BaseModel):
    query: str
    topk: int = 10
    mode: str = "elastic"


# for object counting
class ObjectCountingQuery(BaseModel):
    query: str
    topk: int = 10
    mode: str = "elastic"


class FrameNeighborsQuery(BaseModel):
    video_name: str
    frame_idx: int
    model: str = "vit-b32"
    topk: int = 10
