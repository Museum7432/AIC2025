from pydantic import BaseModel, conlist
from typing import List, Tuple, Dict, Literal
from typing import Union


class SearchResult(BaseModel):
    results: List[dict]
    query: Union[List[str], None] = None


class SingleTextQuery(BaseModel):
    query: str
    topk: int = 10
    model: str = "vit-b32"
    language: str


class MultiQuery(BaseModel):
    query: List[str]
    topk: int = 10
    model: str = "vit-b32"
    metric: str = "exp_dot"  # 'dot' or 'exp_dot'
    language: str = "en"

    # if True then the first query will be splitted
    gpt_split: bool = False


class TemporalQuery(BaseModel):
    query: List[str]
    topk: int = 10
    model: str = "vit-b32"
    language: str = "en"

    metric: str = "exp_dot"  # 'dot' or 'exp_dot'
    # not required
    queries_weights: Union[List[float], None] = None

    # avoid using max_frame_dist and min_frame_dist as the kernel
    # version of temporal_matching is 4 times slower
    max_frame_dist: int = -1 # -1: disable
    min_frame_dist: int = 1 # 1: disable

    gpt_split: bool = False


class TranslationQuery(BaseModel):
    texts: List[str]
    # only vi2en is available


class TranslationResult(BaseModel):
    texts: List[str]


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


class ObjectLocationQuery(BaseModel):
    class_ids: List[int]
    box_cords: List[List[float]]
    topk: int = 10
