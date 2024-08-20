from pydantic import BaseModel, conlist


class Query(BaseModel):
    query: conlist(str, min_items=1, max_items=5) # type: ignore
    k: int = 10
    model: str
    language: str


class SearchResult(BaseModel):
    search_result: List[dict]


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
    mode: str="fast"
class Query_image(BaseModel):
    video_name: str
    idx: int
    model_name:str
    k: int=10