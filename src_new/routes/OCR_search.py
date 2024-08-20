from fastapi import APIRouter, HTTPException

from .models import Query, SearchResult, Query_OCR
from ..searchers import Searchers

router = APIRouter(prefix="/search_OCR")


@router.post("/", response_model=SearchResult)
def search_OCR(query_batch: Query_OCR) -> SearchResult:
    query = query_batch.query
    k = query_batch.k
    # results=search_compare_similirity_word_load_fulldatabase_to_ram(query[0],database=dbOCR,num_img=k)#simple thread
    results = Searchers["ocr_searcher"].search(query=query[0], num_img=k)
    return SearchResult(search_result=results)
