from fastapi import APIRouter, HTTPException

from .models import Query, SearchResult, Query_ASR
from ..searchers import Searchers

router = APIRouter(prefix="/search_ASR")


@router.post("/", response_model=SearchResult)
def search_ASR(query_requets: Query_ASR) -> SearchResult:
    query = query_requets.query
    k = query_requets.k
    mode = query_requets.mode
    if mode == "slow":
        results = Searchers["asr_searcher"].search_slow(text=query, num_img=k)
    elif mode == "fast":
        results = Searchers["asr_searcher"].search_fast(text=query, num_img=k)
    return SearchResult(search_result=results)
