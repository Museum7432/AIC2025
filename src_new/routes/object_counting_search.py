from fastapi import APIRouter, HTTPException

from .models import Query, SearchResult, Query_OCR, Query_ObjectCount
from ..searchers import Searchers

router = APIRouter(prefix="/search_ObjectCount")


@app.post("/", response_model=SearchResult)
def search_ObjectCount(query_batch: Query_ObjectCount) -> SearchResult:
    query = query_batch.query
    k = query_batch.k
    mode = query_batch.mode
    if mode == "slow":
        results = Searchers["objcount_searcher"].search_slow(
            query[0], topk=k, measure_method="l2_norm"
        )
    elif mode == "fast":
        results = Searchers["objcount_searcher"].search_fast(query[0])

    return SearchResult(search_result=results)
