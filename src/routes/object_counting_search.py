from fastapi import APIRouter, HTTPException

from .models import SearchResult, ObjectCountingQuery
from searchers import Searchers

router = APIRouter(prefix="/search_ObjectCount")


@router.post("/", response_model=SearchResult)
def search_ObjectCount(request: ObjectCountingQuery) -> SearchResult:
    query = request.query
    topk = request.topk
    mode = request.mode

    if mode == "slow":
        results = Searchers["objcount_searcher"].search_slow(
            query, topk=topk, measure_method="l2_norm"
        )
    elif mode == "fast":
        results = Searchers["objcount_searcher"].search_fast(query, topk=topk)
    else:
        results = Searchers["objcount_searcher"].elastic_search(query, topk=topk)

    return SearchResult(results=results)
