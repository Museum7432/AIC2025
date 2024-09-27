from fastapi import APIRouter, HTTPException

from .models import SearchResult, TemporalQuery

from searchers import Searchers, get_temporal_searcher

from helpers import gpt4_translate_vi2en

router = APIRouter(prefix="/temporal_search")


@router.post("/", response_model=SearchResult)
def search_temporal(request: TemporalQuery) -> SearchResult:
    queries = request.query
    topk = request.topk

    language = request.language
    metric_type = request.metric

    if request.language == "Vie":
        for i, q in enumerate(queries):
            if not (q.startswith("+") or q.startswith("-") or len(q) == 0):
                queries[i] = gpt4_translate_vi2en(q)

    _searcher = get_temporal_searcher(request.model)

    results = _searcher.search(queries, topk, metric_type=metric_type)

    if request.language == "Vie":
        return SearchResult(results=results, translated_query=queries)
    
    return SearchResult(results=results)