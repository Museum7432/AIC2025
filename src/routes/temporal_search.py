from fastapi import APIRouter, HTTPException

from .models import SearchResult, TemporalQuery

from searchers import Searchers, get_temporal_searcher

from helpers import gpt4_translate_vi2en, gpt4_split_query


router = APIRouter(prefix="/temporal_search")


@router.post("/", response_model=SearchResult)
def search_temporal(request: TemporalQuery) -> SearchResult:
    queries = request.query
    topk = request.topk

    language = request.language
    metric_type = request.metric

    max_frame_dist = request.max_frame_dist
    min_frame_dist = request.min_frame_dist


    if request.gpt_split:
        assert len(queries) == 1
        mode = "static"
        if request.language == "Vie":
            mode = "static_vn"
        
        queries = gpt4_split_query(queries[0], mode=mode)

    elif request.language == "Vie":
        for i, q in enumerate(queries):
            if not (q.startswith("+") or q.startswith("-") or len(q) == 0):
                queries[i] = gpt4_translate_vi2en(q)

    _searcher = get_temporal_searcher(request.model)

    results = _searcher.search(
        queries,
        topk,
        metric_type=metric_type,
        max_frame_dist=max_frame_dist,
        min_frame_dist=min_frame_dist,
    )

    return SearchResult(results=results, query=queries)