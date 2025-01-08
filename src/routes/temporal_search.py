from fastapi import APIRouter, HTTPException

from .models import SearchResult, TemporalQuery

from searchers import Searchers, get_ft_searcher

from helpers import gpt4_translate_uni2en, gpt4_split_query


router = APIRouter(prefix="/temporal_search")


@router.post("/", response_model=SearchResult)
def search_temporal(request: TemporalQuery) -> SearchResult:
    queries = request.query
    topk = request.topk

    language = request.language

    discount_rate = request.discount_rate
    min_frame_dist = request.min_frame_dist

    if request.gpt_split:
        assert len(queries) == 1
        mode = "static"
        if request.language != "en":
            mode = "static_uni"

        queries = gpt4_split_query(queries[0], mode=mode)

    elif request.language != "en":
        for i, q in enumerate(queries):
            queries[i] = gpt4_translate_uni2en(q)

    _searcher = get_ft_searcher(request.model)

    results = _searcher.temporal_search_by_texts(
        queries,
        topk,
        min_item_dist=min_frame_dist,
        discount_rate=discount_rate,
    )

    return SearchResult(results=results, query=queries)
