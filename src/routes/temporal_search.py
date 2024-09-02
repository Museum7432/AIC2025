from fastapi import APIRouter, HTTPException

from .models import SearchResult, TemporalQuery
from searchers import Searchers

router = APIRouter(prefix="/temporal_search")


def get_searcher(model, Searchers=Searchers):
    match model:
        case "Blip2-ViTG":
            return Searchers["blip2_temporal_searcher"]
        case "ViT 5b":
            return Searchers["clip_H_temporal_searcher"]
        case "ViT-bigG-14":
            return Searchers["clip_BigG_temporal_searcher"]
        case "vit-b32":
            return Searchers["B32_temporal_searcher"]
        case _:
            raise NotImplementedError()


@router.post("/", response_model=SearchResult)
def search_temporal(request: TemporalQuery) -> SearchResult:
    queries = request.query
    topk = request.topk

    match_first = request.match_first
    return_match_ids = request.return_match_ids

    _searcher = get_searcher(request.model)

    results = _searcher.search(queries, topk, match_first=match_first, return_match_ids=return_match_ids)

    return SearchResult(results=results)
