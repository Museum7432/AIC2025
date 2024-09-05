from fastapi import APIRouter, HTTPException

from .models import SearchResult, MultiQuery
from searchers import Searchers

router = APIRouter(prefix="/fuse_search")


def get_searcher(model, Searchers=Searchers):
    match model:
        case "Clip-400M":
            return Searchers["S400M_fused_searcher"]
        case "ViT 5b":
            return Searchers["clip_H_fused_searcher"]
        case "ViT-bigG-2B":
            return Searchers["clip_BigG_fused_searcher"]
        case "vit-b32":
            return Searchers["B32_fused_searcher"]
        case _:
            raise NotImplementedError()


@router.post("/", response_model=SearchResult)
def search_fuse(request: MultiQuery) -> SearchResult:
    queries = request.query
    topk = request.topk

    _searcher = get_searcher(request.model)

    results = _searcher.search(queries, topk)

    return SearchResult(results=results)
