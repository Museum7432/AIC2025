from fastapi import APIRouter, HTTPException

from .models import SearchResult, TemporalQuery

from searchers import Searchers, get_temporal_searcher

router = APIRouter(prefix="/temporal_search")


# def get_searcher(model, Searchers=Searchers):
#     match model:
#         case "Clip-400M":
#             return Searchers["S400M_temporal_searcher"]
#         case "ViT 5b":
#             return Searchers["clip_H_temporal_searcher"]
#         case "ViT-bigG-2B":
#             return Searchers["clip_BigG_temporal_searcher"]
#         case "vit-b32":
#             return Searchers["B32_temporal_searcher"]
#         case _:
#             raise NotImplementedError()


@router.post("/", response_model=SearchResult)
def search_temporal(request: TemporalQuery) -> SearchResult:
    queries = request.query
    topk = request.topk

    metric_type = request.metric

    _searcher = get_temporal_searcher(request.model)

    results = _searcher.search(queries, topk, metric_type=metric_type)

    return SearchResult(results=results)
