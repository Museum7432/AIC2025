from fastapi import APIRouter, HTTPException

from .models import SearchResult, MultiQuery
from searchers import Searchers, get_fused_searcher, get_faiss_searcher

router = APIRouter(prefix="/fuse_search")


# def get_searcher(model, Searchers=Searchers):
#     match model:
#         case "Clip-400M":
#             return Searchers["S400M_fused_searcher"]
#         case "ViT 5b":
#             return Searchers["clip_H_fused_searcher"]
#         case "ViT-bigG-2B":
#             return Searchers["clip_BigG_fused_searcher"]
#         case "vit-b32":
#             return Searchers["B32_fused_searcher"]
#         case _:
#             raise NotImplementedError()


@router.post("/", response_model=SearchResult)
def search_fuse(request: MultiQuery) -> SearchResult:
    queries = request.query
    topk = request.topk


    if len(queries) == 1:
        _searcher = get_faiss_searcher(request.model)

        _searcher_method = _searcher.batch_search_by_text

        results = _searcher_method(queries, topk)[0]

        return SearchResult(results=results)

    _searcher = get_fused_searcher(request.model)
    results = _searcher.search(queries, topk)

    return SearchResult(results=results)
