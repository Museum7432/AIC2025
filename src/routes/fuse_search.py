from fastapi import APIRouter, HTTPException

from .models import SearchResult, MultiQuery
from searchers import Searchers, get_fused_searcher, get_faiss_searcher
from helpers import gpt4_translate_vi2en, gpt4_split_query

router = APIRouter(prefix="/fuse_search")


@router.post("/", response_model=SearchResult)
def search_fuse(request: MultiQuery) -> SearchResult:
    queries = request.query
    topk = request.topk

    language = request.language
    metric_type = request.metric

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

    if len(queries) == 1 and not (queries[0].startswith("+") or queries[0].startswith("-")):
        _searcher = get_faiss_searcher(request.model)

        _searcher_method = _searcher.batch_search_by_text

        results = _searcher_method(queries, topk)[0]
    else:
        _searcher = get_fused_searcher(request.model)
        results = _searcher.search(queries, topk, metric_type=metric_type)
    
    return SearchResult(results=results, query=queries)
