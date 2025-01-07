from fastapi import APIRouter, HTTPException

from .models import SearchResult, MultiQuery
from searchers import Searchers, get_ft_searcher
from helpers import gpt4_translate_vi2en, gpt4_split_query

router = APIRouter(prefix="/fuse_search")


@router.post("/", response_model=SearchResult)
def search_fuse(request: MultiQuery) -> SearchResult:
    queries = request.query
    topk = request.topk

    language = request.language


    if request.gpt_split:
        assert len(queries) == 1

        mode = "static"
        if request.language == "Vie":
            mode = "static_vn"
        
        queries = gpt4_split_query(queries[0], mode=mode)

    elif request.language == "Vie":
        for i, q in enumerate(queries):
            queries[i] = gpt4_translate_vi2en(q)
    

    _searcher = get_ft_searcher(request.model)
    
    results = _searcher.fuse_search_by_texts(queries, topk)
    
    return SearchResult(results=results, query=queries)
