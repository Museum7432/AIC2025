from fastapi import APIRouter, HTTPException

from .models import AsrQuery, SearchResult
from searchers import Searchers

router = APIRouter(prefix="/search_ASR")


@router.post("/", response_model=SearchResult)
def search_ASR(query_requets: AsrQuery) -> SearchResult:

    if "asr_searcher" not in Searchers.keys():
        raise HTTPException(
            status_code=500, detail="ASR search is not enabled on this server"
        )

    query = query_requets.query
    topk = query_requets.topk
    mode = query_requets.mode

    if mode == "new":
        results = Searchers["asr_searcher"].elastic_search(query=query, topk=topk)
    elif mode == "slow":
        results = Searchers["asr_searcher"].search_slow(text=query, num_img=topk)
    elif mode == "fast":
        results = Searchers["asr_searcher"].search_fast(text=query, num_img=topk)
    return SearchResult(results=results)
