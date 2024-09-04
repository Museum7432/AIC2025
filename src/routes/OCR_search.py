from fastapi import APIRouter, HTTPException

from .models import SearchResult, OcrQuery
from searchers import Searchers

router = APIRouter(prefix="/search_OCR")


@router.post("/", response_model=SearchResult)
def search_OCR(request: OcrQuery) -> SearchResult:
    query = request.query
    topk = request.topk

    mode = request.mode

    # results=search_compare_similirity_word_load_fulldatabase_to_ram(query[0],database=dbOCR,num_img=k)#simple thread

    if mode == "elastic":
        results = Searchers["ocr_searcher"].elastic_search(query=query, topk=topk)
    else:
        results = Searchers["ocr_searcher"].search(query=query, num_img=topk)

    return SearchResult(results=results)
