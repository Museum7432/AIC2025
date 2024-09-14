from fastapi import APIRouter, HTTPException

from .models import SearchResult, ObjectLocationQuery
from searchers import Searchers

router = APIRouter(prefix="/search_ObjectLoc")


@router.post("/", response_model=SearchResult)
def search_ObjectLoc(request: ObjectLocationQuery) -> SearchResult:
    class_ids = request.class_ids
    box_cords = request.box_cords
    topk = request.topk

    results = Searchers["obj_loc_searcher"].elastic_search(
        class_ids=class_ids, box_cords=box_cords, topk=topk
    )

    return SearchResult(results=results)
