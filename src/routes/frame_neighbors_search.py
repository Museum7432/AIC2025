from fastapi import APIRouter, HTTPException

from .models import SearchResult, FrameNeighborsQuery
from searchers import Searchers, get_faiss_searcher

router = APIRouter(prefix="/search_by_frame")


@router.post("/", response_model=SearchResult)
def search_ASR(request: FrameNeighborsQuery) -> SearchResult:

    _searcher = get_faiss_searcher(request.model)

    results = _searcher.search_by_indexed_image(
        video_name=request.video_name, frame_idx=request.frame_idx, topk=request.topk
    )

    return SearchResult(results=results)
