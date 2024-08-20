from fastapi import APIRouter, HTTPException

from .models import Query, SearchResult
from ..searchers import Searchers, model_name_to_searcher
from ..helpers import gpt4_translate_vi2en

router = APIRouter(prefix="/search")


@router.get("/", response_model=SearchResult)
def search(query_batch: Query):

    query = query_batch.query

    # translate to english
    # TODO: translation should be in a seperate api
    if query_batch.language == "Vie":
        # to be backward compatible with previous version
        #  where a only the first query is translated
        query = [gpt4_translate_vi2en(query[0])]
        print("Gpt-4 output", query)

    topk = query_batch.k

    model = query_batch.model

    _searcher = model_name_to_searcher(model)

    _searcher_method = _searcher.batch_search_by_text

    if not isinstance(query, list):
        HTTPException(status_code=400, detail="Query must be a list")

    elif query[0].startswith("data:image/"):
        query = [
            Image.open(io.BytesIO(base64.b64decode(item.split(",")[1])))
            for item in query
        ]

        _searcher_method = _searcher.batch_search_by_image

    elif not isinstance(query[0], str):
        HTTPException(
            status_code=400,
            detail="Query must be a list of strings or base64 encoded images",
        )

    result = _searcher_method(query, k)

    return SearchResult(search_result=result)
