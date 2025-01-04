from fastapi import APIRouter, HTTPException
import io
import base64

from .models import SingleTextQuery, SearchResult
from searchers import Searchers, get_ft_searcher
from helpers import gpt4_translate_vi2en

router = APIRouter(prefix="/search")


@router.post("/", response_model=SearchResult)
def search(request: SingleTextQuery):

    query = request.query

    # vi2en translation
    # TODO: translation should be in a seperate api
    if request.language == "Vie":
        query = gpt4_translate_vi2en(query)
        print("Gpt-4 output", query)

    topk = request.topk

    model = request.model

    _searcher = get_ft_searcher(model)

    _searcher_method = _searcher.search_by_texts


    if query.startswith("data:image/"):

        query = Image.open(io.BytesIO(base64.b64decode(query.split(",")[1])))

        _searcher_method = _searcher.search_by_images

    result = _searcher_method([query], topk)[0]


    return SearchResult(results=result)
