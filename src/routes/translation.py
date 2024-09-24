from fastapi import APIRouter, HTTPException

from .models import TranslationQuery, TranslationResult

from helpers import gpt4_translate_vi2en

router = APIRouter(prefix="/translate")


@router.post("/", response_model=TranslationResult)
def translate(request: TranslationQuery):

    texts = request.texts

    texts = [gpt4_translate_vi2en(t) for t in texts]

    return TranslationResult(texts=texts)