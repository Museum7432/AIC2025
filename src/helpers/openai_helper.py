from config import settings
from openai import OpenAI
import re
from .prompts import *

client = None
if settings.openai_api_key:
    client = OpenAI(api_key=settings.openai_api_key)
else:
    print("openai api key not provied, translation will be disabled")

from functools import lru_cache


@lru_cache(maxsize=10000)
def gpt4_translate_vi2en(text, source_lang="Vietnamese"):
    assert client is not None, "require openai api key"

    assert source_lang in ["Vietnamese"]

    response = client.chat.completions.create(
        model=settings.gpt_translate_model,  # default to gpt-4o-mini
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant in translating documents from {source_lang} to English, do not translate name, street, organization name, location, character between 2 quotes symbol.",
            },
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


@lru_cache(maxsize=10000)
def gpt4_translate_uni2en(text):
    assert client is not None, "require openai api key"

    response = client.chat.completions.create(
        model=settings.gpt_translate_model,  # default to gpt-4o-mini
        messages=[
            {
                "role": "system",
                "content": sys_prompt_translate_uni,
            },
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


def remove_numeric_prefix(s):
    # Regular expression to match a string that starts with a number followed by a period
    pattern = r"^\d+\.\s*"

    # Use re.sub to replace the prefix with an empty string
    return re.sub(pattern, "", s)


@lru_cache(maxsize=10000)
def gpt4_split_query(query, mode="static"):

    if mode == "static":
        model = "gpt-4o"
        sys_prompt = sys_prompt_fused_en
    elif mode == "video":
        model = "gpt-4o"
        sys_prompt = sys_prompt_temporal_en
    elif mode == "static_uni":
        model = "gpt-4o"
        sys_prompt = sys_prompt_fused_uni
    else:
        model = "gpt-4o"
        sys_prompt = sys_prompt_temporal_uni

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {"role": "user", "content": query},
        ],
    )

    response = response.choices[0].message.content

    response = response.split("\n")

    response = [s.strip() for s in response]

    response = [t for t in response if len(t) > 1]

    # response = [remove_numeric_prefix(s).strip() for s in response]

    return response
