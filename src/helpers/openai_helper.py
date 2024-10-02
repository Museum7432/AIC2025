from config import settings
from openai import OpenAI
import re

client = None
if settings.openai_api_key:
    client = OpenAI(api_key=settings.openai_api_key)
else:
    print("openai api key not provied, translation will be disabled")


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


def remove_numeric_prefix(s):
    # Regular expression to match a string that starts with a number followed by a period
    pattern = r'^\d+\.\s*'
    
    # Use re.sub to replace the prefix with an empty string
    return re.sub(pattern, '', s)


def gpt4_split_query(query, mode="static"):
    assert mode in ["static", "video"]

    if mode == "static":
        sys_prompt = """You are a text processing assistant. Your task is to take a description of an image and split it into multiple smaller independent and descriptive queries suitable for the OpenAI CLIP model. Each query should be an accurate paraphrase of a specific aspect of the original text and must be adequate on its own without adding unnecessary details while enhancing clarity and detail. Do not add new information that was not in the original text. The same details can be repeated when needed. The location of each object whould be relative to other objects. Remove information about the setting or event of the scene. Do not make assumption about the setting or event. Each query should start with \"a photo of\". Return one query if you cannot split it."""
    else:
        sys_prompt = """You are a text processing assistant. Your task is to take a description of what happen in a video and generate multiple smaller independent queries of static image/keyframe suitable for the OpenAI CLIP model. Each query should be an accurate desciption of a event in that video and must be adequate on its own without adding unnecessary details while enhancing clarity and detail. Do not add new information that was not in the original text. The same details can be repeated. The location of each object whould be relative to other objects. Remove information about the setting or event of the scene. Do not make assumption about the setting or event. Each query should start with \"a photo of\". those queries must happen in a temporal order and two queries cannot descibe the same event."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
    response = [remove_numeric_prefix(s).strip() for s in response]


    return response
