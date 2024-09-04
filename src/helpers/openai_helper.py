from config import settings
from openai import OpenAI

client = None
if settings.openai_api_key:
    client = OpenAI(api_key=settings.openai_api_key)
else:
    print("openai api key not provied, translation will be disabled")


def gpt4_translate_vi2en(text):
    assert client is not None, "require openai api key"
    
    response = client.chat.completions.create(
        model=settings.gpt_translate_model,  # default to gpt-4o-mini
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant in translating documents from Vietnamese to English, do not translate name, street, organization name, location, character between 2 quotes symbol.",
            },
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content
