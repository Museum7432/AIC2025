from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union


# https://fastapi.tiangolo.com/advanced/settings/
class Settings(BaseSettings):
    # ignored if .env file is not set
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "ELO@AIC Image Semantic Search"

    openai_api_key: Union[str, None] = None

    gpt_translate_model: str = "gpt-4o-mini"

    ocr_path: Union[str, None] = "data/text_extracted"

    object_counting_path: Union[str, None] = "data/Object_Counting_vector_np"

    asr_path: Union[str, None] = "data/ASR_folder"

    blip2_embs_path: Union[str, None] = "data/keyframes_embs_blip2"

    clip_H_embs_path: Union[str, None] = "data/keyframes_embs_clip_H"

    clip_bigG_embs_path: Union[str, None] = "data/ViT-bigG-14-CLIPA-336-datacomp1b"

    # for testing
    # "data/keyframes_embs_clip_B32", "ViT-B-32", "openai"
    clip_B32_embs_path: Union[str, None] = None

    device: str = "cpu"


settings = Settings()

if __name__ == "__main__":
    print(settings)
