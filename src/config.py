from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union


# https://fastapi.tiangolo.com/advanced/settings/
class Settings(BaseSettings):
    # ignored if .env file is not set
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "ELO@AIC Image Semantic Search"

    openai_api_key: Union[str, None] = None

    gpt_translate_model: str = "gpt-4o-mini"

    ocr_path: Union[str, None] = "data/embeddings/OCR_results"

    object_counting_path: Union[str, None] = "data/embeddings/obj_cnt_data"

    asr_path: Union[str, None] = "data/embeddings/asr_processed"

    blip2_embs_path: Union[str, None] = "data/embeddings/keyframes_embs_blip2"

    clip_H_embs_path: Union[str, None] = "data/embeddings/keyframes_embs_clip_H"

    clip_bigG_embs_path: Union[str, None] = "data/embeddings/ViT-bigG-14-CLIPA-336-datacomp1b"
    
    clip_S400M_embs_path: Union[str, None] = "data/embeddings/keyframes_embs_clip_S400M"

    # for testing
    # "data/keyframes_embs_clip_B32", "ViT-B-32", "openai"
    clip_B32_embs_path: Union[str, None] = None

    device: str = "cpu"
    
    # docker container
    elastic_endpoint: str = "http://elasticsearch:9200"
    # if elastic_password is None, disable all features 
    # that depend on it
    # set up a .env with the sample .env_example
    # TODO: use other type of authentication
    elastic_username:str = "elastic"
    elastic_password: Union[str, None] = None

    # remove the old index on load
    # useful for development
    # should be disabled in production
    # since create a new index is slow 
    remove_old_index:bool=False


settings = Settings()

if __name__ == "__main__":
    print(settings)
