from pydantic_settings import BaseSettings, SettingsConfigDict

# https://fastapi.tiangolo.com/advanced/settings/
class Settings(BaseSettings):
    # ignored if .env file is not set
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "ELO@AIC Image Semantic Search"

    openai_api_key:str=""

    gpt_translate_model:str="gpt-4o-mini"

settings = Settings()