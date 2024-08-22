import io
import base64
from typing import List, Tuple, Dict
from PIL import Image

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, conlist

# load searchers on startup
from searchers import SearchersLifespan
from routes import asr_route, ocr_route, img_search_route, obj_c_route, frame_nei_route

from config import settings

app = FastAPI(title=settings.app_name, lifespan=SearchersLifespan)


# mới thêm 28_7_24
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Cho phép tất cả các nguồn gốc. Bạn có thể giới hạn lại theo nhu cầu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home() -> None:
    return "Welcome to the Image Semantic Search API. Head over http://localhost:8000/docs for more info."


app.include_router(asr_route)
app.include_router(ocr_route)
app.include_router(img_search_route)
app.include_router(obj_c_route)
app.include_router(frame_nei_route)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
