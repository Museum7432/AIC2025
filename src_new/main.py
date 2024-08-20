import io
import base64
from typing import List, Tuple, Dict
from PIL import Image
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, conlist

from searchers import SearchersLifespan




app = FastAPI(title="ELO@AIC Image Semantic Search", lifespan=SearchersLifespan)

#mới thêm 28_7_24
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn gốc. Bạn có thể giới hạn lại theo nhu cầu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

