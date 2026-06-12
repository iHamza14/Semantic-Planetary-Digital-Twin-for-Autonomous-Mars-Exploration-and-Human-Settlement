from fastapi import FastAPI

from .api.routes import router

app = FastAPI(title="Offroad Segmentation API", version="1.0.0")
app.include_router(router)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only, relax
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)