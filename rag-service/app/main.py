import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
from rag_pipeline import router as rag_router

app = FastAPI()

app.include_router(rag_router)

@app.get("/")
def read_root():
    return {"Hello": "World"}