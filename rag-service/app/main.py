import faiss
import json
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query

app = FastAPI()

# Load embeddings + metadata
INDEX_FILE = "faiss_index.bin"
META_FILE = "metadata.jsonl"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

print("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_FILE)
metadata = [json.loads(line) for line in open(META_FILE, "r", encoding="utf-8")]
embedder = SentenceTransformer(MODEL_NAME)
print(f"Loaded {len(metadata)} documents into memory.")

@app.get("/search")
def search(query: str = Query(...), top_k: int = 5):
    # Embed query
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    # Search FAISS
    scores, indices = index.search(q_emb, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(metadata):
            results.append({
                "score": float(score),
                "metadata": metadata[idx]
            })
    return {"query": query, "results": results}