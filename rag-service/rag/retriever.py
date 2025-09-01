from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv()

from .embeddings import EmbedderWrapper as Embedder
from ...vectorstore import FaissStore

class Retriever:
    def __init__(self, index_path: str, backend: str = "hf"):
        self.store = FaissStore(index_path=index_path)
        self.embedder = Embedder()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qvec = self.embedder.embed([query])[0]
        results = self.store.search(qvec, top_k=top_k)
        docs = []
        for score, meta in results:
            docs.append({
                "score": score,
                "content": meta.get("content", ""),
                "metadata": {k: v for k, v in meta.items() if k != "content"}
            })
        return docs
