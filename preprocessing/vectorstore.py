import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np

class FaissStore:
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_path / "vectors.faiss"
        self.meta_file = self.index_path / "metadata.jsonl"

        self.index = None
        self.dim = None
        self.metadata: List[Dict[str, Any]] = []

        if self.index_file.exists() and self.meta_file.exists():
            self._load()

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
        with open(self.meta_file, "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def _load(self):
        self.index = faiss.read_index(str(self.index_file))
        self.metadata = []
        with open(self.meta_file, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        if self.index.ntotal > 0:
            self.dim = self.index.d
        else:
            self.dim = None

    def add(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        vecs = np.array(embeddings, dtype="float32")
        if self.index is None:
            self.dim = vecs.shape[1]
            # use Inner Product over normalized vectors for cosine similarity
            self.index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.metadata.extend(metadatas)
        self._save()

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append((float(score), meta))
        return results
