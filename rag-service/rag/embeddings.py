import os
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()
HF_DEFAULT = os.getenv("HF_EMBED_MODEL", "BAAI/bge-base-en-v1.5")

try:
    # Try to import LocalEmbedder from embedding_generator if present
    from embedding_generator import LocalEmbedder  # type: ignore
    class EmbedderWrapper:
        def __init__(self, model_name: Optional[str] = None):
            self.model_name = model_name or HF_DEFAULT
            self.local = LocalEmbedder(model_name=self.model_name)
        def embed(self, texts: List[str], batch_size: int = 32):
            return self.local.embed(texts, batch_size=batch_size)
except Exception:
    # fallback to sentence-transformers directly
    from sentence_transformers import SentenceTransformer
    class EmbedderWrapper:
        def __init__(self, model_name: Optional[str] = None):
            self.model_name = model_name or HF_DEFAULT
            self.model = SentenceTransformer(self.model_name)
        def embed(self, texts: List[str], batch_size: int = 32):
            emb = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            return emb.tolist()
