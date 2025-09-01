import json
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

class LocalEmbedder:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts, batch_size=32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

def generate_embeddings(chunks_file="Neu_Knowledgebase/chunks_output.json",
                        index_file="faiss_index.bin",
                        metadata_file="metadata.jsonl",
                        append=False):
    # Load and flatten chunks from dict-of-lists format
    texts, metadata = [], []
    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Flatten all chunk lists from all keys
        flat_chunks = []
        for chunk_list in data.values():
            flat_chunks.extend(chunk_list)
        data = flat_chunks

    for obj in data:
        # Flexible key detection
        text = obj.get("content") or obj.get("text") or obj.get("chunk")
        if not text:
            # fallback: longest string value in the object
            text = max((v for v in obj.values() if isinstance(v, str)), key=len, default="")
        texts.append(text)
        obj["content"] = text   # ensure consistency
        metadata.append({k: v for k, v in obj.items() if k != "content"})

    print(f"Loaded {len(texts)} chunks from {chunks_file}")

    # Generate embeddings
    embedder = LocalEmbedder()
    embeddings = embedder.embed(texts)
    dim = embeddings.shape[1]

    # Build FAISS index
    if Path(index_file).exists() and append:
        index = faiss.read_index(str(index_file))
        index.add(embeddings)
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    faiss.write_index(index, str(index_file))
    print(f"FAISS index saved to {index_file}")

    # Save metadata aligned with embeddings
    mode = "a" if append and Path(metadata_file).exists() else "w"
    with open(metadata_file, mode, encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")
    print(f"Metadata saved to {metadata_file}")

if __name__ == "__main__":
    generate_embeddings()