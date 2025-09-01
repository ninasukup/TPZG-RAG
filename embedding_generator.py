import json
import faiss
from sentence_transformers import SentenceTransformer

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

def generate_embeddings(chunks_file="Neu_Knowledgebase/chunks_output.jsonl",
                        index_file="faiss_index.bin",
                        metadata_file="metadata.jsonl"):
    # Load chunks
    texts, metadata = [], []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            metadata.append({k: v for k, v in obj.items() if k != "text"})

    print(f"Loaded {len(texts)} chunks from {chunks_file}")

    # Generate embeddings
    embedder = LocalEmbedder()
    embeddings = embedder.embed(texts)
    dim = embeddings.shape[1]

    # Build FAISS index
    index = faiss.IndexFlatIP(dim)  # cosine similarity (since normalized)
    index.add(embeddings)
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}")

    # Save metadata aligned with embeddings
    with open(metadata_file, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")
    print(f"Metadata saved to {metadata_file}")

if __name__ == "__main__":
    generate_embeddings()