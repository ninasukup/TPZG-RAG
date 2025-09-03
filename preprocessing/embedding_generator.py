import json
import faiss
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    """A wrapper for the SentenceTransformer model."""
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded.")

    def embed(self, texts, batch_size=32):
        """Generates embeddings for a list of texts."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

def generate_embeddings(chunks_file: str = "Neu_Knowledgebase/chunks_flat.jsonl",
                        index_file: str = "faiss_index.bin",
                        metadata_file: str = "metadata.jsonl"):
    """
    Generates embeddings from a flattened chunks file (.jsonl) and builds a FAISS index.
    It also creates a perfectly aligned metadata file for retrieval.
    """
    # --- FIX: Process .jsonl file line by line ---
    # A .jsonl file must be read one JSON object at a time.
    chunks = []
    print(f"Reading chunks from {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"Loaded {len(chunks)} total chunks.")

    # Separate the text content (for embedding) from the metadata
    texts_to_embed = []
    metadata_to_store = []
    for chunk in chunks:
        content = chunk.get("content") or ""
        texts_to_embed.append(content)
        # The entire original chunk object becomes the metadata
        metadata_to_store.append(chunk)

    # Generate embeddings for all the text content
    print("Generating embeddings... (This may take a while depending on the number of chunks)")
    embedder = LocalEmbedder()
    embeddings = embedder.embed(texts_to_embed)
    dimension = embeddings.shape[1]
    print(f"Embeddings generated with dimension: {dimension}")

    # Build and save the FAISS index
    # We use IndexFlatIP for cosine similarity with normalized embeddings
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)
    print(f"✅ FAISS index built and saved to '{index_file}'")

    # Save the aligned metadata
    # This file is now the source of truth for metadata during retrieval
    with open(metadata_file, "w", encoding="utf-8") as f:
        for m in metadata_to_store:
            f.write(json.dumps(m) + "\n")
    print(f"✅ Aligned metadata saved to '{metadata_file}'")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # This script will overwrite your existing index and metadata files.
    # Make sure your input file `chunks_flat.jsonl` is correct.
    generate_embeddings(
        chunks_file="Neu_Knowledgebase/chunks_flat.jsonl",
        index_file="faiss_index.bin",
        metadata_file="metadata.jsonl" # This will be the new metadata file for your RAG pipeline
    )