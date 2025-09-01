# Minimal RAG Service (FastAPI + FAISS) - Updated for Local Embeddings

This variant defaults to **local SentenceTransformers** embeddings (BAAI/bge-base-en-v1.5).
Place your `chunking_pipeline.py` and `embedding_generator.py` at the project root (next to this folder).
See `.env.example` for configuration.
