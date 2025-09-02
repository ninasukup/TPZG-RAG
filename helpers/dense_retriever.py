# ------------------------------------------------------------------------------
# retrievers/dense_retrieval.py - Dense retrieval method for information retrieval
# ------------------------------------------------------------------------------
"""
Implements the dense retrieval method using semantic similarity.
"""

from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from pathlib import Path

from constants import CONFIG_FILE_PATH
from helpers.utils import read_yaml

import time
import logging
import numpy as np

@dataclass(frozen = True)
class DenseRetrieverConfig:
    transformer: Path

class ConfigurationManager:
    def __init__(self,
                config_filepath = CONFIG_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.dense_retriever_config = DenseRetrieverConfig(
            transformer = self.config["dense_retriever"]["transformer"]
        )

    def get_dense_retriever_config(self) -> DenseRetrieverConfig:
        return self.dense_retriever_config
    
config_manager = ConfigurationManager()
dense_retriever_config = config_manager.get_dense_retriever_config()

class DenseRetrieval:

    def __init__(self, config: DenseRetrieverConfig):
        self.config = config
        self.retriever = SentenceTransformer(self.config.transformer)

    @staticmethod
    async def dense_retrieval(query_embedding, index, metadata, top_k):
        try:
            start_time = time.time()

            _, indices = index.search(query_embedding, top_k)

            retrieved_docs = []
            for idx in indices[0]:
                payload = metadata[idx]  # could be {"content": "...", "metadata": {...}} OR {"metadata": {...}}

            if "content" in payload and (payload["content"] or "").strip():
                meta = payload.get("metadata", {})
                retrieved_docs.append({"content": payload["content"], "metadata": meta})
            else:
                meta = payload.get("metadata", payload)
                content = (meta.get("content") or "").strip()
                retrieved_docs.append({"content": content, "metadata": meta})

            elapsed = time.time() - start_time
            print(f"Dense retrieval took {elapsed:.4f} seconds")
            logging.info("Documents Retrieved Successfully! - Dense R")
            return retrieved_docs

        except Exception as e:
            print(f"Error during dense retrieval: {e}")
            raise
        
    def dense_embeddings(dense_model, metadata, batch_size = 32):
        try:
            if not metadata:
                raise ValueError("Metadata Empty or None!")
            
            for entry in metadata:
                if entry.get("text"):
                    texts = [entry.get("text")]

            if not texts:
                raise ValueError("No valid Text entries found in your Metadata!")
            
            print(f"Total valid texts to process: {len(texts)}")

            embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(total_batches):
                batch_start = i * batch_size
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start::batch_end]

                try:
                    print(f"Processing batch {i + 1}/{total_batches}...")
                    batch_embeddings = dense_model.encode(batch_texts, convert_to_numpy=True)
                    embeddings.extend(batch_embeddings)

                except Exception as e:
                    print(f"Error processing batch {i + 1}: {e}")
                    continue

            embeddings = np.array(embeddings)
            print(f"Computed embeddings for {embeddings.shape[0]} entries.")
            return embeddings
    
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise e
        
dense_retrieval_instance = DenseRetrieval(dense_retriever_config)