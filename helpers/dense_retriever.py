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
                payload = metadata[idx]

                if "content" in payload and (payload["content"] or "").strip():
                    meta = payload.get("metadata", {})
                    retrieved_docs.append({
                        "content": payload["content"],
                        "metadata": meta
                    })
                else:
                    meta = payload.get("metadata", payload)
                    content = (meta.get("content") or "").strip()
                    retrieved_docs.append({
                        "content": content,
                        "metadata": meta
                    })

            elapsed = time.time() - start_time
            logging.info(f"Dense retrieval took {elapsed:.4f} seconds - Retrieved {len(retrieved_docs)} docs")
            return retrieved_docs

        except Exception as e:
            print(f"Error during dense retrieval: {e}")
            raise
        
dense_retrieval_instance = DenseRetrieval(dense_retriever_config)