from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class Reranker:
    def __init__(self, model_name: str):
        print("Initializing Reranker...")
        try:
            self.model = CrossEncoder(model_name)
            print("✅ Reranker loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load Reranker model. Error: {e}")
            self.model = None

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        if not self.model or not docs:
            return docs

        # The model expects pairs of [query, document_content]
        pairs = [[query, doc.get('content', '')] for doc in docs]
        
        # Get scores from the model
        scores = self.model.predict(pairs)
        
        # Add scores to the documents and sort
        for i, doc in enumerate(docs):
            doc['rerank_score'] = scores[i]
            
        # Sort documents by the new rerank score in descending order
        reranked_docs = sorted(docs, key=lambda x: x['rerank_score'], reverse=True)
        
        # Return the top N documents
        return reranked_docs[:top_n]

