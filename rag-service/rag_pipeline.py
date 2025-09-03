import time
import logging
import uuid
import numpy as np
from typing import Dict, Any, List

from fastapi import (HTTPException,
                     status,
                     APIRouter)

from helpers.utils import (load_vector_db,
                           extract_content)

from helpers.schemas import (CompletionRequest,
                             ChatCompletionResponse,
                             ChatChoiceResponse,
                             ChatMessageResponse)

from helpers.dense_retriever import dense_retrieval_instance # Dense Retriever

from rag.embeddings import EmbedderWrapper # Embeddder
from rag.generation import LocalGenerator # Local LLM
from rag.rerank import Reranker # Reranker
from prompting import (build_rag_prompt,
                       build_universal_rag_prompt) # Prompt builder

from constants import (VECTORSTORE_PATH,
                       METADATA_PATH,
                       MODEL_PATH,
                       N_GPU_LAYERS,
                       N_CTX,
                       RERANKER_MODEL)

# Initialize Metadata and Vector Store
index, metadata = load_vector_db(index_file=VECTORSTORE_PATH, metadata_file=METADATA_PATH)

# Intialize RAG components
embedding_model_instance = EmbedderWrapper()
reranker_instance = Reranker(model_name=RERANKER_MODEL)
local_llm_instance = LocalGenerator(model_path=str(MODEL_PATH), n_gpu_layers=N_GPU_LAYERS, n_ctx=N_CTX)

# Initialize FastAPI router
router = APIRouter()

# Define the RAG endpoint
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def rag_endpoint(request: CompletionRequest) -> ChatCompletionResponse:
    return await rag_pipeline(request)

# Full RAG pipeline function
async def rag_pipeline(request: CompletionRequest) -> ChatCompletionResponse:
    """
    Full RAG implementation with embedding → retrieval → reranking → local generation.
    """
    try:
        # RAG Pipeline Steps
        
        # 1) Extract user prompt
        user_prompt = request.prompt
        if not user_prompt and request.messages:
            for message in reversed(request.messages):
                role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
                if role == "user":
                    user_prompt = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "")
                    break
        if not user_prompt:
            raise HTTPException(status_code=400, detail="User prompt is empty.")

        start_time = time.time()
        print("Response with RAG + Reranker starting...")

        # 2) Dense Retrieval
        dense_start = time.time()
        # a) Embed the user query
        query_embedding = np.array(embedding_model_instance.embed([user_prompt]), dtype="float32")
        # b) Retrieve top documents from vector store, change `top_k` as needed
        retrieved_docs: List[Dict[str, Any]] = await dense_retrieval_instance.dense_retrieval(
            query_embedding, index, metadata, top_k=20
        )
        dense_time = time.time() - dense_start
        print(f"Dense retrieval took {dense_time:.4f} seconds")
        print(f"Retrieved {len(retrieved_docs)} documents for reranking.")

        if not retrieved_docs:
            raise HTTPException(status_code=404, detail="No documents retrieved for the given query.")
        
        # Debug: Print first few retrieved documents structure
        print("\n--- DEBUG: First Retrieved Document Structure ---")
        if retrieved_docs:
            first_doc = retrieved_docs[0]
            print(f"Keys in first doc: {list(first_doc.keys())}")
            for key, value in first_doc.items():
                if isinstance(value, str):
                    print(f"{key}: {value[:100]}..." if len(str(value)) > 100 else f"{key}: {value}")
                else:
                    print(f"{key}: {type(value)} - {value}")
        print("---")

        # 3) Rerank the retrieved documents
        # Change `top_n` to control how many top documents to keep after reranking
        reranked_docs: List[Dict[str, Any]] = reranker_instance.rerank(user_prompt, retrieved_docs, top_n=8)
        print(f"Reranked and selected top {len(reranked_docs)} documents.")

        # Debug: Print reranked documents structure
        print("\n--- DEBUG: Reranked Documents ---")
        for i, doc in enumerate(reranked_docs):
            print(f"Reranked doc {i}:")
            print(f"  Keys: {list(doc.keys())}")
            # Look for content in common keys
            content_keys = ['content', 'text', 'chunk', 'page_content']
            for key in content_keys:
                if key in doc:
                    content = str(doc[key])
                    print(f"  {key}: {content[:100]}..." if len(content) > 100 else f"  {key}: {content}")
                    break
            else:
                print("  No content found in common keys")
        print("---")

        # 4) Build context string from reranked documents
        context_str = extract_content(reranked_docs)
        print(f"Context String: {context_str}:")

        # 5) RAG Prompt Construction & Final Prompt
        # You can customize the prompt template as needed
        rag_prompt = build_universal_rag_prompt(user_prompt, context_str)
        # rag_prompt = (
        #     "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        #     "You are an expert assistant for analyzing technical proposals. "
        #     "Answer the user's question based *only* on the provided documents. "
        #     "Do not use any outside knowledge. If the answer is not explicitly present, "
        #     "reply exactly: 'Not stated in the provided documents.'\n\n"
        #     "When possible, structure your response into the following sections:\n"
        #     "1. **Summary** – a concise explanation directly answering the user’s question.\n"
        #     "2. **Key Details** – elaborate with supporting information, features, or context from the documents.\n"
        #     "3. **Deployment/Architecture (if relevant)** – describe how it is implemented or used, if available.\n"
        #     "4. **Sources** – list the most relevant source filenames used.\n\n"
        #     f"Here are the most relevant documents (verbatim excerpts):\n\n{context_str}\n"
        #     "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        #     f"Question: {user_prompt}\n\n"
        #     "Provide your response using the structured format above."
        #     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        # )

        # Just to ensure no Windows-style line endings
        final_prompt = rag_prompt.replace("\r\n", "\n")

        print(f"Final Prompt: {final_prompt}:")

        # 6) Generate response from local LLM
        llm_response = local_llm_instance.generate(final_prompt)

        end_time = time.time()
        print(f"Full response took {end_time - start_time:.4f} seconds.")

        # 7) RAG response formatting
        response = ChatCompletionResponse(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=request.model or "local-llama3-8b-reranked",
            choices=[ChatChoiceResponse(index=0, message=ChatMessageResponse(role="assistant", content=llm_response), finish_reason="stop")],
            usage={
                "prompt_tokens": len(final_prompt.split()),
                "completion_tokens": len(llm_response.split()),
                "total_tokens": len(final_prompt.split()) + len(llm_response.split()),
            },
        )
        print(f"--- Final Answer ---\nQuery: {user_prompt}\n\nLLM Response: {llm_response}\n")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error in RAG pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )