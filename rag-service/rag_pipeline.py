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
from helpers.dense_retriever import dense_retrieval_instance
from rag.embeddings import EmbedderWrapper
from rag.generation import LocalGenerator
from rag.rerank import Reranker
from constants import (VECTORSTORE_PATH,
                       METADATA_PATH,
                       MODEL_PATH,
                       N_GPU_LAYERS,
                       N_CTX,
                       RERANKER_MODEL)

# --- Initialize Global Instances ---
index, metadata = load_vector_db(index_file=VECTORSTORE_PATH, metadata_file=METADATA_PATH)
embedding_model_instance = EmbedderWrapper()
reranker_instance = Reranker(model_name=RERANKER_MODEL)
local_llm_instance = LocalGenerator(model_path=str(MODEL_PATH), n_gpu_layers=N_GPU_LAYERS, n_ctx=N_CTX)

router = APIRouter()

@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def rag_endpoint(request: CompletionRequest) -> ChatCompletionResponse:
    return await rag_pipeline(request)


async def rag_pipeline(request: CompletionRequest) -> ChatCompletionResponse:
    """
    Full RAG implementation with retrieval, reranking, and local generation.
    """
    try:
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
        query_embedding = np.array(embedding_model_instance.embed([user_prompt]), dtype="float32")
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

        # 3) Rerank
        reranked_docs: List[Dict[str, Any]] = reranker_instance.rerank(user_prompt, retrieved_docs, top_n=5)
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

        # 5) Build context string
        context_str = extract_content(reranked_docs)

        # 6) Compose prompt (tighten task & output format)
        rag_prompt_template = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are an expert assistant for analyzing technical proposals. Answer the user's question based only on the provided documents."
            " If the answer is not explicitly present, reply exactly: 'Not stated in the provided documents.'\n"
            f"Here are the most relevant documents (verbatim excerpts):\n\n{context_str}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Question: {user_prompt}\n\n"
            "Please answer in one concise sentence. Also include the most likely source filename you used."
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        final_prompt = rag_prompt_template.replace("\r\n", "\n")

        print("\n--- FINAL PROMPT SENT TO LLM ---\n")
        print(final_prompt)
        print("\n---------------------------------\n")

        # 7) Generate
        llm_response = local_llm_instance.generate(final_prompt)

        end_time = time.time()
        print(f"Full response took {end_time - start_time:.4f} seconds.")

        # 8) Return structured response
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