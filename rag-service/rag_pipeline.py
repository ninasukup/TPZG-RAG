# ------------------------------------------------------------------------------
# rag_service/rag_pipeline.py - RAG Logic callable function.
# ------------------------------------------------------------------------------

from fastapi import (HTTPException,
                     status,
                     APIRouter)
import uuid

from helpers.utils import load_vector_db
from helpers.schemas import (CompletionRequest,
                             ChatCompletionResponse,
                             ChatChoiceResponse,
                             ChatMessageResponse)

from helpers.dense_retriever import dense_retrieval_instance
from rag.embeddings import EmbedderWrapper

from constants import (VECTORSTORE_PATH,
                       METADATA_PATH,
                       MODEL_PATH,
                       N_GPU_LAYERS,
                       N_CTX)

import numpy as np
import time
import logging

from rag.generation import LocalGenerator


index, metadata = load_vector_db(index_file=VECTORSTORE_PATH, metadata_file=METADATA_PATH)
embedding_model_instance = EmbedderWrapper()
local_llm_instance = LocalGenerator(model_path=MODEL_PATH, n_gpu_layers=N_GPU_LAYERS, n_ctx=N_CTX)

router = APIRouter()

@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def rag_endpoint(request: CompletionRequest) -> ChatCompletionResponse:
    return await rag_pipeline(request)

async def rag_pipeline(request: CompletionRequest) -> ChatCompletionResponse:
    """
    Full FAU RAG implementation.
    """
    try:
        if not request.prompt and request.messages:
            for message in reversed(request.messages):
                role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
                content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")

                if role == "user":
                    request.prompt = content
                    break

        user_prompt = request.prompt or ""

        start_time = time.time()

        print("Response with RAG starting...")

        # Step 1: RAG retrieval
        query_embedding = embedding_model_instance.embed([user_prompt])

        query_embedding = np.array(query_embedding, dtype="float32")

        retrieved_docs = await dense_retrieval_instance.dense_retrieval(query_embedding, index, metadata, top_k=20)
        print(f"Retrieved Documents: {retrieved_docs}")
        logging.info(f"Retrieved Documents: {retrieved_docs}")
        print(f"Retrieved {len(retrieved_docs)} documents.")

        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            # doc is expected to be a dict with 'content' and 'metadata' keys
            content = doc.get('content', '')
            source = doc.get('metadata', {}).get('source_filename', 'Unknown')
            page = doc.get('metadata', {}).get('page_numbers', ['N/A'])[0]
            context_str += f"--- Document {i+1} (Source: {source}, Page: {page}) ---\n"
            context_str += content + "\n\n"

        # Step 2: Format the RAG prompt
        rag_prompt = (
            f"You are a helpful university admissions assistant. Your task is to answer the user's question based *only* on the provided documents: {info}. "
            f"The user's question is: '{user_prompt}'.\n\n"
            
            f"Please provide a comprehensive and well-structured answer in English. Your answer should:\n"
            f"1.  **Start with a direct summary** of the main point.\n"
            f"2.  **Use Markdown headings (##)** to create clear sections for different topics (e.g., 'Required Documents', 'Application Process').\n"
            f"3.  **Elaborate on each point.** Instead of just listing a document, briefly explain its significance. For example, if a language certificate is needed, explain what it proves.\n"
            f"4.  **Synthesize information** into a cohesive narrative rather than just extracting bullet points.\n"
            f"5.  **Bold key terms** to make the answer easy to scan.\n"
            f"6.  Conclude with a 'Sources' section, listing the relevant URLs you used from the documents.\n\n"
            
            f"--- [BEGIN RESPONSE] ---"
        )

        # Step 3: LLM Generation
        llm_response = local_llm_instance.generate(rag_prompt)

        # Step 4: Create response object
        response = ChatCompletionResponse(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=request.model or "local-llama3-8b",
            choices=[
                ChatChoiceResponse(
                    index=0,
                    message=ChatMessageResponse(
                        role="assistant",
                        content=llm_response
                    ),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(rag_prompt.split()),
                "completion_tokens": len(llm_response.split()),
                "total_tokens": len(rag_prompt.split()) + len(llm_response.split())
            }
        )

        print(f"--- Final Answer ---\nQuery: {user_prompt}\n\nLLM Response: {llm_response}\n")
        return response

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )