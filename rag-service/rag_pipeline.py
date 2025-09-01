# ------------------------------------------------------------------------------
# rag_service/rag_pipeline.py - RAG Logic callable function.
# ------------------------------------------------------------------------------

from fastapi import (HTTPException,
                     status,
                     APIRouter)

from helpers.utils import load_vector_db
from helpers.schemas import (CompletionRequest,
                             ChatCompletionResponse,
                             ChatChoiceResponse,
                             ChatMessageResponse)

from helpers.dense_retriever import dense_retrieval_instance
from rag.embeddings import EmbedderWrapper
# from rag_service.university_api import query_university_endpoint
from constants import (VECTORSTORE_PATH,
                       METADATA_PATH)

import numpy as np
import time
import logging

index, metadata = load_vector_db(index_file=VECTORSTORE_PATH, metadata_file=METADATA_PATH)
embedding_model_instance = EmbedderWrapper()

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

        # Step 2: Format the RAG query
        # rag_query = (
        #     f"You are a helpful university admissions assistant. Your task is to answer the user's question based *only* on the provided documents: {info}. "
        #     f"The user's question is: '{user_prompt}'.\n\n"
            
        #     f"Please provide a comprehensive and well-structured answer in English. Your answer should:\n"
        #     f"1.  **Start with a direct summary** of the main point.\n"
        #     f"2.  **Use Markdown headings (##)** to create clear sections for different topics (e.g., 'Required Documents', 'Application Process').\n"
        #     f"3.  **Elaborate on each point.** Instead of just listing a document, briefly explain its significance. For example, if a language certificate is needed, explain what it proves.\n"
        #     f"4.  **Synthesize information** into a cohesive narrative rather than just extracting bullet points.\n"
        #     f"5.  **Bold key terms** to make the answer easy to scan.\n"
        #     f"6.  Conclude with a 'Sources' section, listing the relevant URLs you used from the documents.\n\n"
            
        #     f"--- [BEGIN RESPONSE] ---"
        # )

        # print(f"RAG Query: {rag_query}")
        # logging.info(f"RAG Query: {rag_query}")

        # # Step 3: Call the LLM via the university endpoint
        # uni_response = await query_university_endpoint(rag_query, 'FAU RAG')
        # logging.info(f"{uni_response}")

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Response took {elapsed_time:.4f} seconds")
        # print(f"Citations: {citation_section}")
        # logging.info(f"Finished query in: {end_time - start_time} seconds")

        # final_content = f"{uni_response}{citation_section}"

        # response = ChatCompletionResponse(
        #     id=str(uuid.uuid4()),
        #     object="chat.completion",
        #     created=int(time.time()),
        #     model=request.model or "default-model",
        #     choices=[
        #         ChatChoiceResponse(
        #             index=0,
        #             message=ChatMessageResponse(
        #                 role="assistant",
        #                 content=f"{uni_response}{citation_section}"
        #             ),
        #             finish_reason="stop"
        #         )
        #     ],
        #     usage={
        #         "prompt_tokens": len(user_prompt.split()),
        #         "completion_tokens": len(uni_response.split()),
        #         "total_tokens": len(user_prompt.split()) + len(uni_response.split())
        #     }
        # )

        # print(f"Query:\n{user_prompt}\n\nUniversity Response with Citations:\n{final_content}\n")
        # return response

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )