import time
import logging
import uuid
import numpy as np
from typing import Dict, Any, List, Tuple

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

# ------------------------
# Helpers
# ------------------------

def _rehydrate_content(doc: Dict[str, Any], meta_lookup: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return (content, meta) for a retrieved doc with multiple fallbacks.

    The main issue you are seeing (context showing only table-of-contents lines)
    usually happens because the retriever returns items whose text is stored
    under different keys or needs a metadata lookup. This function standardizes
    that.
    """
    # Prefer explicit text-like fields first
    content = (
        doc.get("content")
        or doc.get("text")
        or doc.get("chunk")
        or doc.get("page_content")
        or ""
    )

    meta = doc.get("metadata", {})

    # Some implementations store text in metadata
    if not content:
        content = meta.get("content", meta.get("text", ""))

    # As a last resort, try to rehydrate via metadata lookup by id / document_id
    if not content:
        lookup_keys = [
            doc.get("id"),
            doc.get("document_id"),
            meta.get("id"),
            meta.get("document_id"),
        ]
        for k in lookup_keys:
            if k and k in meta_lookup:
                maybe = meta_lookup[k]
                if isinstance(maybe, dict):
                    content = (
                        maybe.get("content")
                        or maybe.get("text")
                        or maybe.get("chunk")
                        or maybe.get("page_content")
                        or maybe.get("raw_text", "")
                    )
                    meta = {**maybe, **meta}
                elif isinstance(maybe, str):
                    content = maybe
                if content:
                    break

    # If *still* empty, attempt a human-usable fallback (section headers, etc.)
    if not content:
        if "section_hierarchy" in doc and isinstance(doc["section_hierarchy"], list):
            content = "\n".join(doc["section_hierarchy"])  # last ditch fallback
        else:
            content = ""

    return content, meta


def _format_source(meta: Dict[str, Any], doc: Dict[str, Any]) -> Tuple[str, str]:
    source = meta.get("source_filename") or doc.get("source_filename") or meta.get("source") or "Unknown"
    # Try a few common page encodings
    page = (
        meta.get("page")
        or (meta.get("page_numbers", ["N/A"]) or ["N/A"])[0]
        or meta.get("page_number")
        or "N/A"
    )
    return str(source), str(page)


def _expand_query(user_prompt: str) -> str:
    """Lightweight query expansion for common acronyms/aliases.

    Keeps embedding simple while improving recall for terse questions like
    "What MCS do we have?".
    """
    expansions = [
        "MCS", "Mission Control System", "Mission Control Systems",
        "SCOS-2000", "SCOS 2000", "SCOS2000", "Mission Operations System",
    ]
    # Only add expansions that are not already present to avoid prompt bloat
    to_add = [e for e in expansions if e.lower() not in user_prompt.lower()]
    if not to_add:
        return user_prompt
    return f"{user_prompt} (Also consider: {', '.join(to_add)})"


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def rag_endpoint(request: CompletionRequest) -> ChatCompletionResponse:
    return await rag_pipeline(request)


async def rag_pipeline(request: CompletionRequest) -> ChatCompletionResponse:
    """
    Full RAG implementation with retrieval, reranking, and local generation.
    Now with:
      - Query expansion for acronyms
      - Robust content rehydration & filtering
      - Safer prompt construction
      - Better logging for observability
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

        # 2) Dense Retrieval (with light query expansion to improve recall)
        expanded_query = _expand_query(user_prompt)
        query_embedding = np.array(embedding_model_instance.embed([expanded_query]), dtype="float32")
        retrieved_docs: List[Dict[str, Any]] = await dense_retrieval_instance.dense_retrieval(
            query_embedding, index, metadata, top_k=60
        )
        print(f"Retrieved {len(retrieved_docs)} documents for reranking.")

        if not retrieved_docs:
            raise HTTPException(status_code=404, detail="No documents retrieved for the given query.")

        # 3) Rerank
        reranked_docs: List[Dict[str, Any]] = reranker_instance.rerank(user_prompt, retrieved_docs, top_n=25)
        print(f"Reranked and selected top {len(reranked_docs)} documents.")

        # 4) Rehydrate + filter useless chunks (empty/too short)
        processed: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []  # (content, meta, original_doc)
        for d in reranked_docs:
            content, meta = _rehydrate_content(d, metadata or {})
            content = (content or "").strip()
            if len(content) >= 40:  # keep your existing noise filter
                processed.append((content, meta, d))
            if len(processed) >= 10:
                break

        # If we still have fewer than 10, allow short-but-non-empty text to reach 10
        if len(processed) < 10:
            for d in reranked_docs:
                content, meta = _rehydrate_content(d, metadata or {})
                text = (content or "").strip()
                if text and all(p[2] is not d for p in processed):
                    processed.append((text, meta, d))
                    if len(processed) >= 10:
                        break

        # Cap to 10
        processed = processed[:10]
        if not processed:
            # Fall back to best-effort: keep up to 10 original docs (even if short)
            for d in reranked_docs[:10]:
                content, meta = _rehydrate_content(d, metadata or {})
                processed.append(((content or "").strip(), meta, d))

        # 5) Build context string
        context_parts: List[str] = []
        for i, (content, meta, original) in enumerate(processed, start=1):
            source, page = _format_source(meta, original)
            context_parts.append(f"--- Document {i} (Source: {source}, Page: {page}) ---\n{content}\n")
        context_str = "\n".join(context_parts)

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