import faiss
import json
import logging
import yaml
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
from typing import Dict, Any, Tuple, List


def load_vector_db(index_file, metadata_file):
    try:
        print("Loading FAISS index...")
        index = faiss.read_index(str(index_file))

        metadata = []
        # Read JSONL in UTF-8 (BOM-tolerant) so weird bytes don't blow up on Windows
        with open(str(metadata_file), "r", encoding="utf-8-sig") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    metadata.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Bad JSON on line {lineno}: {e.msg}") from e

        logging.info("Data Loaded From Vector DB Successfully (FAISS and JSONL)!")
        return index, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load vector DB: {e}")

    
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read Yaml file and return ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"Yaml file: {path_to_yaml} loaded successfully!")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e
    

def _format_source(doc: Dict[str, Any]) -> Tuple[str, str]:
    """Extract source and page information from document."""
    # Get metadata if it exists
    meta = doc.get("metadata", {})
    
    # Try multiple possible keys for source
    source = (
        doc.get("source_filename") 
        or doc.get("source") 
        or meta.get("source_filename")
        or meta.get("source")
        or "Unknown"
    )
    
    # Try multiple possible keys for page
    page = (
        doc.get("page")
        or doc.get("page_number")
        or meta.get("page")
        or meta.get("page_number")
        or (meta.get("page_numbers", ["N/A"]) or ["N/A"])[0]
        or "N/A"
    )
    
    return str(source), str(page)
    

def extract_content(reranked_docs: List[Dict[str, Any]]) -> str:
    """Extract and format content from reranked documents."""
    processed_docs = []
    
    for i, doc in enumerate(reranked_docs):
        # Try multiple possible content keys
        content = (
            doc.get("content")
            or doc.get("text") 
            or doc.get("chunk")
            or doc.get("page_content")
            or ""
        )
        
        # If still no content, check metadata
        if not content and "metadata" in doc:
            metadata_obj = doc["metadata"]
            content = (
                metadata_obj.get("content")
                or metadata_obj.get("text")
                or metadata_obj.get("chunk")
                or metadata_obj.get("page_content")
                or ""
            )
        
        content = str(content).strip()
        
        # Only keep documents with meaningful content
        if len(content) >= 10:  # Minimum content length
            processed_docs.append({
                "content": content,
                "source_info": doc,
                "index": i + 1
            })

    print(f"Processed {len(processed_docs)} documents with valid content.")

    if not processed_docs:
        print("WARNING: No documents with valid content found!")
        # Fallback: use original docs even if content is minimal
        for i, doc in enumerate(reranked_docs[:4]):
            content = str(doc.get("content", doc.get("text", "No content available")))
            processed_docs.append({
                "content": content,
                "source_info": doc,
                "index": i + 1
            })

    # Build context string
    context_parts = []
    for doc_info in processed_docs:
        source, page = _format_source(doc_info["source_info"])
        content = doc_info["content"]
        index = doc_info["index"]
        
        context_parts.append(
            f"--- Document {index} (Source: {source}, Page: {page}) ---\n{content}\n"
        )
    
    context_str = "\n".join(context_parts)
    return context_str