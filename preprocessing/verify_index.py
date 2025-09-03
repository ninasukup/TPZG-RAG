import faiss
import json

def verify_vector_store(index_file="Neu_Knowledgebase/faiss_index.bin", metadata_file="Neu_Knowledgebase/metadata.jsonl"):
    """
    Checks if the FAISS index and metadata file are synchronized and valid.
    """
    print("--- Starting Vector Store Verification ---")
    
    # 1. Check the FAISS Index
    try:
        index = faiss.read_index(index_file)
        num_vectors_in_index = index.ntotal
        print(f"✅ FAISS index ('{index_file}') loaded successfully.")
        print(f"    📊 It contains {num_vectors_in_index} vectors.")
    except Exception as e:
        print(f"❌ ERROR: Could not load or read the FAISS index file '{index_file}'.")
        print(f"   Reason: {e}")
        return

    # 2. Check the Metadata File
    num_lines_in_metadata = 0
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                num_lines_in_metadata += 1
        print(f"✅ Metadata file ('{metadata_file}') read successfully.")
        print(f"    📖 It contains {num_lines_in_metadata} lines/documents.")
    except FileNotFoundError:
        print(f"❌ ERROR: Metadata file not found at '{metadata_file}'.")
        return
        
    # 3. Compare and give a final diagnosis
    print("\n--- Diagnosis ---")
    if num_vectors_in_index == num_lines_in_metadata:
        if num_vectors_in_index > 1:
            print("✅ SUCCESS: Your vector store appears to be correctly built and synchronized.")
            print("   The problem likely lies elsewhere in the retrieval logic if issues persist.")
        else:
            print("❗ ISSUE: Your vector store is synchronized, but it only contains ONE entry.")
            print("   This is the reason you are only retrieving one document.")
            print("   SOLUTION: You must re-run the `embedding_generator.py` on your full `chunks_flat.jsonl` file.")
    else:
        print("❌ CRITICAL ERROR: Mismatch detected!")
        print(f"   The FAISS index has {num_vectors_in_index} vectors, but the metadata file has {num_lines_in_metadata} documents.")
        print("   These two numbers MUST be identical for the RAG pipeline to work.")
        print("   SOLUTION: Delete both '{index_file}' and '{metadata_file}' and re-run the `embedding_generator.py` script.")

if __name__ == "__main__":
    verify_vector_store()
