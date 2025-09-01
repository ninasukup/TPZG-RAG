import json
import argparse
import os
from typing import Dict, List, Any

# --- New Imports ---
# Make sure to install the required libraries:
# pip install sentence-transformers torch
from sentence_transformers import SentenceTransformer
import torch

def load_chunks_from_file(input_file: str) -> Dict[str, List[Dict[str, Any]]]:
    """Loads the chunked data from the JSON file produced by the chunking pipeline."""
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return {}
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file}'. The file might be corrupt or empty.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return {}
    
def main():
    """
    Main function to generate embeddings for pre-chunked text data.
    """
    parser = argparse.ArgumentParser(description="Generate embeddings for chunked proposal documents.")
    parser.add_argument("input_file", type=str, help="Path to the JSON file containing the chunks (e.g., 'chunks_output.json').")
    parser.add_argument("--output_file", type=str, default="embeddings_output.json", help="Path to save the JSON output file with embeddings.")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="The sentence-transformer model to use for generating embeddings.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size for processing chunks, tune based on your VRAM.")
    args = parser.parse_args()

    # 1. Load the chunked data
    proposals_data = load_chunks_from_file(args.input_file)
    if not proposals_data:
        return

    # 2. Initialize the Sentence Transformer model
    # Check for GPU availability and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        print(f"Loading model '{args.model_name}'... This may take a moment.")
        model = SentenceTransformer(args.model_name, device=device)
    except Exception as e:
        print(f"Error loading the model '{args.model_name}'. Please ensure it's a valid sentence-transformer model. Details: {e}")
        return

    # 3. Prepare texts for embedding
    all_chunks_with_info = []
    texts_to_embed = []
    
    # Flatten the structure to process all chunks at once, but keep track of their origin
    for proposal_id, chunks in proposals_data.items():
        for chunk in chunks:
            all_chunks_with_info.append({'proposal_id': proposal_id, 'chunk': chunk})
            # The model works best on the pure text content
            texts_to_embed.append(chunk['content'])

    if not texts_to_embed:
        print("No content found to embed. Exiting.")
        return
        
    print(f"Found a total of {len(texts_to_embed)} chunks to process.")

    # 4. Generate embeddings in batches
    print(f"Generating embeddings in batches of {args.batch_size}...")
    embeddings = model.encode(
        texts_to_embed,
        batch_size=args.batch_size,
        show_progress_bar=True
    )
    print("Embedding generation complete.")

    # 5. Add embeddings back to their corresponding chunks
    for i, chunk_info in enumerate(all_chunks_with_info):
        # The embedding is a NumPy array, convert it to a list for JSON serialization
        chunk_info['chunk']['embedding'] = embeddings[i].tolist()

    # 6. Reconstruct the original nested dictionary structure
    output_data = {}
    for item in all_chunks_with_info:
        proposal_id = item['proposal_id']
        if proposal_id not in output_data:
            output_data[proposal_id] = []
        output_data[proposal_id].append(item['chunk'])

    # 7. Save the final data with embeddings
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Successfully saved embeddings to '{args.output_file}'")
    except Exception as e:
        print(f"\n❌ Error saving output to file: {e}")


if __name__ == "__main__":
    main()