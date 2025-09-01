# verify_index.py
import faiss, json, itertools
from pathlib import Path

INDEX = "faiss_index.bin"
META  = "metadata.jsonl"

# Load FAISS
index = faiss.read_index(INDEX)
print("Vectors in index:", index.ntotal)

# Count metadata lines
with open(META, "r", encoding="utf-8") as f:
    meta_lines = sum(1 for _ in f)
print("Metadata rows:", meta_lines)

# Peek first 3 metadata rows
print("\nSample metadata rows:")
with open(META, "r", encoding="utf-8") as f:
    for line in itertools.islice(f, 3):
        print(json.loads(line))