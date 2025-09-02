# flatten_to_jsonl.py
import json

src = "Neu_Knowledgebase/chunks_output.json"
dst = "Neu_Knowledgebase/chunks_flat.jsonl"

with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(dst, "w", encoding="utf-8") as out:
    for doc_title, chunks in data.items():
        for ch in chunks:
            meta = ch.get("metadata", {}).copy()
            pn = meta.get("page_numbers") or []
            if "page" not in meta and isinstance(pn, list) and pn:
                meta["page"] = pn[0]
            meta.setdefault("document_id", doc_title)
            rec = {"content": ch.get("content", ""), "metadata": meta}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Done:", dst)
