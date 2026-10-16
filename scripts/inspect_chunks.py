
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

if not INPUT_FILE.exists():
    raise SystemExit("Arquivo não encontrado. Execute chunk_pages.py primeiro.")

with INPUT_FILE.open("r", encoding="utf-8") as file:
    chunks = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

print(f"Total de chunks: {len(chunks)}")

for chunk in chunks[:10]:
    print("\n" + "=" * 70)
    print(f"ID: {chunk['chunk_id']}")
    print(f"Tipo: {chunk['source_type']}")
    print(f"Documento: {chunk['document_name']}")
    print(f"Página: {chunk['page']}")
    print(f"Caracteres: {chunk['char_count']}")
    print("-" * 70)
    print(chunk["text"])
