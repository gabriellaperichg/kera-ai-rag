
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "processed" / "pages.jsonl"

if not input_file.exists():
    raise SystemExit("Arquivo não encontrado. Execute extract_pdfs.py primeiro.")

with input_file.open("r", encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]

print(f"Total de páginas: {len(records)}")

for record in records:
    print("\n" + "=" * 70)
    print(f"Documento: {record['document_name']}")
    print(f"Página: {record['page']}")
    print(f"Caracteres: {record['char_count']}")
    print(f"Revisar: {record['needs_review']}")
    print("-" * 70)
    print(record["text"][:1200] or "[SEM TEXTO EXTRAÍDO]")
