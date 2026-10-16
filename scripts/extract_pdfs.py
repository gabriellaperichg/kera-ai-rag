
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


# Raiz do projeto: pasta imediatamente acima de scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "raw" / "documents"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "raw" / "transcripts"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "pages.jsonl"


def normalize_text(text: str) -> str:
    """Normaliza espaços sem apagar a separação entre parágrafos."""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_source_type(pdf_path: Path) -> str:
    """Identifica se o PDF veio da pasta de documentos ou transcrições."""
    if TRANSCRIPTS_DIR in pdf_path.parents:
        return "transcript"

    return "document"


def extract_pdf(pdf_path: Path) -> list[dict]:
    """Extrai texto página por página e preserva os metadados da origem."""
    records = []
    source_type = get_source_type(pdf_path)

    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = normalize_text(
                page.get_text("text", sort=True)
            )

            records.append({
                "document_id": f"{source_type}:{pdf_path.stem}",
                "document_name": pdf_path.name,
                "source_type": source_type,
                "page": page_index + 1,
                "text": text,
                "char_count": len(text),
                "needs_review": len(text) < 80,
            })

    return records


def main() -> int:
    source_dirs = [DOCUMENTS_DIR, TRANSCRIPTS_DIR]

    existing_dirs = [
        folder for folder in source_dirs if folder.is_dir()
    ]

    if not existing_dirs:
        print(
            "ERRO: nenhuma pasta de origem foi encontrada.\n"
            f"Verifique:\n- {DOCUMENTS_DIR}\n- {TRANSCRIPTS_DIR}"
        )
        return 1

    pdf_files = sorted(
        (
            path
            for folder in existing_dirs
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: str(path).lower(),
    )

    if not pdf_files:
        print(
            "Nenhum PDF encontrado nas pastas documents e transcripts."
        )
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    failed_files = []

    for pdf_path in pdf_files:
        try:
            records = extract_pdf(pdf_path)
            all_records.extend(records)

            flagged = sum(
                record["needs_review"] for record in records
            )
            source_type = get_source_type(pdf_path)

            print(
                f"OK: [{source_type}] {pdf_path.name} — "
                f"{len(records)} páginas, "
                f"{flagged} para revisar"
            )

        except Exception as exc:
            failed_files.append((pdf_path, str(exc)))
            print(f"ERRO ao ler {pdf_path}: {exc}")

    # Evita substituir o resultado anterior por uma extração parcial.
    if failed_files:
        print(
            "\nExtração incompleta. "
            "O arquivo final não foi atualizado."
        )
        return 1

    # Grava primeiro em um arquivo temporário.
    # Só substitui o resultado anterior depois de concluir a gravação.
    temp_file = OUTPUT_DIR / "pages.tmp"

    try:
        with temp_file.open("w", encoding="utf-8") as output:
            for record in all_records:
                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        temp_file.replace(OUTPUT_FILE)

    except Exception as exc:
        print(f"ERRO ao salvar o resultado: {exc}")
        if temp_file.exists():
            temp_file.unlink()
        return 1

    document_pages = sum(
        record["source_type"] == "document"
        for record in all_records
    )
    transcript_pages = sum(
        record["source_type"] == "transcript"
        for record in all_records
    )
    review_count = sum(
        record["needs_review"] for record in all_records
    )

    print("\nExtração concluída!")
    print(f"PDFs processados: {len(pdf_files)}")
    print(f"Páginas de documentos: {document_pages}")
    print(f"Páginas de transcrições: {transcript_pages}")
    print(f"Páginas sinalizadas para revisão: {review_count}")
    print(f"Resultado: {OUTPUT_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
