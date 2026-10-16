
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "pages.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

# Valores iniciais, a serem avaliados com perguntas reais.
TARGET_CHARS = 2200
OVERLAP_CHARS = 300
MIN_CHARS = 100


def read_jsonl(path: Path) -> list[dict]:
    """Lê um arquivo JSONL, ignorando linhas vazias."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido na linha {line_number}: {exc}"
                ) from exc

    return records


def split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    """Divide parágrafos longos em partes, preferindo limites de frases."""
    if len(paragraph) <= limit:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts = []
    current = ""

    for sentence in sentences:
        # Trata também frases excepcionalmente longas.
        if len(sentence) > limit:
            if current:
                parts.append(current)
                current = ""

            for start in range(0, len(sentence), limit):
                parts.append(sentence[start:start + limit].strip())
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = sentence

    if current:
        parts.append(current)

    return parts


def split_into_chunks(text: str) -> list[str]:
    """Agrupa parágrafos sem ultrapassar muito o tamanho-alvo."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    units = []
    for paragraph in paragraphs:
        units.extend(split_long_paragraph(paragraph, TARGET_CHARS))

    chunks = []
    current = ""

    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit

        if len(candidate) <= TARGET_CHARS:
            current = candidate
            continue

        if current:
            chunks.append(current)

        # Mantém uma pequena sobreposição textual entre chunks.
        overlap = current[-OVERLAP_CHARS:].strip() if current else ""
        current = f"{overlap}\n\n{unit}".strip() if overlap else unit

        # Se a sobreposição tornou o bloco grande demais, descarta-a.
        if len(current) > TARGET_CHARS:
            current = unit

    if current:
        chunks.append(current)

    return chunks


def build_chunks(pages: list[dict]) -> list[dict]:
    """Cria chunks e mantém metadados de documento, fonte e página."""
    chunks = []

    for page_record in pages:
        text = page_record.get("text", "").strip()

        if not text:
            print(
                "AVISO: página sem texto — "
                f"{page_record.get('document_name')} "
                f"(página {page_record.get('page')})"
            )
            continue

        page_chunks = split_into_chunks(text)

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            if len(chunk_text) < MIN_CHARS:
                print(
                    "AVISO: chunk curto — "
                    f"{page_record.get('document_name')} "
                    f"(página {page_record.get('page')}, "
                    f"chunk {chunk_index}, {len(chunk_text)} caracteres)"
                )

            document_id = page_record.get(
                "document_id",
                Path(page_record["document_name"]).stem,
            )
            page_number = page_record["page"]

            chunks.append({
                "chunk_id": (
                    f"{document_id}:page:{page_number}:chunk:{chunk_index}"
                ),
                "document_id": document_id,
                "document_name": page_record["document_name"],
                "source_type": page_record.get("source_type", "unknown"),
                "page": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "char_count": len(chunk_text),
            })

    return chunks


def main() -> int:
    if not INPUT_FILE.exists():
        print(f"ERRO: arquivo não encontrado: {INPUT_FILE}")
        print("Execute primeiro scripts/extract_pdfs.py.")
        return 1

    try:
        pages = read_jsonl(INPUT_FILE)
        chunks = build_chunks(pages)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERRO ao processar os dados: {exc}")
        return 1

    if not chunks:
        print("Nenhum chunk foi criado. Verifique a extração.")
        return 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    try:
        with temp_file.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )

        temp_file.replace(OUTPUT_FILE)
    except OSError as exc:
        print(f"ERRO ao salvar os chunks: {exc}")
        if temp_file.exists():
            temp_file.unlink()
        return 1

    document_chunks = sum(
        chunk["source_type"] == "document" for chunk in chunks
    )
    transcript_chunks = sum(
        chunk["source_type"] == "transcript" for chunk in chunks
    )

    print("\nChunking concluído!")
    print(f"Páginas lidas: {len(pages)}")
    print(f"Total de chunks: {len(chunks)}")
    print(f"Chunks de documentos: {document_chunks}")
    print(f"Chunks de transcrições: {transcript_chunks}")
    print(
        "Média de caracteres por chunk: "
        f"{sum(c['char_count'] for c in chunks) // len(chunks)}"
    )
    print(f"Resultado: {OUTPUT_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
