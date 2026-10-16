
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "pages.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"

TARGET_CHARS = 2200
OVERLAP_CHARS = 300
MIN_CHUNK_CHARS = 500

# Títulos frequentes na seção estruturada dos documentos estratégicos.
SECTION_LABELS = (
    "CONHECIMENTO",
    "REGRA",
    "PROCESSO",
    "DECISÃO",
    "EXCEÇÃO",
    "ALERTA",
)

def split_long_text(text: str, limit: int = TARGET_CHARS) -> list[str]:
    """Divide textos longos priorizando frases e palavras completas."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    pieces = []
    current = ""

    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                pieces.append(current)
                current = ""

            words = sentence.split()
            part = ""

            for word in words:
                candidate = f"{part} {word}".strip()

                if len(candidate) > limit and part:
                    pieces.append(part)
                    part = word
                else:
                    part = candidate

            if part:
                current = part

        else:
            candidate = f"{current} {sentence}".strip()

            if len(candidate) > limit and current:
                pieces.append(current)
                current = sentence
            else:
                current = candidate

    if current:
        pieces.append(current)

    return pieces


def load_pages() -> list[dict]:
    """Carrega e ordena as páginas extraídas dos PDFs."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_FILE}\n"
            "Execute primeiro scripts/extract_pdfs.py."
        )

    pages = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                page = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido na linha {line_number}: {exc}"
                ) from exc

            pages.append(page)

    return sorted(
        pages,
        key=lambda page: (page["document_id"], page["page"]),
    )


def separate_section_labels(text: str) -> str:
    """Separa rótulos estruturais que vieram colados ao texto anterior."""
    for label in SECTION_LABELS:
        text = re.sub(
            rf"(?<!\n)\s*{label}\s*(?=[●•\-])",
            f"\n\n{label}\n",
            text,
        )

    return text


def is_standalone_heading(text: str) -> bool:
    """Identifica títulos curtos sem pontuação final."""
    clean = text.strip()

    if not clean or len(clean) > 100:
        return False

    if clean.startswith(("●", "•", "-", "1.", "2.", "3.", "4.")):
        return False

    if clean.endswith((".", "!", "?", ";", ":")):
        return False

    # Títulos em caixa alta, inclusive siglas e acentos.
    letters = [char for char in clean if char.isalpha()]

    return bool(letters) and all(
        char.isupper() for char in letters
    )


def make_units(pages: list[dict]) -> list[dict]:
    """
    Cria unidades de texto com página de origem.
    Títulos isolados são anexados ao conteúdo seguinte.
    """
    units = []

    for page in pages:
        text = page.get("text", "").strip()

        if not text:
            continue

        text = separate_section_labels(text)
        paragraphs = re.split(r"\n\s*\n", text)
        page_units = []

        for paragraph in paragraphs:
            paragraph = re.sub(r"\s+", " ", paragraph).strip()

            if not paragraph:
                continue

            for piece in split_long_text(paragraph):
                page_units.append({
                    "document_id": page["document_id"],
                    "document_name": page["document_name"],
                    "source_type": page["source_type"],
                    "page": page["page"],
                    "text": piece,
                })

        # Um título isolado passa a fazer parte do próximo trecho.
        merged = []
        pending_heading = None

        for unit in page_units:
            if is_standalone_heading(unit["text"]):
                if pending_heading:
                    merged.append({
                        **unit,
                        "text": pending_heading,
                    })

                pending_heading = unit["text"]
                continue

            if pending_heading:
                unit = {
                    **unit,
                    "text": f"{pending_heading}\n{unit['text']}",
                }
                pending_heading = None

            merged.append(unit)

        # Se um título estiver no fim da página, preserva-o.
        if pending_heading:
            if merged:
                merged[-1]["text"] += f"\n{pending_heading}"
            else:
                merged.append({
                    "document_id": page["document_id"],
                    "document_name": page["document_name"],
                    "source_type": page["source_type"],
                    "page": page["page"],
                    "text": pending_heading,
                })

        units.extend(merged)

    return units


def create_chunks(units: list[dict]) -> list[dict]:
    """Agrupa unidades, preservando contexto e páginas de origem."""
    chunks = []
    document_units = []
    current_document = None

    def process_document(doc_units: list[dict]) -> None:
        if not doc_units:
            return

        # Primeiro montamos grupos sem sobreposição.
        groups = []
        start = 0

        while start < len(doc_units):
            end = start
            char_count = 0

            while end < len(doc_units):
                addition = len(doc_units[end]["text"])
                if end > start:
                    addition += 2

                if end > start and char_count + addition > (
                    TARGET_CHARS - OVERLAP_CHARS
                ):
                    break

                char_count += addition
                end += 1

            groups.append((start, end))
            start = end

        # Agrupa o último trecho curto com o anterior quando couber.
        if len(groups) > 1:
            previous_start, previous_end = groups[-2]
            last_start, last_end = groups[-1]

            combined_size = sum(
                len(unit["text"])
                for unit in doc_units[previous_start:last_end]
            ) + 2 * (last_end - previous_start - 1)

            if (
                combined_size <= TARGET_CHARS - OVERLAP_CHARS
                and last_end - last_start > 0
                and sum(
                    len(unit["text"])
                    for unit in doc_units[last_start:last_end]
                ) < MIN_CHUNK_CHARS
            ):
                groups[-2:] = [(previous_start, last_end)]

        # Adiciona sobreposição com unidades completas.
        for chunk_index, (group_start, group_end) in enumerate(
            groups, start=1
        ):
            start = group_start

            if chunk_index > 1:
                overlap_size = 0
                candidate_start = group_start

                for index in range(group_start - 1, -1, -1):
                    # Não ultrapassa o grupo anterior.
                    if index < groups[chunk_index - 2][0]:
                        break

                    unit_size = len(doc_units[index]["text"]) + 2

                    if overlap_size + unit_size > OVERLAP_CHARS:
                        break

                    overlap_size += unit_size
                    candidate_start = index

                start = candidate_start

            selected = doc_units[start:group_end]
            text = "\n\n".join(unit["text"] for unit in selected)
            pages = sorted({unit["page"] for unit in selected})
            first = selected[0]

            chunks.append({
                "chunk_id": (
                    f"{first['document_id']}:chunk:{chunk_index:04d}"
                ),
                "document_id": first["document_id"],
                "document_name": first["document_name"],
                "source_type": first["source_type"],
                "chunk_index": chunk_index,
                "page": pages[0],
                "page_start": pages[0],
                "page_end": pages[-1],
                "pages": pages,
                "text": text,
                "char_count": len(text),
            })

    for unit in units:
        document_id = unit["document_id"]

        if current_document is not None and document_id != current_document:
            process_document(document_units)
            document_units = []

        current_document = document_id
        document_units.append(unit)

    process_document(document_units)
    return chunks


def main() -> int:
    try:
        pages = load_pages()
        units = make_units(pages)
        chunks = create_chunks(units)

        if not chunks:
            print("Nenhum chunk foi criado. Verifique os textos extraídos.")
            return 1

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        temp_file = OUTPUT_DIR / "chunks.tmp"

        try:
            with temp_file.open("w", encoding="utf-8") as file:
                for chunk in chunks:
                    file.write(
                        json.dumps(chunk, ensure_ascii=False) + "\n"
                    )

            temp_file.replace(OUTPUT_FILE)

        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise

        counts = Counter(chunk["source_type"] for chunk in chunks)
        sizes = [chunk["char_count"] for chunk in chunks]
        short_chunks = sum(
            size < MIN_CHUNK_CHARS for size in sizes
        )

        print("\nChunking concluído!")
        print(f"Páginas de entrada: {len(pages)}")
        print(f"Unidades de texto: {len(units)}")
        print(f"Chunks gerados: {len(chunks)}")
        print(f"Chunks de documentos: {counts['document']}")
        print(f"Chunks de transcrições: {counts['transcript']}")
        print(f"Chunks abaixo de {MIN_CHUNK_CHARS} caracteres: {short_chunks}")
        print(f"Tamanho mínimo: {min(sizes)} caracteres")
        print(f"Tamanho médio: {sum(sizes) // len(sizes)} caracteres")
        print(f"Tamanho máximo: {max(sizes)} caracteres")
        print(f"Arquivo gerado: {OUTPUT_FILE}")

        return 0

    except Exception as exc:
        print(f"ERRO: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
