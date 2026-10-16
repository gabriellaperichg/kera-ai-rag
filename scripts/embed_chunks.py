
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "index" / "chroma"

MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "kera_knowledge"
BATCH_SIZE = 8


def load_chunks() -> list[dict]:
    """Carrega os chunks validados."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_FILE}\n"
            "Execute primeiro scripts/chunk_pages.py."
        )

    chunks = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido na linha {line_number}: {exc}"
                ) from exc

            if not chunk.get("chunk_id") or not chunk.get("text", "").strip():
                raise ValueError(
                    f"Chunk sem ID ou texto na linha {line_number}."
                )

            chunks.append(chunk)

    if not chunks:
        raise ValueError("Nenhum chunk encontrado.")

    ids = [chunk["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Existem chunk_ids duplicados.")

    return chunks


def prepare_metadata(chunk: dict) -> dict:
    """Converte os metadados para formatos aceitos pelo ChromaDB."""
    pages = chunk.get("pages", [chunk.get("page")])
    pages = [page for page in pages if page is not None]

    return {
        "document_id": str(chunk["document_id"]),
        "document_name": str(chunk["document_name"]),
        "source_type": str(chunk["source_type"]),
        "chunk_index": int(chunk["chunk_index"]),
        "page_start": int(chunk["page_start"]),
        "page_end": int(chunk["page_end"]),
        "pages": ",".join(str(page) for page in pages),
    }


def main() -> int:
    try:
        chunks = load_chunks()

        print(f"Chunks carregados: {len(chunks)}")
        print(f"Modelo: {MODEL_NAME}")
        print("Carregando modelo de embeddings...")

        model = SentenceTransformer(MODEL_NAME)

        print("Modelo carregado.")
        print("Preparando índice vetorial local...")

        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(INDEX_DIR))

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": MODEL_NAME,
            },
        )

        # O script reconstrói o índice com base no arquivo atual.
        # Isso evita registros antigos ou duplicados após alterações
        # nos chunks.
        existing = collection.get(include=[])
        existing_ids = existing["ids"]

        for start in range(0, len(existing_ids), 500):
            collection.delete(ids=existing_ids[start:start + 500])

        total = len(chunks)

        for start in range(0, total, BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]

            texts = [chunk["text"] for chunk in batch]
            ids = [chunk["chunk_id"] for chunk in batch]
            metadatas = [prepare_metadata(chunk) for chunk in batch]

            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings.tolist(),
            )

            processed = min(start + len(batch), total)
            print(f"Processados: {processed}/{total} chunks")

        print("\nEmbeddings gerados e armazenados!")
        print(f"Modelo: {MODEL_NAME}")
        print(f"Dimensão dos vetores: {model.get_sentence_embedding_dimension()}")
        print(f"Registros no índice: {collection.count()}")
        print(f"Coleção: {COLLECTION_NAME}")
        print(f"Índice local: {INDEX_DIR}")

        return 0

    except Exception as exc:
        print(f"\nERRO: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
