
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "index" / "chroma"

MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "kera_knowledge"
TOP_K = 5


def main() -> int:
    if len(sys.argv) < 2:
        print(
            'Uso: python scripts/search_chunks.py '
            '"Sua pergunta aqui"'
        )
        return 1

    question = " ".join(sys.argv[1:]).strip()

    if not question:
        print("Informe uma pergunta.")
        return 1

    try:
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        collection = client.get_collection(COLLECTION_NAME)

        print("Carregando modelo de embeddings...")
        model = SentenceTransformer(MODEL_NAME)

        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(TOP_K, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        print(f"\nPergunta: {question}")
        print(f"Resultados: {len(results['ids'][0])}")

        for rank, chunk_id in enumerate(results["ids"][0], start=1):
            metadata = results["metadatas"][0][rank - 1]
            document = results["documents"][0][rank - 1]
            distance = results["distances"][0][rank - 1]

            print("\n" + "=" * 70)
            print(f"Resultado {rank} | Distância: {distance:.4f}")
            print(f"Fonte: {metadata['document_name']}")
            print(f"Tipo: {metadata['source_type']}")
            print(f"Páginas: {metadata['pages']}")
            print(f"Chunk: {chunk_id}")
            print("-" * 70)
            print(document)

        return 0

    except Exception as exc:
        print(f"ERRO: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
