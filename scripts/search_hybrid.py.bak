
import json
import re
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# --------------------------------------------------
# Configuração
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.jsonl"
CHROMA_PATH = ROOT_DIR / "data" / "index" / "chroma"

COLLECTION_NAME = "kera_knowledge"

# Modelo para busca semântica
MODEL_NAME = "BAAI/bge-m3"

# Modelo para reordenar os resultados
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

# Quantidade de resultados exibidos no final
TOP_K = 5

# Quantidade de candidatos recuperados por método
CANDIDATE_K = 10

# Quantidade de candidatos avaliados pelo reranker
RERANK_CANDIDATES = 15

# Parâmetro da fusão Reciprocal Rank Fusion
RRF_K = 60


# --------------------------------------------------
# Funções auxiliares
# --------------------------------------------------

def tokenize(text):
    """Normaliza o texto e separa palavras para o BM25."""
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def load_chunks():
    """Carrega os chunks usados como corpus da busca BM25."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de chunks não encontrado: {CHUNKS_PATH}"
        )

    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido em {CHUNKS_PATH}, "
                    f"linha {line_number}: {exc}"
                ) from exc

    if not chunks:
        raise ValueError("O arquivo de chunks está vazio.")

    return chunks


def get_chunk_id(chunk):
    """Obtém o identificador estável de um chunk."""
    for field in ("chunk_id", "id"):
        if chunk.get(field):
            return str(chunk[field])

    document_id = chunk.get("document_id")
    chunk_index = chunk.get("chunk_index")

    if document_id is not None and chunk_index is not None:
        return f"{document_id}:chunk:{int(chunk_index):04d}"

    raise ValueError(
        "Um chunk não possui chunk_id, id ou os campos "
        "document_id e chunk_index."
    )


def get_chunk_text(chunk):
    """Obtém o texto principal do chunk."""
    text = chunk.get("text")

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"Chunk sem texto válido: {get_chunk_id(chunk)}"
        )

    return text


def get_metadata(chunk):
    """Padroniza os metadados dos chunks."""
    return {
        "document_name": chunk.get(
            "document_name", "Documento desconhecido"
        ),
        "source_type": chunk.get("source_type", "unknown"),
        "pages": chunk.get(
            "pages",
            chunk.get("page", "Não informado"),
        ),
    }


# --------------------------------------------------
# Busca BM25
# --------------------------------------------------

def search_bm25(question, chunks, bm25, top_k=CANDIDATE_K):
    """Recupera chunks com correspondência lexical."""
    query_tokens = tokenize(question)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices:
        # Ignora chunks sem correspondência lexical.
        if scores[index] <= 0:
            continue

        chunk = chunks[index]
        chunk_id = get_chunk_id(chunk)

        results.append(
            (
                chunk_id,
                {
                    "text": get_chunk_text(chunk),
                    "metadata": get_metadata(chunk),
                    "bm25_score": float(scores[index]),
                },
            )
        )

        if len(results) >= top_k:
            break

    return results


# --------------------------------------------------
# Busca vetorial
# --------------------------------------------------

def search_vector(question, model, collection, top_k=CANDIDATE_K):
    """Recupera chunks semanticamente semelhantes à pergunta."""
    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    results = []

    ids = response["ids"][0]
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    for chunk_id, text, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        results.append(
            (
                str(chunk_id),
                {
                    "text": text,
                    "metadata": metadata or {},
                    "vector_distance": float(distance),
                },
            )
        )

    return results


# --------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# --------------------------------------------------

def reciprocal_rank_fusion(rankings, k=RRF_K):
    """
    Combina rankings de diferentes métodos.

    Cada ranking tem o formato:
    (nome_do_metodo, resultados).

    A pontuação RRF combina as posições dos resultados,
    sem comparar diretamente as escalas dos métodos.
    """
    fused = {}

    for method_name, ranking in rankings:
        for rank, (chunk_id, result) in enumerate(
            ranking, start=1
        ):
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "score": 0.0,
                    "result": result,
                    "ranks": {},
                }

            fused[chunk_id]["score"] += 1.0 / (k + rank)
            fused[chunk_id]["ranks"][method_name] = rank

    return sorted(
        fused.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )


# --------------------------------------------------
# Reranking
# --------------------------------------------------

def rerank_results(question, fused_results, reranker):
    """
    Reavalia os melhores candidatos do RRF com um CrossEncoder.

    Retorna os candidatos ordenados pela pontuação do reranker.
    """
    candidates = fused_results[:RERANK_CANDIDATES]

    if not candidates:
        return []

    pairs = [
        (question, item["result"]["text"])
        for _, item in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked_results = []

    for (chunk_id, item), score in zip(candidates, scores):
        reranked_results.append(
            (
                chunk_id,
                {
                    **item,
                    "rerank_score": float(score),
                },
            )
        )

    reranked_results.sort(
        key=lambda item: item[1]["rerank_score"],
        reverse=True,
    )

    return reranked_results


# --------------------------------------------------
# Exibição dos resultados
# --------------------------------------------------

def print_results(results):
    """Exibe os resultados finais com pontuações e procedência."""
    if not results:
        print("\nNenhum resultado relevante foi recuperado.")
        return

    print(f"\nMelhores resultados após reranking: {min(TOP_K, len(results))}")

    for position, (chunk_id, item) in enumerate(
        results[:TOP_K], start=1
    ):
        result = item["result"]
        metadata = result.get("metadata", {})
        ranks = item.get("ranks", {})

        print("\n" + "=" * 80)
        print(f"Resultado {position}")
        print(f"Pontuação reranker: {item['rerank_score']:.4f}")
        print(f"Pontuação RRF: {item['score']:.6f}")

        print(
            "Posição vetorial: "
            f"{ranks.get('vetorial', 'não recuperado')}"
        )
        print(
            "Posição BM25: "
            f"{ranks.get('BM25', 'não recuperado')}"
        )

        print(f"Chunk: {chunk_id}")
        print(
            "Fonte: "
            f"{metadata.get('document_name', 'Não informada')}"
        )
        print(
            "Tipo de fonte: "
            f"{metadata.get('source_type', 'Não informado')}"
        )

        pages = metadata.get(
            "pages",
            metadata.get("page_start", "Não informado"),
        )
        print(f"Páginas: {pages}")

        print("-" * 80)
        print(result["text"])

    print("\n" + "=" * 80)


# --------------------------------------------------
# Preparação reutilizável dos recursos de busca
# --------------------------------------------------

def load_search_resources():
    """Carrega uma única vez os recursos usados em várias consultas."""
    print("Carregando chunks para BM25...")
    chunks = load_chunks()

    corpus = [
        tokenize(get_chunk_text(chunk))
        for chunk in chunks
    ]
    bm25 = BM25Okapi(corpus)

    print("Conectando ao Chroma...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Não foi possível abrir a coleção '{COLLECTION_NAME}'. "
            "Verifique se o índice Chroma foi criado."
        ) from exc

    collection_size = collection.count()
    if collection_size == 0:
        raise ValueError("A coleção Chroma está vazia.")

    print("Carregando modelo de embeddings...")
    model = SentenceTransformer(MODEL_NAME)

    print("Carregando reranker...")
    reranker = CrossEncoder(RERANKER_NAME)

    return {
        "chunks": chunks,
        "bm25": bm25,
        "client": client,
        "collection": collection,
        "model": model,
        "reranker": reranker,
        "collection_size": collection_size,
    }


def search(question, resources, top_k=TOP_K, verbose=False):
    """
    Executa busca vetorial + BM25 + RRF + reranking.

    Retorna lista de tuplas (chunk_id, item), mantendo o formato
    esperado por print_results(). Os recursos são carregados à parte
    para que várias perguntas reutilizem os mesmos modelos.
    """
    question = question.strip()
    if not question:
        return []

    vector_results = search_vector(
        question,
        resources["model"],
        resources["collection"],
        top_k=CANDIDATE_K,
    )

    bm25_results = search_bm25(
        question,
        resources["chunks"],
        resources["bm25"],
        top_k=CANDIDATE_K,
    )

    fused_results = reciprocal_rank_fusion(
        [
            ("vetorial", vector_results),
            ("BM25", bm25_results),
        ]
    )

    results = rerank_results(
        question,
        fused_results,
        resources["reranker"],
    )

    if verbose:
        print(f"Pergunta: {question}")
        print(f"Chunks no corpus BM25: {len(resources['chunks'])}")
        print(f"Chunks no Chroma: {resources['collection_size']}")
        print(
            "Candidatos recuperados: "
            f"vetorial={len(vector_results)}, "
            f"BM25={len(bm25_results)}"
        )
        print(f"Candidatos únicos após RRF: {len(fused_results)}")
        print_results(results[:top_k])

    return results[:top_k]


# --------------------------------------------------
# Execução principal
# --------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(
            'Uso: python scripts/search_hybrid.py '
            '"Sua pergunta aqui"'
        )
        return

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Informe uma pergunta.")
        return

    resources = load_search_resources()
    search(question, resources, top_k=TOP_K, verbose=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecução interrompida.")
    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        sys.exit(1)
