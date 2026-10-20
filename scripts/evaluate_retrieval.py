import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.search_hybrid import load_search_resources, search


EVALUATION_FILE = ROOT_DIR / "data" / "evaluation" / "retrieval_questions.jsonl"
REPORT_FILE = ROOT_DIR / "data" / "evaluation" / "retrieval_report.jsonl"


def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de avaliação não encontrado: {path}\n"
            "Coloque retrieval_questions.jsonl em data/evaluation/."
        )

    with path.open("r", encoding="utf-8") as file:
        questions = [json.loads(line) for line in file if line.strip()]

    if not questions:
        raise ValueError(f"Nenhuma pergunta encontrada em {path}.")

    return questions


def normalize_name(value) -> str:
    """Normaliza nomes de documentos para comparação aproximada."""
    return str(value or "").replace("\\", "/").split("/")[-1].strip().casefold()


def result_metadata(result_pair: tuple) -> tuple[str, set[int], dict]:
    """Extrai nome da fonte, páginas e metadados de um resultado reranqueado."""
    chunk_id, item = result_pair
    result = item.get("result", {})
    metadata = result.get("metadata", {}) or {}

    document = metadata.get("document_name") or metadata.get("document_id") or ""
    pages_value = metadata.get("pages", metadata.get("page", ""))

    if isinstance(pages_value, list):
        page_values = pages_value
    else:
        page_values = str(pages_value).replace(";", ",").split(",")

    pages = set()
    for value in page_values:
        try:
            pages.add(int(str(value).strip()))
        except (TypeError, ValueError):
            pass

    for key in ("page", "page_start", "page_end"):
        try:
            if metadata.get(key) is not None:
                pages.add(int(metadata[key]))
        except (TypeError, ValueError):
            pass

    return normalize_name(document), pages, {
        "chunk_id": chunk_id,
        "text": result.get("text", ""),
        "metadata": metadata,
        "rerank_score": item.get("rerank_score"),
        "rrf_score": item.get("score"),
        "ranks": item.get("ranks", {}),
    }


def check_expected_sources(expected_sources: list[dict], results: list[tuple]) -> dict:
    checks = []

    for expected in expected_sources or []:
        # O conjunto de avaliação pode usar "document" ou "document_name".
        expected_name = normalize_name(
            expected.get("document")
            or expected.get("document_name")
            or expected.get("source")
            or ""
        )
        expected_page = expected.get("page", expected.get("pages"))
        if isinstance(expected_page, list):
            expected_pages = {int(p) for p in expected_page if str(p).isdigit()}
        elif expected_page is None or str(expected_page).strip() == "":
            expected_pages = set()
        else:
            try:
                expected_pages = {int(expected_page)}
            except (TypeError, ValueError):
                expected_pages = set()

        matched_result = None
        for result_pair in results:
            found_name, found_pages, compact = result_metadata(result_pair)

            same_document = (
                not expected_name
                or expected_name in found_name
                or found_name in expected_name
            )
            same_page = not expected_pages or bool(expected_pages & found_pages)

            if same_document and same_page:
                matched_result = compact
                break

        checks.append({
            "expected_document": expected.get("document")
                or expected.get("document_name")
                or expected.get("source"),
            "expected_page": expected_page,
            "found": matched_result is not None,
            "matched_chunk_id": (
                matched_result["chunk_id"] if matched_result else None
            ),
        })

    total = len(checks)
    covered = sum(check["found"] for check in checks)
    return {
        "source_page_checks": checks,
        "source_page_coverage": covered / total if total else None,
        "covered_expected_sources": covered,
        "total_expected_sources": total,
    }


def main():
    questions = load_questions(EVALUATION_FILE)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Perguntas carregadas: {len(questions)}")
    print("Preparando recursos de busca (modelos carregados uma vez)...")
    resources = load_search_resources()

    with REPORT_FILE.open("w", encoding="utf-8") as report:
        for index, question_item in enumerate(questions, start=1):
            question = question_item["question"]
            results = search(question, resources, top_k=5)

            coverage = check_expected_sources(
                question_item.get("expected_sources", []),
                results,
            )

            score = coverage["source_page_coverage"]
            report_item = {
                "id": question_item.get("id", f"Q{index:02d}"),
                "question": question,
                "answerable_from_sources": question_item.get(
                    "answerable_from_sources"
                ),
                **coverage,
                "retrieved_results": [
                    result_metadata(pair)[2] for pair in results
                ],
            }

            report.write(
                json.dumps(report_item, ensure_ascii=False, default=str) + "\n"
            )

            score_label = f"{score:.0%}" if score is not None else "N/A"
            print(
                f'{report_item["id"]} | cobertura das fontes: {score_label} | '
                f"{question}"
            )

    print(f"\nRelatório salvo em: {REPORT_FILE}")
    print(
        "\nObservação: cobertura de fonte/página não é Recall@5 nem MRR. "
        "Essas métricas exigem a identificação dos chunks relevantes "
        "para cada pergunta."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAvaliação interrompida.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        sys.exit(1)
