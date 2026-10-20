import json
from pathlib import Path

REPORT = Path("data/evaluation/retrieval_report.jsonl")
LABELS = Path("data/evaluation/relevance_labels.json")
K = 5


def main():
    with REPORT.open(encoding="utf-8") as f:
        records = {
            row["id"]: row
            for line in f
            if line.strip()
            for row in [json.loads(line)]
        }

    with LABELS.open(encoding="utf-8") as f:
        labels = json.load(f)

    recalls = []
    reciprocal_ranks = []

    for question_id, relevant_ids in labels.items():
        record = records.get(question_id)

        if record is None:
            print(f"{question_id}: não encontrada no relatório")
            continue

        results = record.get("retrieved_results", [])[:K]
        ranked_ids = [item.get("chunk_id") for item in results]

        relevant = set(relevant_ids)
        hits = sum(chunk_id in relevant for chunk_id in ranked_ids)
        recall = hits / len(relevant) if relevant else 0.0

        first_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(ranked_ids, start=1)
                if chunk_id in relevant
            ),
            None,
        )
        reciprocal_rank = 1 / first_rank if first_rank else 0.0

        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

        print(
            f"{question_id} | Recall@{K}: {recall:.3f} "
            f"| MRR: {reciprocal_rank:.3f} "
            f"| primeiro relevante: {first_rank or 'não encontrado'}"
        )

    if recalls:
        print("\n=== MÉDIAS DO CONJUNTO ROTULADO ===")
        print(f"Recall@{K}: {sum(recalls) / len(recalls):.3f}")
        print(f"MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}")
        print(f"Perguntas avaliadas: {len(recalls)}")


if __name__ == "__main__":
    main()