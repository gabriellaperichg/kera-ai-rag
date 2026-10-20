
import json
from pathlib import Path

REPORT = Path("data/evaluation/retrieval_report.jsonl")


def main():
    with REPORT.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    for record in records:
        print("\n" + "=" * 90)
        print(f"{record['id']}: {record['question']}")
        print(f"Respondível pelas fontes: {record.get('answerable_from_sources')}")

        for rank, result in enumerate(
            record.get("retrieved_results", [])[:5],
            start=1,
        ):
            metadata = result.get("metadata", {})
            text = " ".join(result.get("text", "").split())

            print(f"\n  {rank}. {result['chunk_id']}")
            print(
                f"     Fonte: {metadata.get('document_name', '')}"
                f" | Páginas: {metadata.get('pages', '')}"
            )
            print(f"     Trecho: {text[:350]}")

    print("\n" + "=" * 90)
    print(f"Total de perguntas inspecionadas: {len(records)}")


if __name__ == "__main__":
    main()
