"""Run a small Recall@K / MRR retrieval evaluation against Qdrant."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from kv_eval.rag.evaluation import RetrievalEvalCase, RetrievalEvalResult, mean, score_retrieval_case
from kv_eval.rag.retriever import retrieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qdrant retrieval with simple relevance labels"
    )
    parser.add_argument("--eval-file", type=Path, default=Path("data/retrieval_eval.example.json"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_cases(path: Path) -> list[RetrievalEvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [RetrievalEvalCase.model_validate(item) for item in data["cases"]]


def print_case(result: RetrievalEvalResult, top_k: int) -> None:
    rank = result.first_relevant_rank if result.first_relevant_rank is not None else "-"
    print(
        f"{result.name}: recall@{top_k}={result.recall_at_k:.3f} "
        f"rr={result.reciprocal_rank:.3f} first_rank={rank}"
    )
    if result.page_recall_at_k is not None:
        page_rank = (
            result.first_relevant_page_rank
            if result.first_relevant_page_rank is not None
            else "-"
        )
        print(
            f"  page: hit={'Y' if result.page_hit_at_k else 'N'} "
            f"recall@{top_k}={result.page_recall_at_k:.3f} "
            f"rr={result.page_reciprocal_rank:.3f} first_rank={page_rank}"
        )
    print(f"  returned_doc_ids: {', '.join(result.returned_doc_ids)}")
    print(f"  returned_pages: {', '.join(str(page) for page in result.returned_pages)}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    cases = load_cases(args.eval_file)

    results: list[RetrievalEvalResult] = []
    for case in cases:
        chunks = retrieve(
            query=case.query,
            tech_id=case.tech_id,
            doc_types=case.doc_types,
            top_k=args.top_k,
        )
        result = score_retrieval_case(case, chunks)
        results.append(result)
        print_case(result, args.top_k)

    print()
    print(f"Cases: {len(results)}")
    print(f"Recall@{args.top_k}: {mean([result.recall_at_k for result in results]):.3f}")
    print(f"MRR: {mean([result.reciprocal_rank for result in results]):.3f}")

    page_scored = [result for result in results if result.page_recall_at_k is not None]
    if page_scored:
        print(f"Page-labelled cases: {len(page_scored)}")
        print(
            f"Page Hit@{args.top_k}: "
            f"{mean([float(bool(r.page_hit_at_k)) for r in page_scored]):.3f}"
        )
        print(
            f"Page Recall@{args.top_k}: "
            f"{mean([r.page_recall_at_k or 0.0 for r in page_scored]):.3f}"
        )
        print(f"Page MRR: {mean([r.page_reciprocal_rank or 0.0 for r in page_scored]):.3f}")


if __name__ == "__main__":
    main()
