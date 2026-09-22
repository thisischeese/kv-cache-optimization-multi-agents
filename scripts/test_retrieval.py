"""Run a manual retrieval query against Qdrant."""

import argparse

from dotenv import load_dotenv

from kv_eval.rag.retriever import retrieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test RAG retrieval")
    parser.add_argument("--query", required=True)
    parser.add_argument("--tech-id")
    parser.add_argument("--doc-type", action="append", dest="doc_types")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    chunks = retrieve(
        query=args.query,
        tech_id=args.tech_id,
        doc_types=args.doc_types,
        top_k=args.top_k,
    )

    for rank, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.text.split())[:220]
        print(f"{rank}. score={chunk.score:.4f} [{chunk.doc_id} p.{chunk.page}] {chunk.doc_type}")
        print(f"   {preview}")


if __name__ == "__main__":
    main()
