"""Interactive side-by-side retrieval comparison between two Qdrant collections.

The embedding model loads once, so follow-up questions answer immediately.

    uv run python scripts/compare_retrieval.py
    uv run python scripts/compare_retrieval.py --query "..." --tech-id kivi
"""

import argparse
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from kv_eval.rag.retriever import retrieve
from kv_eval.rag.types import RetrievedChunk

PREVIEW_CHARS = 150


@dataclass
class Filters:
    tech_id: str | None = None
    doc_types: list[str] | None = None
    top_k: int = 5

    def describe(self) -> str:
        types = ",".join(self.doc_types) if self.doc_types else "-"
        return f"tech_id={self.tech_id or '-'}  doc_types={types}  top_k={self.top_k}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval quality across collections")
    parser.add_argument("--query", help="Run once and exit instead of prompting.")
    parser.add_argument("--collections", nargs="+", default=["kv_cache_docs_v1", "kv_cache_docs_v2"])
    parser.add_argument("--tech-id")
    parser.add_argument("--doc-type", action="append", dest="doc_types")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def retrieve_from(collection: str, query: str, filters: Filters) -> list[RetrievedChunk]:
    # retrieve() resolves the collection from the environment on every call, so
    # switching it here needs no change to the agent-facing interface.
    os.environ["QDRANT_COLLECTION"] = collection
    return retrieve(
        query=query,
        tech_id=filters.tech_id,
        doc_types=filters.doc_types,
        top_k=filters.top_k,
    )


def print_results(collection: str, chunks: list[RetrievedChunk]) -> None:
    print(f"--- {collection} ---")
    if not chunks:
        print("  (no results)")
        return
    for rank, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.text.split())[:PREVIEW_CHARS]
        print(f"  {rank}. {chunk.score:.4f} [{chunk.doc_id} p.{chunk.page}] {chunk.doc_type}")
        print(f"     {preview}")
    pages = ", ".join(f"{c.doc_id} p.{c.page}" for c in chunks)
    print(f"  pages: {pages}")


def run_query(query: str, collections: list[str], filters: Filters) -> None:
    print()
    print(f"Q: {query}")
    print(f"   {filters.describe()}")
    print()
    for collection in collections:
        try:
            print_results(collection, retrieve_from(collection, query, filters))
        except Exception as exc:
            print(f"--- {collection} ---")
            print(f"  ERROR: {exc}")
        print()


def apply_command(line: str, filters: Filters) -> bool:
    """Handle a `:command`. Returns False when the session should end."""
    parts = line.split()
    command, value = parts[0], parts[1:]

    if command in (":q", ":quit", ":exit"):
        return False
    if command == ":tech":
        filters.tech_id = value[0] if value else None
    elif command == ":type":
        filters.doc_types = value or None
    elif command == ":k" and value:
        filters.top_k = int(value[0])
    elif command == ":help":
        print_help()
        return True
    else:
        print(f"unknown command: {command}  (try :help)")
        return True

    print(f"filters -> {filters.describe()}")
    return True


def print_help() -> None:
    print(
        "\n".join(
            [
                "Type a question, or a command:",
                "  :tech kivi          filter by tech_id (`:tech` alone clears it)",
                "  :type core followup filter by doc_type (`:type` alone clears it)",
                "  :k 5                number of results",
                "  :help               this message",
                "  :q                  quit",
            ]
        )
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    filters = Filters(tech_id=args.tech_id, doc_types=args.doc_types, top_k=args.top_k)

    if args.query:
        run_query(args.query, args.collections, filters)
        return

    print(f"Comparing: {', '.join(args.collections)}")
    print("Loading the embedding model on the first question may take a moment.")
    print_help()

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith(":"):
            if not apply_command(line, filters):
                return
            continue
        run_query(line, args.collections, filters)


if __name__ == "__main__":
    main()
