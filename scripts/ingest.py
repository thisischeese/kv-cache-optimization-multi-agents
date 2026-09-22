"""Ingest paper PDFs into Qdrant."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from kv_eval.ingestion.indexer import ingest_manifest
from kv_eval.ingestion.splitter import DEFAULT_CHUNK_OVERLAP_CHARS, DEFAULT_CHUNK_SIZE_CHARS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDF papers into Qdrant")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pdf-root", type=Path, default=Path("data/papers"))
    parser.add_argument(
        "--collection",
        help="Target collection. Defaults to QDRANT_COLLECTION.",
    )
    parser.add_argument("--chunk-size-chars", type=int, default=DEFAULT_CHUNK_SIZE_CHARS)
    parser.add_argument("--chunk-overlap-chars", type=int, default=DEFAULT_CHUNK_OVERLAP_CHARS)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    summary = ingest_manifest(
        manifest_path=args.manifest,
        pdf_root=args.pdf_root,
        collection_name=args.collection,
        chunk_size_chars=args.chunk_size_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
    )

    print(f"Documents: {summary.documents}")
    print(f"Pages processed: {summary.pages_processed}")
    print(f"Chunks created: {summary.chunks_created}")
    print(f"Chunks upserted: {summary.chunks_upserted}")
    print(f"Collection: {summary.collection}")


if __name__ == "__main__":
    main()
