"""Create payload indexes needed for metadata-filtered retrieval."""

from dotenv import load_dotenv

from kv_eval.config import qdrant_collection
from kv_eval.rag.qdrant_store import (
    INTEGER_PAYLOAD_FIELDS,
    KEYWORD_PAYLOAD_FIELDS,
    ensure_payload_indexes,
    get_qdrant_client,
)


def main() -> None:
    load_dotenv()
    collection = qdrant_collection()
    client = get_qdrant_client()
    if not client.collection_exists(collection):
        raise RuntimeError(f"Qdrant collection {collection!r} does not exist")

    ensure_payload_indexes(client, collection)
    print("Payload indexes: OK")
    print(f"Collection: {collection}")
    print(f"Keyword fields: {', '.join(KEYWORD_PAYLOAD_FIELDS)}")
    print(f"Integer fields: {', '.join(INTEGER_PAYLOAD_FIELDS)}")


if __name__ == "__main__":
    main()
