"""Check Qdrant Cloud configuration and connectivity."""

from dotenv import load_dotenv

from kv_eval.config import qdrant_api_key, qdrant_collection, qdrant_endpoint
from kv_eval.rag.qdrant_store import collection_vector_name, collection_vector_names, get_qdrant_client


def main() -> None:
    load_dotenv()

    endpoint = qdrant_endpoint()
    api_key = qdrant_api_key()
    collection = qdrant_collection()

    if not endpoint:
        raise RuntimeError("QDRANT_ENDPOINT is not set")
    if not api_key:
        raise RuntimeError("QDRANT_API_KEY is not set")

    client = get_qdrant_client()
    collections = client.get_collections()
    names = {item.name for item in collections.collections}

    print("Qdrant connection: OK")
    print(f"Collection: {collection}")
    exists = collection in names
    print(f"Collection exists: {'yes' if exists else 'no'}")
    if exists:
        try:
            vector_name = collection_vector_name(client, collection)
            print(f"Vector mode: {'named' if vector_name else 'unnamed'}")
            if vector_name:
                print(f"Vector name: {vector_name}")
        except RuntimeError as exc:
            names = collection_vector_names(client, collection)
            if names:
                print("Vector mode: multiple named")
                print(f"Vector names: {', '.join(names)}")
            else:
                print("Vector mode: no dense vectors")
            print(f"Vector selection error: {exc}")


if __name__ == "__main__":
    main()
