"""
embedding.py — Step 4 of the build.

Responsibility: embed chunks locally (sentence-transformers) and store
them in a persistent per-document ChromaDB collection.

Before wiring this into retrieval.py or the LLM, write a standalone
script that embeds a test doc and manually checks that similarity
search on 5 known questions returns the chunks you'd expect.
"""

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "data/chroma_db"

_model = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model once per process (it's ~90MB)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def embed_and_store(chunks: list[dict], collection_name: str) -> None:
    """Embed all chunks and upsert into a named ChromaDB collection.

    collection_name should be unique per uploaded document/session,
    e.g. a hash of the filename + upload timestamp, so re-uploading
    a doc doesn't collide with a previous session's collection.
    """
    model = get_embedding_model()
    client = get_chroma_client()
    collection = client.get_or_create_collection(collection_name)

    texts = [c["chunk_text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"page_num": c["page_num"], "source": c["source"]} for c in chunks],
    )
