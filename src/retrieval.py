"""
retrieval.py — Step 4 (second half) of the build.

Responsibility: given a user question, embed it and pull the top-k
most relevant chunks from ChromaDB. Test this standalone (print
results for 5 hardcoded questions) before connecting to generation.py.
"""

from .embedding import get_embedding_model, get_chroma_client

TOP_K = 5


def retrieve(question: str, collection_name: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k most relevant chunks for a question.

    Output: list[dict] — {"chunk_text", "page_num", "source", "distance"}
    Lower "distance" = more similar. Log this value during testing —
    a consistently high distance on your top result is a signal your
    retrieval (or chunking) is broken, not that the doc lacks the answer.
    """
    model = get_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection(collection_name)

    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "chunk_text": results["documents"][0][i],
            "page_num": results["metadatas"][0][i]["page_num"],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })
    return retrieved
