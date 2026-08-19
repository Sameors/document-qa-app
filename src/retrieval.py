import chromadb
import re
import hashlib
from src.embedding import embed_query

MAX_COLLECTION_NAME_LENGTH = 63  
HASH_LENGTH = 10  
SEPARATOR_LENGTH = 1


def get_chroma_client(persist_dir: str = "data/chroma_db") -> chromadb.PersistentClient:
    """
    Create (or connect to) the persistent ChromaDB client.
    """
    return chromadb.PersistentClient(path= persist_dir)


def store_chunks(chunks: list[dict], client: chromadb.PersistentClient) -> None:
    """
    Store embedded chunks (output of embed_chunks()) into a per-document
    ChromaDB collection.
    """
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    for chunk in chunks:
        ids.append(chunk['chunk_id'])
        documents.append(chunk['chunk_text'])
        embeddings.append(chunk['embedding'])
        metadatas.append({'page_num': chunk['page_num'],
                        'source': chunk['source'],
                        'is_table': chunk['is_table']}
                        )
    collection_name = get_collection_name(chunks[0]["source"])
    collection = client.get_or_create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}})
    collection.upsert(
            ids = ids,
            documents = documents,
            embeddings = embeddings,
            metadatas = metadatas
            )
    print(f"stored {collection.count()} chunks (expected {len(chunks)})")
    

def query_chunks(query: str, model, client: chromadb.PersistentClient,
                  source: str, n_results: int = 5) -> list[dict]:
    """
    Given a user's question and a document's source filename, return
    the top-k most relevant chunks from that document's collection.
    """
    
    query_embedding = embed_query(query, model)
    query_name = get_collection_name(source)
    collection = client.get_collection(name = query_name)
    results = collection.query(
        query_embeddings= [query_embedding],
        n_results= n_results,
        )
    matched_chunks = []
    for chunk_id, chunk_text, metadata, distance in zip(
            results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0]):
        matched_chunks.append({
            'chunk_id': chunk_id,
            'chunk_text': chunk_text,
            'page_num': metadata['page_num'],
            'source': metadata['source'],
            'is_table': metadata['is_table'],
            'distance': distance,
            })
    return matched_chunks
    
def get_collection_name(source: str) -> str:
    
    """
    Generate a valid, deterministic ChromaDB collection name from a
    source filename.
    """
    name_without_ext = re.sub(r'\.[^.]+$', '', source)
    safe_ext = (re.search(r'\.[^.]+$', source)).group(0)
    cleaned_prefix_text = re.sub(r'\W', '_', name_without_ext)
    if cleaned_prefix_text.startswith('_') or not cleaned_prefix_text:
        cleaned_prefix_text = safe_ext[1:] + cleaned_prefix_text
    max_prefix_length = MAX_COLLECTION_NAME_LENGTH - (HASH_LENGTH + SEPARATOR_LENGTH) - 1
    collection_prefix = cleaned_prefix_text.lower()[:max_prefix_length]
    full_hash = hashlib.sha256(source.encode('utf-8')).hexdigest()
    return f"{collection_prefix}_{full_hash[:HASH_LENGTH]}"
