"""
embedding.py — Step 4 of the build.

Responsibility: embed chunks locally (sentence-transformers) and store
them in a persistent per-document ChromaDB collection.

Before wiring this into retrieval.py or the LLM, write a standalone
script that embeds a test doc and manually checks that similarity
search on 5 known questions returns the chunks you'd expect.
"""

from sentence_transformers import SentenceTransformer

def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    
    """Load and return a SentenceTransformer model.
        """
        
    model = SentenceTransformer(model_name)
    return model


def validate_chunk_length(text: str, tokenizer, max_tokens: int = 256) -> int:
    """
    Check the TRUE token count of `text` as the embedding model will
    actually see it at encode time.
    """
    token_ids = tokenizer(text)["input_ids"]  
    return len(token_ids)


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    
    """
        Input: list of chunk dicts (each with at least a 'chunk_text' field
        and metadata like 'chunk_id', 'page_num', 'source', 'is_table').
    
        Output: the SAME chunk dicts, each with a new 'embedding' field added
        — NOT a bare list of vectors. Keeps vector + metadata bound together.
        """
    chunk_text_list=[]
    for chunk in chunks:
        chunk_length= validate_chunk_length(text=chunk['chunk_text'], tokenizer= model.tokenizer)
        if chunk_length > model.max_seq_length:
            print(f"chunk_length({chunk_length}) is greater than allowed size {model.max_seq_length}")
        chunk_text_list.append( chunk['chunk_text'])
    
    embeddings = model.encode(chunk_text_list, normalize_embeddings=True, batch_size=32)
    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding
    return chunks
        
    
def embed_query(query: str, model: SentenceTransformer):
    
    """
    Embed a single user query string — called at RETRIEVAL time, not
    indexing time. Lives here (not in retrieval.py) because it's still
    fundamentally "text -> vector via this model" 
    """
    query_length= len(model.tokenizer(query)["input_ids"])
    if (query_length) > model.max_seq_length:
        print(f"chunk_length({query_length}) is greater than allowed size {model.max_seq_length}")
    query_embeddings = model.encode(query, normalize_embeddings=True)
    return query_embeddings
    