import tempfile
from pathlib import Path
from src.retrieval import get_chroma_client , get_collection_name ,store_chunks ,query_chunks
from src.extraction import extract
from src.chunking import chunk_blocks
from src.embedding import embed_chunks
from src.generation import generate_answer
from sentence_transformers import SentenceTransformer
import anthropic
import os

anthropic_client = anthropic.Anthropic()

chroma_client = get_chroma_client()
model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = model.tokenizer

def answer_question(file_bytes: bytes, filename: str, question: str, chroma_client, anthropic_client) -> str:
    """
    Orchestrates: hash -> check existing -> (ingest if new) -> retrieve -> generate.
    """
    collection_name = get_collection_name(filename,file_bytes)
    try:
        collection = chroma_client.get_collection(name = collection_name)
        already_ingested = True
    except Exception:
        already_ingested = False

    if not already_ingested:
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            try: 
                tmp.write(file_bytes)
                tmp.flush()
                tmp.close()
            
                blocks = extract(tmp.name)  
                chunks = chunk_blocks(blocks, tokenizer)
                embedded_chunks = embed_chunks(chunks, model)
                store_chunks(embedded_chunks, chroma_client, collection_name)
            finally:
                os.unlink(tmp.name) 

    retrieved_chunks = query_chunks(question,model,chroma_client,collection_name, n_results=5 )
    
    answer = generate_answer(question, retrieved_chunks, anthropic_client)
    return answer