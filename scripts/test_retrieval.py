import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
from src.extraction import extract
from src.chunking import chunk_blocks
from src.embedding import embed_chunks
from learn.retrieval import store_chunks, get_chroma_client ,query_chunks   # your new functions
from pathlib import Path

model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = model.tokenizer
client = get_chroma_client()   # or however you named it — loaded once, same principle

TEST_DOCS_DIR = Path("data/test_docs")

for file_path in sorted(TEST_DOCS_DIR.iterdir()):
    blocks = extract(str(file_path))
    chunks = chunk_blocks(blocks, tokenizer)
    embedded_chunks = embed_chunks(chunks, model)
    active_document_source = "Sample_Bank_Statement.pdf"  # or whichever doc you stored
    matches = query_chunks(
        "who is the account holder ?",
        model, client, source=active_document_source, n_results=5
        )
    for m in matches:
        print(m['distance'], '|', m['page_num'], '|', m['chunk_text'][:100],'|', m['chunk_id'])
    
    # collection_count, chunk_length = store_chunks(embedded_chunks, client)
    # print(f"{file_path.name} : collection_count :{collection_count}, chunk_length : {chunk_length}")
    #print(f"{file_path.name}: collection_count={collection_count}, chunk_length={chunk_length}")
    
    # store_chunks() already prints count() vs len(chunks) internally —
    # watch that output per file