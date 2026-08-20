import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np


from sentence_transformers import SentenceTransformer
from src.extraction import extract
from src.chunking import chunk_blocks
from src.embedding import embed_chunks , embed_query # once you've written it
from pathlib import Path

# Load model ONCE — this is the "load once, pass down" pattern again,
# now spanning across two different files (chunking needs the tokenizer,
# embedding needs the whole model)
model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = model.tokenizer



#TEST_DOCS_DIR = Path("data/test_docs")
TEST_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "test_docs"/"Renaissance_Notes.txt"
blocks = extract(str(TEST_DOCS_DIR))
chunks = chunk_blocks(blocks,tokenizer)
embedded_chunks = embed_chunks(chunks, model)
query_vector = embed_query("what was my largest withdrawal in July?", model)

for chunk in embedded_chunks:
    #print(chunk)
    assert len(chunk['embedding']) == 384, f"{chunk['chunk_id']} has wrong dimension: {len(chunk['embedding'])}"
    if 'embedding' not in chunk:
        print(f"MISSING: {chunk['chunk_id']} has no embedding at all")
print("done checking for missing embeddings")
print("all embeddings are 384-dim")
print(f"query_chunks : {query_vector} , length :{len(query_vector)}" )


#first_chunk = embedded_chunks[0]
#length_of_vector = np.linalg.norm(first_chunk['embedding'])
#print(length_of_vector)

# print(f"{len(embedded_chunks)} chunks embedded")
# first = embedded_chunks[0]
# print("keys:", first.keys())              # confirm 'embedding' key exists now
# print("embedding length:", len(first['embedding']))  # should be 384 for MiniLM
# print("chunk_text preserved:", first['chunk_text'][:50])  # confirm original fields untouched

# for file_path in sorted(TEST_DOCS_DIR.iterdir()):
#     blocks = extract_pdf(str(file_path))
#     chunks = chunk_blocks(blocks, tokenizer)
#     embedded_chunks = embed_chunks(chunks, model)

#     # sanity checks — don't just trust it ran without erroring
#     print(f"{file_path.name}: {len(embedded_chunks)} chunks embedded")
#     first = embedded_chunks[0]
#     print("keys:", first.keys())              # confirm 'embedding' key exists now
#     print("embedding length:", len(first['embedding']))  # should be 384 for MiniLM
#     print("chunk_text preserved:", first['chunk_text'][:50])  # confirm original fields untouched