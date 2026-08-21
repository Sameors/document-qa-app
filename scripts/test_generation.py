import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import anthropic
from dotenv import load_dotenv
import os


from sentence_transformers import SentenceTransformer
from src.extraction import extract
from src.chunking import chunk_blocks
from src.embedding import embed_chunks
from src.retrieval import store_chunks, get_chroma_client ,query_chunks   # your new functions
from src.generation import format_context , build_prompt , call_claude,generate_answer
from pathlib import Path

load_dotenv()
os.environ["ANTHROPIC_API_KEY"]

model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = model.tokenizer
client = get_chroma_client()
anthropic_client = anthropic.Anthropic()# or however you named it — loaded once, same principle

TEST_DOCS_DIR = Path("data/test_docs")
active_document_source = "Monarch_Butterfly_Migration.pdf" 
Query =  "what is the lifecycle of Monarch Butterfly"

for file_path in sorted(TEST_DOCS_DIR.iterdir()):
    if (Path(file_path).name) == active_document_source:
        blocks = extract(str(file_path))
        chunks = chunk_blocks(blocks, tokenizer)
        #print(chunks)
        embedded_chunks = embed_chunks(chunks, model)
        store_chunks(chunks, client)
        matches = query_chunks(
            Query,
            model, client, source=active_document_source, n_results=5
            )
        # for match in matches:
        #     print(f'chunk_id : {match['chunk_id']} ,distance : {match['distance']}')
        # #print(matches)
        #formatted_content = format_context(matches)
        #prompt = build_prompt(Query, formatted_content)
        response = generate_answer(Query,matches,anthropic_client)
        print(response)
        
     
    # for m in matches:
    #     print(m['distance'], '|', m['page_num'], '|', m['chunk_text'][:100],'|', m['chunk_id'])
    
    # collection_count, chunk_length = store_chunks(embedded_chunks, client)
    # print(f"{file_path.name} : collection_count :{collection_count}, chunk_length : {chunk_length}")
    #print(f"{file_path.name}: collection_count={collection_count}, chunk_length={chunk_length}")
    
    # store_chunks() already prints count() vs len(chunks) internally —
    # watch that output per file