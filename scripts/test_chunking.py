"""
scripts/test_chunking.py — verify the full extract -> chunk pipeline
against every real test document, using the actual shipped modules
(src.extraction, src.chunking), not scratch/learn code.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction import extract
from src.chunking import chunk_blocks

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = model.tokenizer

TEST_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "test_docs"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def summarize(chunks: list[dict], file_path: Path) -> None:
    set_id = []
    print(f"chunk-count - {len(chunks)}")
    for chunk in chunks:
        print(f"chunk-id - {chunk["chunk_id"]} ,is-table - {chunk["is_table"]} ,page_num - {chunk["page_num"]} ")
        set_id.append(chunk["chunk_id"])
    if len(set_id)== len(chunks):
        print(f"chunks are matched")
    else:
        print(f"chunk length mismatch")
        
    pass


def main() -> None:
    
    TEST_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "test_docs"/"Blank_Page_Then_Table.pdf"
    blocks = extract(str(TEST_DOCS_DIR))
    chunks = chunk_blocks(blocks,tokenizer)
    summarize(chunks=chunks,file_path=TEST_DOCS_DIR)
    
    # files = [f for f in TEST_DOCS_DIR.iterdir()
    #          if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    # for file_path in sorted(files):
    #      blocks = extract(str(file_path))
    #      chunks = chunk_blocks(blocks,tokenizer)
    #      #print(chunks)
    #      summarize(chunks=chunks,file_path=file_path)
      
    pass


if __name__ == "__main__":
    main()