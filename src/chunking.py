"""
chunking.py — Step 3 of the build.

Responsibility: turn extraction.py's text blocks into retrieval-sized
chunks with metadata preserved. This is the #1 place naive RAG demos
silently break — verify output by reading 10-15 chunks yourself before
moving to embedding.py.

Contract:
    Input:  list[dict] from extraction.py — {"text", "page_num", "source"}
    Output: list[dict] — {
        "chunk_text": str,
        "page_num": int,
        "source": str,
        "chunk_id": str,       # unique id, e.g. f"{source}_p{page_num}_c{i}"
    }
"""

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50


def chunk_blocks(blocks: list[dict], chunk_size: int = CHUNK_SIZE_TOKENS,
                  overlap: int = CHUNK_OVERLAP_TOKENS) -> list[dict]:
    """Fixed-size token chunking with overlap, per extracted block.

    NOTE: this is the fallback strategy. Before relying on it, decide
    (per REQUIREMENTS.md) whether you're implementing structural
    chunking (split by heading/paragraph boundary) first and only
    falling back to fixed-size for blocks without clear structure.
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []

    for block in blocks:
        tokens = enc.encode(block["text"])
        if len(tokens) <= chunk_size:
            chunks.append(_make_chunk(block["text"], block, len(chunks)))
            continue

        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = enc.decode(tokens[start:end])
            chunks.append(_make_chunk(chunk_text, block, len(chunks)))
            if end == len(tokens):
                break
            start = end - overlap  # step forward, keep overlap

    return chunks


def _make_chunk(text: str, block: dict, idx: int) -> dict:
    return {
        "chunk_text": text.strip(),
        "page_num": block["page_num"],
        "source": block["source"],
        "chunk_id": f"{block['source']}_p{block['page_num']}_c{idx}",
    }
