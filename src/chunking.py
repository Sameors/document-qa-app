"""
chunking.py — Step 3 of the build.

Responsibility: turn extraction.py's text blocks into retrieval-sized
chunks with metadata preserved. This is the #1 place naive RAG demos
silently break — verify output by reading 10-15 chunks yourself before
moving to embedding.py.

Contract:
    Input:  list[dict] from extraction.py — {"text", "page_num", "source", "is_table"}
    Output: list[dict] — {
        "chunk_text": str,
        "page_num": int,
        "source": str,
        "chunk_id": str,       # unique id, e.g. f"{source}_p{page_num}_c{i}"
        "is_table": bool,      # carried through so retrieval/generation can
                                #   know the source shape if needed later
    }

CRITICAL: is_table=True blocks are NEVER split by token-count chunking,
even if they exceed chunk_size. Splitting a Markdown table mid-row
re-introduces the exact header/value separation bug that table
extraction was built to fix (REQUIREMENTS.md FR10). A large table
becomes one large chunk instead — acceptable for v1; revisit only if
you hit tables large enough to blow the LLM's context window.
"""

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50


def chunk_blocks(blocks: list[dict], chunk_size: int = CHUNK_SIZE_TOKENS,
                  overlap: int = CHUNK_OVERLAP_TOKENS) -> list[dict]:
    """Fixed-size token chunking with overlap, per extracted block.

    Table blocks (is_table=True) bypass splitting entirely and become
    a single atomic chunk, regardless of size.

    NOTE: for prose, this is the fallback strategy. Before relying on
    it fully, decide (per REQUIREMENTS.md) whether you're implementing
    structural chunking (split by heading/paragraph boundary) first and
    only falling back to fixed-size for blocks without clear structure.
    """
    enc = None  # lazy-loaded — only needed if a non-table block requires splitting
    chunks = []

    for block in blocks:
        if block.get("is_table"):
            # atomic — never split a table mid-row
            chunks.append(_make_chunk(block["text"], block, len(chunks)))
            continue

        if enc is None:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")

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
        "is_table": block.get("is_table", False),
    }
