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
        "chunk_id": str,       # globally unique across the WHOLE document —
                                #   assigned in a second pass over every chunk,
                                #   after all blocks are processed (see
                                #   chunk_blocks below). A per-block-only index
                                #   collided when multiple tables shared one page.
        "is_table": bool,      # carried through so retrieval/generation can
                                #   know the source shape if needed later
    }

CRITICAL: is_table=True blocks are NEVER split by token-count chunking,
even if they exceed chunk_size. Splitting a Markdown table mid-row
re-introduces the exact header/value separation bug that table
extraction was built to fix (REQUIREMENTS.md FR10). A large table
becomes one large chunk instead — acceptable for v1; revisit only if
you hit tables large enough to blow the LLM's context window.

Decomposed into three functions, each independently testable:
    split_into_chunks(text, ...) -> fixed-size token splitting w/ overlap,
                                     used only for prose
    chunk_block(block, ...)      -> one block -> list of chunk dicts
                                     (always a list, even for tables)
    chunk_blocks(blocks, ...)    -> the full block list -> flattened,
                                     globally-unique-ID chunk list
"""


CHUNK_SIZE_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 20


def split_into_chunks(text: str, tokenizer ,chunk_size: int, overlap: int = 0) -> list[str]:
    """Split text into chunks of at most chunk_size tokens, with overlap."""
    # enc = tiktoken.get_encoding("cl100k_base")
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    result_list = []
    for start in range(0, len(token_ids), chunk_size - overlap):
        end = start + chunk_size
        result_list.append(tokenizer.decode(token_ids[start:end]))
    return result_list


def chunk_block(block: dict, tokenizer, chunk_size: int, overlap: int) -> list[dict]:
    """Turn one extracted block into one or more chunk dicts.

    Table blocks are never split — always returned as a single,
    unsplit chunk, regardless of chunk_size, to protect table
    structure (see REQUIREMENTS.md, the header/value separation bug).
    """
    if block['is_table']:
        chunk_id = f"{block['source']}_p{block['page_num']}_table"
        block_tokens = tokenizer(block['text'])["input_ids"]

        if (len(block_tokens)) < CHUNK_SIZE_TOKENS:
            return [{
            'chunk_text': block['text'],
            'page_num': block['page_num'],
            'source': block['source'],
            'is_table': block['is_table'],
            'chunk_id': chunk_id
        }]
        else:
            #print("long table", block)
            
            table_chunk= []
            table_group = split_table_by_rows(table_text= block['text'], 
                                tokenizer = tokenizer, chunk_size=chunk_size)
            for group in table_group:
                single_table_group = {"chunk_text": group,
                             "page_num": block['page_num'],
                             "source": block['source'],
                             "is_table": block['is_table'],
                             "chunk_id": f"{block['source']}_p{block['page_num']}_table"}
                table_chunk.append(single_table_group)
        return table_chunk         
    else:
        prose_chunk = []
        chunks = split_into_chunks(text=block["text"],tokenizer=tokenizer ,chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            single_prose = {"chunk_text": chunk,
                             "page_num": block['page_num'],
                             "source": block['source'],
                             "is_table": block['is_table'],
                             "chunk_id": f"{block['source']}_p{block['page_num']}_prose"}
            prose_chunk.append(single_prose)
    return prose_chunk

def split_table_by_rows(table_text: str, tokenizer, chunk_size: int) -> list[str]:
    """
    table_text: the full table block's text, structured as lines
    (header line, separator line, then one data row per line — confirm this).

    Returns a list of chunk strings. Each chunk must start with the
    header (and separator line, if you're keeping markdown formatting)
    so it's independently interpretable. """
    table_split=[]
    current_group = []
    lines = table_text.split('\n')
    header_lines = lines[:2]
    header_token = tokenizer('\n'.join(header_lines),add_special_tokens=False)["input_ids"]
    if len(header_token) > CHUNK_SIZE_TOKENS:
        print(f"WARNING: table chunk exceeds {CHUNK_SIZE_TOKENS} tokens: {len(header_token)}")
    
    for line in lines[2:]:
        table_line_token = tokenizer(line,add_special_tokens=False)["input_ids"]
        current_group_token = tokenizer('\n'.join(current_group),add_special_tokens=False)["input_ids"]
        
        if len(current_group_token) + len(table_line_token) + len(header_token) < CHUNK_SIZE_TOKENS:
            current_group.append(line)
        else :
           
            table_split.append('\n'.join(header_lines + current_group))
            current_group.clear()
            current_group.append(line)
            current_group_token = tokenizer('\n'.join(current_group),add_special_tokens=False)["input_ids"]
            
            if len(current_group_token) > CHUNK_SIZE_TOKENS:
                print(f"WARNING: table data chunk exceeds {CHUNK_SIZE_TOKENS} tokens: {len(current_group_token)}")
                table_split.append('\n'.join(header_lines + current_group))
                current_group.clear()
                
            
    table_split.append('\n'.join(header_lines + current_group))
    return table_split
    
pass

def chunk_blocks(blocks: list[dict], tokenizer,chunk_size: int = CHUNK_SIZE_TOKENS,
                  overlap: int = CHUNK_OVERLAP_TOKENS) -> list[dict]:
    """Flatten every block's chunks into one list, then assign each
    chunk a globally unique ID based on its final position — fixes a
    real collision found in testing when multiple tables shared a page.
    """
    all_chunks = []
    
    for block in blocks:
        
        one_chunk = chunk_block(block=block,tokenizer=tokenizer, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(one_chunk)

    for i, chunk in enumerate(all_chunks):
        chunk['chunk_id'] = f"{chunk['chunk_id']}{i}"

    return all_chunks