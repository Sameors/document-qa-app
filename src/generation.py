import anthropic

MODEL_NAME = "claude-haiku-4-5-20251001"   
#NO_ANSWER_PHRASE = "I cannot find this information in the document." 
REFUSAL_MESSAGE = "I cannot find this information in the document."


def format_context(chunks: list[dict]) -> str:
    """
        Take the list of retrieved chunk dicts (chunk_text, page_num, source,
        distance, ...) and turn them into a single formatted string to embed
        in the prompt.
        """
        
    blocks = []
    for chunk in chunks:
        blocks.append(f'<chunk page="{chunk["page_num"]}">\n{chunk["chunk_text"]}\n</chunk>')
    return "\n\n".join(blocks)


def build_prompt(question: str, formatted_context: str) -> str:   
    
    
    return f'''
            [INSTRUCTIONS]
                - Evaluate context in formatted_context to answer the question.
                - Must communicate, clearly and unambiguously.
                - Answer ONLY using the provided context, not general knowledge.
                - Cite the page number(s) the answer came from.
                - If the context doesn't contain the answer, say strictly {REFUSAL_MESSAGE} only.
                  
            [CONTEXT]
            {formatted_context}
            
            [QUESTION]
            {question}

           '''

def call_claude(prompt: str, client: anthropic.Anthropic) -> str:
    """
    Send prompt to Claude and return the generated answer text.
    """
    response = client.messages.create(
        model= MODEL_NAME,    
         max_tokens=800,         
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text


def generate_answer(question: str, chunks: list[dict], client: anthropic.Anthropic) -> str:
    """
    The orchestrating function
    """
    if not chunks:
        return REFUSAL_MESSAGE
    formatted_context = format_context(chunks)
    prompt = build_prompt(question, formatted_context)
    Response = call_claude(prompt, client) 
    return Response

