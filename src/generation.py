"""
generation.py — Step 5 of the build. The most important file in the repo.

Responsibility: assemble retrieved chunks + chat history into a prompt,
call Claude, and return a grounded answer with citations.

Before moving to the UI step, test this ADVERSARIALLY: ask questions
whose answers are NOT in the document and confirm the model refuses
instead of guessing. Log every refusal/non-refusal during eval (Step 7).
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question \
using ONLY the context chunks provided below. Do not use any outside knowledge, \
even if you are confident about the answer.

If the answer cannot be found in the provided context, respond exactly with: \
"This document does not contain that information." Do not guess, infer beyond \
what's stated, or fill gaps with general knowledge.

When you do answer, cite which chunk(s) you used by their page number."""


def generate_answer(question: str, retrieved_chunks: list[dict],
                     chat_history: list[dict] | None = None) -> dict:
    """Generate a grounded answer.

    Output: {"answer": str, "sources": list[dict]} where sources is the
    subset of retrieved_chunks actually used (for the UI's citation panel).
    """
    context = "\n\n".join(
        f"[Page {c['page_num']}]: {c['chunk_text']}" for c in retrieved_chunks
    )

    messages = []
    if chat_history:
        # keep only last 3 turns per REQUIREMENTS.md — don't blow up context
        messages.extend(chat_history[-3:])

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    answer_text = response.content[0].text
    return {"answer": answer_text, "sources": retrieved_chunks}
