# Document Q&A App — Requirements Documentation

## 1. Problem Statement
Users need to ask questions against their own documents (PDF/DOCX/TXT) and
get answers that are **grounded in the document's actual content** — not
the model's general knowledge — with **verifiable citations** back to the
source text. The system must explicitly refuse to answer when the
information is not present in the document, rather than hallucinating.

## 2. Scope (v1)

### In scope
- File types: PDF, DOCX, TXT (text-based only)
- Single document per session (no multi-document cross-referencing in v1)
- Multi-turn conversation (last 3 turns retained as context)
- Citation to page number (PDF) or paragraph number (DOCX/TXT) for every answer
- Explicit refusal when the answer isn't in the document
- Local execution (Streamlit), deployed later to Streamlit Cloud/HF Spaces

### Explicitly out of scope (v1)
- OCR / scanned or image-based PDFs
- DOCX table extraction (PDF tables ARE in scope as of FR10 below — DOCX
  table support is a real gap, flagged as a follow-up, not silently dropped)
- Multi-document / cross-document Q&A
- User authentication or multi-user persistence

## 3. Functional Requirements

| ID | Requirement | Acceptance Criteria |
|----|-------------|----------------------|
| FR1 | Accept PDF, DOCX, TXT uploads | Each format extracts without error on 5 real test documents |
| FR2 | Chunk extracted text | Chunks preserve page/paragraph metadata; manually verified not to cut mid-sentence on test docs |
| FR3 | Embed chunks locally | `sentence-transformers` (`all-MiniLM-L6-v2`), no external API call for embeddings |
| FR4 | Persist vectors | ChromaDB, one collection per uploaded document/session |
| FR5 | Retrieve top-k relevant chunks | k=5, cosine similarity; manually verified relevant on 10 test questions |
| FR6 | Generate grounded answers | Claude Haiku API; system prompt enforces context-only answering |
| FR7 | Refuse when unanswerable | On questions with no answer in doc, model responds with the fixed refusal string ≥90% of the time (measured in eval set, Step 7) |
| FR8 | Show citations | Every non-refused answer displays the exact source chunk + page/paragraph number |
| FR9 | Multi-turn context | Follow-up questions referencing prior turns are answered correctly on at least 3 manual test cases |
| FR10 | PDF table extraction | Tables in PDFs are detected via `pdfplumber` and converted to Markdown, preserving header/value (row/column) association. Verified against a real bank-statement-style test document — see Section 9 for the evidence that motivated this requirement. Table chunks are never split mid-table by the chunking layer. |

## 4. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR1 | Answer latency < 5 seconds for a 50-page document, top-k=5 |
| NFR2 | No API keys committed to git; loaded via `.env` (gitignored) |
| NFR3 | Corrupted, empty, or unsupported files produce a clean UI error, never a raw stack trace |
| NFR4 | Codebase split by responsibility (extraction / chunking / embedding / retrieval / generation) — no monolithic script |

## 5. Architecture

```
Upload (PDF/DOCX/TXT)
      ↓
extraction.py    → per-format parser → text blocks + page metadata
      ↓
chunking.py      → structural/fixed-size split → chunks + metadata
      ↓
embedding.py     → sentence-transformers → vectors → ChromaDB
      ↓
[user question]
      ↓
retrieval.py     → embed query → top-k chunk retrieval
      ↓
generation.py    → context + refusal-enforcing prompt → Claude Haiku
      ↓
app/streamlit_app.py → answer + expandable sources panel
```

## 6. Key Design Decisions (locked for v1)

1. **Embeddings are local/free** (`sentence-transformers`); **generation
   uses a paid API** (Claude Haiku) — local LLMs are too unreliable at
   grounded refusal for this project's core claim to hold up.
2. **Refusal prompt is the most important artifact in the codebase.**
   It must be adversarially tested, not just written once and trusted.
3. **Citation format differs by file type**: page number for PDF,
   paragraph number for DOCX/TXT — decided up front to avoid rework.
4. **Chunk size**: 500 tokens, 50-token overlap, as a fallback for
   blocks without clear structural boundaries.

## 7. Evaluation Plan
- Build a 15-question test set against one real test document:
  - ~10 questions with known, verifiable correct answers
  - ~4-5 "trick" questions with no answer in the document
- Metrics tracked: retrieval hit-rate (right chunk in top-k), answer
  correctness, refusal correctness (did it refuse when it should have,
  and only when it should have)
- Pass rate is reported in the README — not optional.

## 8. Known Limitations (to state explicitly in README)
- No OCR — scanned/image PDFs will extract as empty or garbled text
- DOCX tables are not extracted at all (python-docx's paragraph API skips
  table content entirely) — a real gap, not yet fixed
- Complex PDF tables (merged cells, nested tables, multi-line cell wraps)
  may still extract imperfectly — pdfplumber handles simple/common table
  layouts reliably, not every layout
- Single document per session only
- Local-only in v1 (deployment is a later milestone)

## 9. Evidence Log

**FR10 motivation (PDF table extraction):** Initial Step 2 testing used
5 documents that were all clean, single-column, table-free text — a
selection bias that produced a false-positive "extraction works" signal.
Retesting with a bank-statement-style PDF containing an "Account Summary"
table exposed the real failure: PyMuPDF's flat text extraction separated
table headers ("Beginning Balance", "Total Deposits"...) from their
values ("$4,812.55", "$6,240.00"...) into two disconnected text runs,
making it impossible to reliably determine which value belonged to
which label. Since financial/structured documents are a core target
use case for this project (not a peripheral one), this was reclassified
from an accepted limitation to a functional requirement (FR10) and
fixed via pdfplumber-based table detection + Markdown conversion,
verified against a synthetic reproduction of the same table structure.
