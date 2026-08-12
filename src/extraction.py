"""
extraction.py — Step 2 of the build.

Responsibility: turn a raw uploaded file (PDF/DOCX/TXT) into a list of
text blocks with page/paragraph metadata attached. Nothing downstream
should ever touch a raw file — this is the only layer that does.

Contract every extract_* function must follow:
    Input:  file path (str)
    Output: list[dict], each dict = {
        "text": str,          # raw extracted text for this unit
        "page_num": int,      # 1-indexed page (PDF) or paragraph index (DOCX/TXT)
        "source": str,        # original filename
        "is_table": bool,     # True if this block is a structured table
                               #   (Markdown format). False for prose blocks.
                               #   chunking.py MUST treat is_table=True blocks
                               #   as atomic — never split mid-table.
    }

PDF table extraction (FR10, REQUIREMENTS.md): tables are detected via
pdfplumber and converted to Markdown so row/column association survives
as explicit structure, instead of collapsing into a flat, ambiguous
string. This was added after Step 2 testing found PyMuPDF's plain-text
extraction silently separates table headers from their values on
structured documents (e.g. bank statements) — see REQUIREMENTS.md
Known Limitations for the original evidence.

Do not proceed to chunking.py until each function below has been run
against real messy documents (including at least one table-heavy PDF
and one multi-column PDF) and the output manually inspected.
"""

from pathlib import Path


def _table_to_markdown(rows: list[list]) -> str:
    """Convert a pdfplumber-extracted table (list of row lists) to a
    Markdown table string. None cells become empty strings so the
    LLM sees a blank cell rather than the literal word 'None'.
    """
    if not rows:
        return ""

    clean_rows = [[(cell if cell is not None else "").strip() for cell in row]
                  for row in rows]

    header, *body = clean_rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        # pad/truncate rows that don't match header length rather than
        # crashing — malformed tables are common, don't let one bad row
        # kill extraction for the whole document
        row = (row + [""] * len(header))[:len(header)]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def extract_pdf(file_path: str) -> list[dict]:
    """Extract text from a PDF, page by page, using pdfplumber.

    Tables are detected and extracted as separate, structured Markdown
    blocks (is_table=True). Prose is extracted with table regions
    cropped out first, so table content never gets duplicated as
    garbled flat text alongside the clean Markdown version.
    """
    import pdfplumber

    source = Path(file_path).name
    blocks = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()

            for table in tables:
                rows = table.extract()
                md = _table_to_markdown(rows)
                if md.strip():
                    blocks.append({
                        "text": md,
                        "page_num": page_num,
                        "source": source,
                        "is_table": True,
                    })

            # crop out every detected table's bounding box before pulling
            # prose text, so table content isn't duplicated in flat form
            prose_page = page
            for table in tables:
                prose_page = prose_page.outside_bbox(table.bbox)

            prose_text = (prose_page.extract_text() or "").strip()
            if prose_text:
                blocks.append({
                    "text": prose_text,
                    "page_num": page_num,
                    "source": source,
                    "is_table": False,
                })

    return blocks


def extract_docx(file_path: str) -> list[dict]:
    """Extract text from a DOCX, paragraph by paragraph (no true page concept).

    KNOWN GAP: this does not yet extract DOCX tables (document.tables in
    python-docx) as structured Markdown the way extract_pdf does. If a
    DOCX contains a table, its cell text is currently NOT extracted at
    all — python-docx's .paragraphs does not include table content.
    This is a real, undocumented-until-now hole; flagged in
    REQUIREMENTS.md as a follow-up, not silently ignored.
    """
    import docx

    document = docx.Document(file_path)
    source = Path(file_path).name
    blocks = []
    for i, para in enumerate(document.paragraphs, start=1):
        text = para.text.strip()
        if text:
            blocks.append({"text": text, "page_num": i, "source": source, "is_table": False})
    return blocks


def extract_txt(file_path: str) -> list[dict]:
    """Extract text from a plain TXT file, split by paragraph (blank-line separated)."""
    source = Path(file_path).name
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return [
        {"text": p, "page_num": i, "source": source, "is_table": False}
        for i, p in enumerate(paragraphs, start=1)
    ]


def extract(file_path: str) -> list[dict]:
    """Dispatch to the right extractor based on file extension.

    Raises ValueError on unsupported types — the UI layer must catch
    this and show a clean error, not a stack trace.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    elif ext == ".txt":
        return extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
