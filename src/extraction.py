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
    }

Do not proceed to chunking.py until each function below has been run
against 5 real, messy documents and the output manually inspected.
"""

from pathlib import Path


def extract_pdf(file_path: str) -> list[dict]:
    """Extract text from a PDF, page by page, using PyMuPDF (fitz)."""
    import pymupdf as fitz  # PyMuPDF — `import fitz` directly is deprecated

    doc = fitz.open(file_path)
    source = Path(file_path).name
    blocks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:  # skip blank pages, don't create empty chunks
            blocks.append({"text": text, "page_num": page_num, "source": source})
    doc.close()
    return blocks


def extract_docx(file_path: str) -> list[dict]:
    """Extract text from a DOCX, paragraph by paragraph (no true page concept)."""
    import docx

    document = docx.Document(file_path)
    source = Path(file_path).name
    blocks = []
    for i, para in enumerate(document.paragraphs, start=1):
        text = para.text.strip()
        if text:
            blocks.append({"text": text, "page_num": i, "source": source})
    return blocks


def extract_txt(file_path: str) -> list[dict]:
    """Extract text from a plain TXT file, split by paragraph (blank-line separated)."""
    source = Path(file_path).name
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return [
        {"text": p, "page_num": i, "source": source}
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
