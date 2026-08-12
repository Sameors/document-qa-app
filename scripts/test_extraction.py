"""
scripts/test_extraction.py — Step 2 manual verification script.

Run from the project root:
    python scripts/test_extraction.py

Loops through every file in data/test_docs/, runs it through
src.extraction.extract(), and prints a summary plus the first and
last block so you can eyeball whether chunk boundaries make sense.

This is NOT an automated pass/fail test — it's a tool for YOU to read
the output and judge quality. Automated eval comes later, in Step 7,
once you know what "good" extraction looks like.
"""

import sys
from pathlib import Path

# allow running this script directly via `python scripts/test_extraction.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction import extract

TEST_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "test_docs"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def summarize(blocks: list[dict], file_path: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"FILE: {file_path.name}")
    print(f"{'=' * 70}")

    if not blocks:
        print("  ⚠️  NO BLOCKS EXTRACTED — extraction returned an empty list.")
        print("  This usually means: scanned/image-only PDF, empty file,")
        print("  or a parsing failure that was silently swallowed.")
        return

    total_chars = sum(len(b["text"]) for b in blocks)
    empty_blocks = [b for b in blocks if not b["text"].strip()]
    avg_block_len = total_chars / len(blocks)

    print(f"  Blocks extracted : {len(blocks)}")
    print(f"  Total characters  : {total_chars}")
    print(f"  Avg chars/block   : {avg_block_len:.0f}")
    print(f"  Empty blocks      : {len(empty_blocks)}  "
          f"{'⚠️  should be 0 — extraction.py should filter these' if empty_blocks else '✓'}")

    file_size_kb = file_path.stat().st_size / 1024
    chars_per_kb = total_chars / file_size_kb if file_size_kb else 0
    print(f"  File size         : {file_size_kb:.1f} KB")
    print(f"  Chars per KB      : {chars_per_kb:.0f}  "
          f"{'⚠️  suspiciously low — check for a scanned/image PDF' if chars_per_kb < 50 else ''}")

    print(f"\n  --- FIRST BLOCK (page/para {blocks[0]['page_num']}) ---")
    print(f"  {blocks[0]['text'][:500]}")
    if len(blocks[0]["text"]) > 500:
        print(f"  ...[{len(blocks[0]['text']) - 500} more chars]")

    print(f"\n  --- LAST BLOCK (page/para {blocks[-1]['page_num']}) ---")
    print(f"  {blocks[-1]['text'][:500]}")
    if len(blocks[-1]["text"]) > 500:
        print(f"  ...[{len(blocks[-1]['text']) - 500} more chars]")


def main() -> None:
    if not TEST_DOCS_DIR.exists():
        print(f"ERROR: {TEST_DOCS_DIR} does not exist. Create it and add test files.")
        return

    files = [f for f in TEST_DOCS_DIR.iterdir()
             if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        print(f"No supported files (.pdf/.docx/.txt) found in {TEST_DOCS_DIR}")
        return

    print(f"Found {len(files)} test file(s) in {TEST_DOCS_DIR}\n")

    for file_path in sorted(files):
        try:
            blocks = extract(str(file_path))
            summarize(blocks, file_path)
        except Exception as e:
            print(f"\n{'=' * 70}")
            print(f"FILE: {file_path.name}")
            print(f"{'=' * 70}")
            print(f"  ❌ EXTRACTION FAILED: {type(e).__name__}: {e}")

    print(f"\n{'=' * 70}")
    print("Done. Read every block above manually — do NOT skip this.")
    print("Look for: garbled tables, mid-sentence cuts, missing content")
    print("you know exists in the source file, and any ⚠️ flags above.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
