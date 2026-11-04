import json
import re
import unicodedata
from pathlib import Path

import pymupdf

from src.rag.config import PROJECT_ROOT

ARXIV_DIR = PROJECT_ROOT / "data" / "raw" / "arxiv"
PRIVATE_DIR = PROJECT_ROOT / "data" / "private"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pages.jsonl"
MIN_PAGE_CHARS = 100


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    pages = []
    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            pages.append((page_number, page.get_text("text")))
    return pages


def read_pages(path: Path) -> list[tuple[int, str]]:
    if path.suffix.lower() == ".pdf":
        return extract_pdf_pages(path)
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def load_arxiv_documents() -> list[dict]:
    manifest_path = ARXIV_DIR / "manifest.jsonl"
    documents = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            documents.append(
                {
                    "doc_id": record["arxiv_id"],
                    "source_type": "arxiv",
                    "title": record["title"],
                    "url": record["abs_url"],
                    "path": ARXIV_DIR / record["filename"],
                }
            )
    return documents


def load_private_documents() -> list[dict]:
    documents = []
    if not PRIVATE_DIR.exists():
        return documents
    for path in sorted(PRIVATE_DIR.rglob("*")):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        stem = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_").lower()
        documents.append(
            {
                "doc_id": f"course_{stem}",
                "source_type": "course",
                "title": path.stem,
                "url": None,
                "path": path,
            }
        )
    return documents


def main() -> None:
    documents = load_arxiv_documents() + load_private_documents()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    failed = []
    counts = {"arxiv": 0, "course": 0}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for doc in documents:
            try:
                raw_pages = read_pages(doc["path"])
            except Exception as error:
                print(f"FAILED {doc['doc_id']}: {error}")
                failed.append(doc["doc_id"])
                continue

            kept = 0
            for page_number, raw_text in raw_pages:
                text = clean_text(raw_text)
                if len(text) < MIN_PAGE_CHARS:
                    continue
                record = {
                    "doc_id": doc["doc_id"],
                    "source_type": doc["source_type"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "page": page_number,
                    "text": text,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

            counts[doc["source_type"]] += 1
            total_pages += kept
            print(f"{doc['doc_id']}: kept {kept}/{len(raw_pages)} pages")

    print(f"\nDocuments: {counts['arxiv']} arxiv, {counts['course']} course, {len(failed)} failed")
    print(f"Pages written: {total_pages}  ->  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()