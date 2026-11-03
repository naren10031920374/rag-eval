import json
import re
import time

import arxiv
import requests

from src.rag.config import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "arxiv"
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"

QUERIES = [
    'all:"retrieval augmented generation"',
    'all:"dense passage retrieval"',
    'all:"hybrid search" AND all:"BM25"',
    'all:"cross-encoder" AND all:"reranking"',
    'all:"RAG" AND all:"evaluation" AND all:"faithfulness"',
]
PAPERS_PER_QUERY = 10
DELAY_SECONDS = 3


def safe_filename(arxiv_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", arxiv_id) + ".pdf"


def load_seen_ids() -> set[str]:
    seen = set()
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            for line in f:
                seen.add(json.loads(line)["arxiv_id"])
    return seen


def download_pdf(pdf_url: str, destination) -> None:
    url = pdf_url.replace("http://", "https://")
    response = requests.get(url, timeout=60, headers={"User-Agent": "rag-eval-student-project"})
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("response was not a PDF")
    destination.write_bytes(response.content)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=PAPERS_PER_QUERY, delay_seconds=DELAY_SECONDS, num_retries=3)
    seen = load_seen_ids()
    downloaded = 0

    with open(MANIFEST_PATH, "a", encoding="utf-8") as manifest:
        for query in QUERIES:
            print(f"\nQuery: {query}")
            search = arxiv.Search(
                query=query,
                max_results=PAPERS_PER_QUERY,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for paper in client.results(search):
                arxiv_id = paper.get_short_id()
                if arxiv_id in seen:
                    print(f"  skip (already have) {arxiv_id}")
                    continue

                filename = safe_filename(arxiv_id)
                try:
                    download_pdf(paper.pdf_url, RAW_DIR / filename)
                except Exception as error:
                    print(f"  FAILED {arxiv_id}: {error}")
                    continue

                record = {
                    "arxiv_id": arxiv_id,
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "published": paper.published.date().isoformat(),
                    "abs_url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "filename": filename,
                    "found_by_query": query,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                seen.add(arxiv_id)
                downloaded += 1
                print(f"  OK {arxiv_id}  {paper.title[:70]}")
                time.sleep(DELAY_SECONDS)

    print(f"\nDone. Downloaded {downloaded} new papers. Total in manifest: {len(seen)}")


if __name__ == "__main__":
    main()