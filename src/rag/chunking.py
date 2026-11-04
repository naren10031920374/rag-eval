import json

from src.rag.config import PROJECT_ROOT

PAGES_PATH = PROJECT_ROOT / "data" / "processed" / "pages.jsonl"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

CHUNK_WORDS = 180
OVERLAP_WORDS = 40


def chunk_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    step = size - overlap
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(words[start:end])
        if end == len(words):
            break
        start += step
    return chunks


def main() -> None:
    chunk_count = 0
    word_counts = []
    by_source = {}

    with open(PAGES_PATH, encoding="utf-8") as pages, open(
        CHUNKS_PATH, "w", encoding="utf-8"
    ) as out:
        for line in pages:
            page = json.loads(line)
            words = page["text"].split()

            for index, piece in enumerate(chunk_words(words, CHUNK_WORDS, OVERLAP_WORDS)):
                record = {
                    "chunk_id": f"{page['doc_id']}::p{page['page']}::c{index}",
                    "doc_id": page["doc_id"],
                    "source_type": page["source_type"],
                    "title": page["title"],
                    "url": page["url"],
                    "page": page["page"],
                    "chunk_index": index,
                    "n_words": len(piece),
                    "text": " ".join(piece),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_count += 1
                word_counts.append(len(piece))
                by_source[page["source_type"]] = by_source.get(page["source_type"], 0) + 1

    average = sum(word_counts) / len(word_counts)
    print(f"Chunks written: {chunk_count}  ->  {CHUNKS_PATH}")
    print(f"By source: {by_source}")
    print(f"Words per chunk: avg {average:.0f}, min {min(word_counts)}, max {max(word_counts)}")


if __name__ == "__main__":
    main()