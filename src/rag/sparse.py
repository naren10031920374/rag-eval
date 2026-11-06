import json
import re
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.config import PROJECT_ROOT

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class SparseRetriever:
    def __init__(self) -> None:
        self.chunks = self._load_chunks()
        tokenized_chunks = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    @staticmethod
    def _load_chunks() -> list[dict]:
        chunks = []
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_ids = np.argsort(-scores)[:k]

        results = []
        for rank, chunk_index in enumerate(top_ids, start=1):
            score = float(scores[chunk_index])
            if score <= 0:
                break
            results.append({"rank": rank, "score": score, **self.chunks[int(chunk_index)]})
        return results


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = [
            "How does a cross-encoder reranker differ from a bi-encoder?",
            "How do you measure the faithfulness of RAG answers?",
            "Why combine BM25 with dense retrieval?",
        ]

    started = time.perf_counter()
    retriever = SparseRetriever()
    print(f"Indexed {len(retriever.chunks)} chunks in {time.perf_counter() - started:.1f}s\n")

    for query in queries:
        started = time.perf_counter()
        results = retriever.search(query, k=5)
        elapsed_ms = (time.perf_counter() - started) * 1000

        print("=" * 80)
        print(f"QUESTION: {query}   ({elapsed_ms:.0f} ms)")
        for result in results:
            print(f"\n#{result['rank']}  score {result['score']:.2f}  {result['chunk_id']}")
            print(f"    {result['title'][:80]}  (page {result['page']})")
            print(f"    {result['text'][:300]}...")
        print()


if __name__ == "__main__":
    main()