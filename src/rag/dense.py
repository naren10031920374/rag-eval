import json
import sys
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.config import PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
META_PATH = PROCESSED_DIR / "embeddings_meta.json"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    def __init__(self) -> None:
        self.chunks = self._load_chunks()
        self.vectors = np.load(EMBEDDINGS_PATH).astype(np.float32)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))

        if len(self.chunks) != self.vectors.shape[0]:
            raise ValueError(
                f"Chunk count ({len(self.chunks)}) does not match vector count "
                f"({self.vectors.shape[0]}). Re-run build_embeddings."
            )

        self.model = SentenceTransformer(meta["model_name"])
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    @staticmethod
    def _load_chunks() -> list[dict]:
        chunks = []
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_vector = self.model.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True
        ).astype(np.float32)

        scores, ids = self.index.search(query_vector, k)

        results = []
        for rank, (score, chunk_index) in enumerate(zip(scores[0], ids[0]), start=1):
            if chunk_index == -1:
                continue
            chunk = self.chunks[chunk_index]
            results.append({"rank": rank, "score": float(score), **chunk})
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
    retriever = DenseRetriever()
    print(f"Loaded {len(retriever.chunks)} chunks in {time.perf_counter() - started:.1f}s\n")

    for query in queries:
        started = time.perf_counter()
        results = retriever.search(query, k=5)
        elapsed_ms = (time.perf_counter() - started) * 1000

        print("=" * 80)
        print(f"QUESTION: {query}   ({elapsed_ms:.0f} ms)")
        for result in results:
            print(f"\n#{result['rank']}  score {result['score']:.3f}  {result['chunk_id']}")
            print(f"    {result['title'][:80]}  (page {result['page']})")
            print(f"    {result['text'][:300]}...")
        print()


if __name__ == "__main__":
    main()