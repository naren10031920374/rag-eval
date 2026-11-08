import sys
import time

from src.rag.dense import DenseRetriever
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.sparse import SparseRetriever


class HybridRetriever:
    def __init__(self) -> None:
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in self.dense.chunks}

    def search(
        self, query: str, k: int = 5, candidates: int = 50, rrf_k: int = 60
    ) -> list[dict]:
        dense_results = self.dense.search(query, k=candidates)
        sparse_results = self.sparse.search(query, k=candidates)

        dense_ranks = {r["chunk_id"]: r["rank"] for r in dense_results}
        sparse_ranks = {r["chunk_id"]: r["rank"] for r in sparse_results}

        fused = reciprocal_rank_fusion(
            [
                [r["chunk_id"] for r in dense_results],
                [r["chunk_id"] for r in sparse_results],
            ],
            k=rrf_k,
        )

        results = []
        for rank, (chunk_id, score) in enumerate(fused[:k], start=1):
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "dense_rank": dense_ranks.get(chunk_id),
                    "sparse_rank": sparse_ranks.get(chunk_id),
                    **self.chunks_by_id[chunk_id],
                }
            )
        return results


def where(rank: int | None) -> str:
    return f"#{rank}" if rank is not None else "-"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = [
            "What is reciprocal rank fusion?",
            "How can a system find documents that mean the same thing but use different words?",
            "Why is a cross-encoder slower than a bi-encoder?",
            "How do you evaluate the faithfulness of generated answers?",
        ]

    started = time.perf_counter()
    retriever = HybridRetriever()
    print(f"Loaded in {time.perf_counter() - started:.1f}s\n")

    for query in queries:
        started = time.perf_counter()
        results = retriever.search(query, k=5)
        elapsed_ms = (time.perf_counter() - started) * 1000

        print("=" * 90)
        print(f"QUESTION: {query}   ({elapsed_ms:.0f} ms)")
        for result in results:
            print(
                f"\n#{result['rank']}  rrf {result['score']:.4f}   "
                f"dense {where(result['dense_rank'])}   bm25 {where(result['sparse_rank'])}   "
                f"{result['chunk_id']}"
            )
            print(f"    {result['title'][:80]}  (page {result['page']})")
            print(f"    {result['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
    