import sys
import time

from sentence_transformers import CrossEncoder

from src.rag.hybrid import HybridRetriever

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = RERANKER_NAME) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], k: int = 5) -> list[dict]:
        if not candidates:
            return []

        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        order = sorted(range(len(candidates)), key=lambda i: -scores[i])

        results = []
        for new_rank, index in enumerate(order[:k], start=1):
            candidate = candidates[index]
            results.append(
                {
                    **candidate,
                    "hybrid_rank": candidate["rank"],
                    "rrf_score": candidate["score"],
                    "rank": new_rank,
                    "score": float(scores[index]),
                }
            )
        return results


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

    hybrid = HybridRetriever()
    reranker = Reranker()
    reranker.model.predict([("warm up", "warm up")])

    for query in queries:
        candidates = hybrid.search(query, k=50)

        started = time.perf_counter()
        reranked = reranker.rerank(query, candidates, k=5)
        rerank_ms = (time.perf_counter() - started) * 1000

        hybrid_top = {c["chunk_id"] for c in candidates[:5]}
        reranked_top = {c["chunk_id"] for c in reranked}

        print("=" * 90)
        print(f"QUESTION: {query}")
        print(f"Reranking 50 candidates took {rerank_ms:.0f} ms")
        print(f"Chunks in both the hybrid top 5 and the reranked top 5: {len(hybrid_top & reranked_top)}")

        print("\n  HYBRID top 5 (before reranking):")
        for c in candidates[:5]:
            print(f"    #{c['rank']}  rrf {c['score']:.4f}  {c['chunk_id']}  {c['title'][:40]}")

        print("\n  RERANKED top 5:")
        for c in reranked:
            print(
                f"\n    #{c['rank']}  score {c['score']:.2f}   was hybrid #{c['hybrid_rank']}   "
                f"{c['chunk_id']}"
            )
            print(f"        {c['title'][:80]}  (page {c['page']})")
            print(f"        {c['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()