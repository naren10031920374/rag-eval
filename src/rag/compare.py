import sys

from src.rag.dense import DenseRetriever
from src.rag.sparse import SparseRetriever

DEFAULT_QUERIES = [
    "What is reciprocal rank fusion?",
    "How can a system find documents that mean the same thing but use different words?",
    "Why is a cross-encoder slower than a bi-encoder?",
    "How do you evaluate the faithfulness of generated answers?",
]


def short_label(result: dict) -> str:
    title = result["title"][:42]
    return f"{result['chunk_id']}  {title}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEFAULT_QUERIES

    print("Loading retrievers...")
    dense = DenseRetriever()
    sparse = SparseRetriever()

    for query in queries:
        dense_results = dense.search(query, k=5)
        sparse_results = sparse.search(query, k=5)

        dense_ids = {r["chunk_id"] for r in dense_results}
        sparse_ids = {r["chunk_id"] for r in sparse_results}
        overlap = dense_ids & sparse_ids

        print("\n" + "=" * 100)
        print(f"QUESTION: {query}")
        print(f"Chunks in both top-5 lists: {len(overlap)}")

        print("\n  DENSE (meaning):")
        for r in dense_results:
            mark = "*" if r["chunk_id"] in overlap else " "
            print(f"   {mark} #{r['rank']}  {r['score']:.3f}  {short_label(r)}")

        print("\n  BM25 (keywords):")
        for r in sparse_results:
            mark = "*" if r["chunk_id"] in overlap else " "
            print(f"   {mark} #{r['rank']}  {r['score']:.2f}  {short_label(r)}")

        print("\n  Top BM25 chunk text:")
        if sparse_results:
            print(f"    {sparse_results[0]['text'][:300]}...")
        print("  Top dense chunk text:")
        print(f"    {dense_results[0]['text'][:300]}...")


if __name__ == "__main__":
    main()