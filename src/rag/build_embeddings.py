import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.config import PROJECT_ROOT

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
META_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_meta.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32


def load_chunk_texts() -> list[str]:
    texts = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def main() -> None:
    texts = load_chunk_texts()
    print(f"Loaded {len(texts)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    started = time.time()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    elapsed = time.time() - started

    vectors = vectors.astype(np.float32)
    np.save(EMBEDDINGS_PATH, vectors)

    meta = {
        "model_name": MODEL_NAME,
        "n_chunks": int(vectors.shape[0]),
        "dimensions": int(vectors.shape[1]),
        "normalized": True,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size_mb = EMBEDDINGS_PATH.stat().st_size / (1024 * 1024)
    lengths = np.linalg.norm(vectors, axis=1)
    print(f"\nShape: {vectors.shape}")
    print(f"Time: {elapsed:.0f} seconds")
    print(f"File size: {size_mb:.1f} MB  ->  {EMBEDDINGS_PATH}")
    print(f"Vector lengths: min {lengths.min():.4f}, max {lengths.max():.4f}")


if __name__ == "__main__":
    main()