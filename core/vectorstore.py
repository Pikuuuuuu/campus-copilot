"""Semantic search without a paid vector database.

Embeddings come from Google's free embedding endpoint (no card required). The vectors
live in a NumPy file inside the repo, and search is a cosine similarity over a matrix.
For a corpus of a few thousand chunks this is fast, costs nothing to host, and removes
a whole external service from the deploy. Pinecone stays available behind the same
interface for when the corpus outgrows a single file.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import requests

from core import config
from core.ingest import Chunk
from core.providers import LLMError, _post

VECTORS_PATH = config.INDEX_DIR / "vectors.npz"


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT", batch_size: int = 50,
          pace_seconds: float = 1.0) -> np.ndarray:
    """Embed texts with Google's free embedding model. Paced to respect free-tier limits."""
    key = config.GEMINI_API_KEY
    if not key:
        raise LLMError("GEMINI_API_KEY (or GOOGLE_API_KEY) is needed for semantic search.")
    model = config.EMBED_MODEL
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents",
            {"x-goog-api-key": key, "Content-Type": "application/json"},
            {"requests": [
                {"model": f"models/{model}", "content": {"parts": [{"text": t[:8000]}]},
                 "taskType": task, "outputDimensionality": config.EMBED_DIM}
                for t in batch
            ]},
        )
        vectors.extend(item["values"] for item in data.get("embeddings", []))
        if start + batch_size < len(texts):
            time.sleep(pace_seconds)
    if len(vectors) != len(texts):
        raise LLMError(f"Embedding count mismatch: asked for {len(texts)}, got {len(vectors)}")
    return normalise(np.asarray(vectors, dtype=np.float32))


def normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


# --------------------------------------------------------------------------
# Local store
# --------------------------------------------------------------------------
class LocalVectorStore:
    """Cosine search over a NumPy matrix persisted next to the chunk store."""

    def __init__(self, ids: list[str], matrix: np.ndarray, chunks: list[Chunk] | None = None):
        self.ids = ids
        self.matrix = matrix
        self.by_id = {c.id: c for c in (chunks or [])}

    # -- persistence --
    @classmethod
    def build(cls, chunks: list[Chunk], path: Path = VECTORS_PATH) -> "LocalVectorStore":
        matrix = embed([f"{c.title}\n{c.text}" for c in chunks], task="RETRIEVAL_DOCUMENT")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=np.array([c.id for c in chunks]), matrix=matrix,
                            model=np.array([config.EMBED_MODEL]))
        return cls([c.id for c in chunks], matrix, chunks)

    @classmethod
    def load(cls, chunks: list[Chunk], path: Path = VECTORS_PATH) -> "LocalVectorStore | None":
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        ids = [str(i) for i in data["ids"]]
        store = cls(ids, data["matrix"], chunks)
        # An index built from older documents would cite the wrong pages: ignore it.
        if set(ids) - set(store.by_id):
            return None
        return store

    # -- search --
    def search(self, query: str, k: int, category: str | None = None) -> list[tuple[Chunk, float]]:
        if not self.ids:
            return []
        q = embed([query], task="RETRIEVAL_QUERY", pace_seconds=0)[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)
        out: list[tuple[Chunk, float]] = []
        for i in order:
            chunk = self.by_id.get(self.ids[int(i)])
            if chunk is None or (category and chunk.category != category):
                continue
            out.append((chunk, float(scores[int(i)])))
            if len(out) >= k:
                break
        return out


class PineconeStore:
    """Optional: same interface, hosted index. Only used when PINECONE_API_KEY is set."""

    BATCH = 90

    def __init__(self, chunks: list[Chunk] | None = None):
        from pinecone import Pinecone

        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.name = config.PINECONE_INDEX
        self._index = None
        self.by_id = {c.id: c for c in (chunks or [])}

    @property
    def index(self):
        if self._index is None:
            self._index = self.pc.Index(self.name)
        return self._index

    def ensure_index(self):
        from pinecone import ServerlessSpec

        if not self.pc.has_index(self.name):
            self.pc.create_index(name=self.name, dimension=config.EMBED_DIM, metric="cosine",
                                 spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION))
        return self.index

    def rebuild(self, chunks: list[Chunk]) -> int:
        index = self.ensure_index()
        try:
            index.delete(delete_all=True)
        except Exception:
            pass
        matrix = embed([f"{c.title}\n{c.text}" for c in chunks])
        records = [
            {"id": c.id, "values": vec.tolist(),
             "metadata": {"text": c.text, "source": c.source, "title": c.title,
                          "category": c.category, "page": c.page, "url": c.url}}
            for c, vec in zip(chunks, matrix)
        ]
        for start in range(0, len(records), 100):
            index.upsert(vectors=records[start : start + 100])
        return len(records)

    def search(self, query: str, k: int, category: str | None = None) -> list[tuple[Chunk, float]]:
        vector = embed([query], task="RETRIEVAL_QUERY", pace_seconds=0)[0].tolist()
        kwargs = {"vector": vector, "top_k": k, "include_metadata": True}
        if category:
            kwargs["filter"] = {"category": {"$eq": category}}
        res = self.index.query(**kwargs)
        matches = res["matches"] if isinstance(res, dict) else res.matches
        out = []
        for m in matches:
            md = m["metadata"] if isinstance(m, dict) else m.metadata
            mid = m["id"] if isinstance(m, dict) else m.id
            score = float(m["score"] if isinstance(m, dict) else m.score)
            chunk = self.by_id.get(mid) or Chunk(
                id=mid, text=md.get("text", ""), source=md.get("source", ""), title=md.get("title", ""),
                category=md.get("category", "general"), page=int(md.get("page", 1)), url=md.get("url", ""),
            )
            out.append((chunk, score))
        return out


def get_store(chunks: list[Chunk]):
    """Pick the vector backend from configuration. Returns None for keyword-only mode."""
    if not config.GEMINI_API_KEY:
        return None
    if config.PINECONE_API_KEY:
        try:
            return PineconeStore(chunks)
        except Exception:
            return None
    return LocalVectorStore.load(chunks)
