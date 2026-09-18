"""Hybrid retrieval: BM25 keyword search + Pinecone semantic search, fused with RRF.

Why hybrid?
- Students ask with exact tokens ("CGPA", "SGPA", "backlog", "Autumn 2026-27", clause numbers).
  Keyword search nails these; embeddings often blur them.
- Students also paraphrase ("can I skip classes if I'm sick") where semantic search wins.
- Reciprocal Rank Fusion merges both lists without having to calibrate their scores.

Keyword search needs no keys or accounts at all, so the app still works if the
semantic index is missing or the embedding quota runs out for the day.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Plus

from core import config
from core.ingest import Chunk

TOKEN_RE = re.compile(r"[\w\-\.]+", re.UNICODE)
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or",
    "what", "when", "where", "how", "who", "which", "do", "does", "did", "i", "my", "me", "can",
    "be", "it", "this", "that", "with", "as", "at", "by", "from", "if", "will", "should", "we",
}


def tokenize(text: str) -> list[str]:
    tokens = [t.strip(".-").lower() for t in TOKEN_RE.findall(text)]
    return [t for t in tokens if t and t not in STOPWORDS]


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused RRF score
    bm25_rank: int | None = None
    vector_rank: int | None = None
    vector_score: float | None = None


# --------------------------------------------------------------------------
# Keyword index
# --------------------------------------------------------------------------
class KeywordIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.corpus = [tokenize(f"{c.title} {c.text}") for c in chunks]
        self.token_sets = [set(doc) for doc in self.corpus]
        # BM25Plus keeps IDF positive, so rare-but-shared terms still score in small corpora.
        self.bm25 = BM25Plus(self.corpus) if chunks else None

    def search(self, query: str, k: int, category: str | None = None) -> list[tuple[Chunk, float]]:
        if not self.bm25:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        q_set = set(q_tokens)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if not (q_set & self.token_sets[i]):  # require at least one shared term
                continue
            if category and self.chunks[i].category != category:
                continue
            out.append((self.chunks[i], float(scores[i])))
            if len(out) >= k:
                break
        return out


# --------------------------------------------------------------------------
# Hybrid
# --------------------------------------------------------------------------
def reciprocal_rank_fusion(
    keyword: list[tuple[Chunk, float]],
    vector: list[tuple[Chunk, float]],
    k: int,
    rrf_k: int = 60,
) -> list[Hit]:
    hits: dict[str, Hit] = {}
    for rank, (chunk, _score) in enumerate(keyword, start=1):
        hit = hits.setdefault(chunk.id, Hit(chunk=chunk, score=0.0))
        hit.score += 1.0 / (rrf_k + rank)
        hit.bm25_rank = rank
    for rank, (chunk, score) in enumerate(vector, start=1):
        hit = hits.setdefault(chunk.id, Hit(chunk=chunk, score=0.0))
        hit.score += 1.0 / (rrf_k + rank)
        hit.vector_rank = rank
        hit.vector_score = score
    return sorted(hits.values(), key=lambda h: h.score, reverse=True)[:k]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], vector_store=None, use_vectors: bool | None = None):
        self.keyword = KeywordIndex(chunks)
        if vector_store is None and use_vectors is not False:
            from core.vectorstore import get_store

            vector_store = get_store(chunks)
        self.vector = vector_store
        self.vector_error: str | None = None

    @property
    def mode(self) -> str:
        if self.vector is None:
            return "keyword (BM25)"
        backend = "Pinecone" if type(self.vector).__name__ == "PineconeStore" else "local vectors"
        return f"hybrid (BM25 + {backend})"

    def search(self, query: str, k: int = config.TOP_K, category: str | None = None) -> list[Hit]:
        candidates = k * 2
        kw = self.keyword.search(query, candidates, category)
        vec: list[tuple[Chunk, float]] = []
        if self.vector:
            try:
                vec = self.vector.search(query, candidates, category)
                self.vector_error = None
            except Exception as exc:  # degrade gracefully to keyword-only
                self.vector_error = str(exc)
        return reciprocal_rank_fusion(kw, vec, k)
