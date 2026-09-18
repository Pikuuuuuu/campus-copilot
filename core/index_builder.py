"""Rebuild the keyword chunk store and, when a key is available, the semantic index."""
from __future__ import annotations

from core import config
from core.ingest import build_chunks, save_chunks
from core.vectorstore import LocalVectorStore, PineconeStore


def rebuild_index(with_vectors: bool = True) -> str:
    chunks = build_chunks()
    if not chunks:
        return "No documents found in data/raw/. Nothing indexed."
    save_chunks(chunks)
    docs = len({c.source for c in chunks})
    msg = f"Indexed {docs} documents into {len(chunks)} chunks (keyword index)."

    if not with_vectors:
        return msg
    if not config.GEMINI_API_KEY:
        return msg + " No GEMINI_API_KEY, so semantic search is off (keyword-only mode still works)."
    try:
        if config.PINECONE_API_KEY:
            n = PineconeStore(chunks).rebuild(chunks)
            return msg + f" Upserted {n} vectors to Pinecone index '{config.PINECONE_INDEX}'."
        LocalVectorStore.build(chunks)
        return msg + f" Embedded {len(chunks)} chunks into data/index/vectors.npz."
    except Exception as exc:
        return msg + f" Semantic index FAILED ({exc}). The app still runs in keyword-only mode."
