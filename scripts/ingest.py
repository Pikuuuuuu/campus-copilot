"""CLI: python scripts/ingest.py   (chunks data/raw/, writes data/index/chunks.json, syncs Pinecone)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.index_builder import rebuild_index  # noqa: E402

if __name__ == "__main__":
    print(rebuild_index())
