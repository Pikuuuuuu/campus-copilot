"""Document ingestion: load PDFs / text files, clean them, and split into cited chunks.

Folder convention:  data/raw/<category>/<file>.pdf
The folder name becomes the chunk's category (used for filtering and escalation).
Optional data/raw/sources.json maps file names to official URLs and titles so
every citation can link back to the original notice.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from core import config

SUPPORTED = {".pdf", ".txt", ".md"}


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # file name
    title: str
    category: str
    page: int
    url: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for a file. Text files count as one page."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def clean_text(text: str) -> str:
    """Normalise whitespace and strip the repeated PDF furniture that hurts retrieval."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"Page \d+ of \d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)  # keep paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # join hard-wrapped lines
    return text.strip()


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def split_text(text: str, size: int = config.CHUNK_CHARS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware splitter: packs paragraphs up to `size` chars, with overlap.

    Rules and regulations are written in numbered clauses, so breaking on paragraph
    boundaries keeps a clause and its conditions together far more often than
    fixed-width slicing does.
    """
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:  # hard-split any single paragraph longer than size
        if len(para) <= size:
            pieces.append(para)
            continue
        sentences = re.split(r"(?<=[.;:])\s+", para)
        buf = ""
        for s in sentences:
            while len(s) > size:  # pathological: no punctuation at all
                pieces.append(s[:size])
                s = s[size - overlap :]
            if len(buf) + len(s) + 1 > size and buf:
                pieces.append(buf.strip())
                buf = ""
            buf += " " + s
        if buf.strip():
            pieces.append(buf.strip())

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) + 2 > size and current:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # start the overlap at a word boundary
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            current = tail + "\n\n" + piece if len(tail) + len(piece) + 2 <= size else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _chunk_id(source: str, page: int, idx: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{page}|{idx}|{text[:64]}".encode()).hexdigest()[:16]
    return f"{Path(source).stem[:40]}-p{page}-{idx}-{digest}"


def load_sources_manifest(raw_dir: Path) -> dict:
    manifest = raw_dir / "sources.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    return {}


def chunk_file(path: Path, category: str, manifest: dict | None = None) -> list[Chunk]:
    manifest = manifest or {}
    meta = manifest.get(path.name, {})
    title = meta.get("title") or path.stem.replace("_", " ").replace("-", " ").title()
    url = meta.get("url", "")
    chunks: list[Chunk] = []
    for page_no, raw in load_pages(path):
        for idx, piece in enumerate(split_text(clean_text(raw))):
            if len(piece) < 40:  # skip headers / stray page numbers
                continue
            chunks.append(
                Chunk(
                    id=_chunk_id(path.name, page_no, idx, piece),
                    text=piece,
                    source=path.name,
                    title=title,
                    category=category,
                    page=page_no,
                    url=url,
                )
            )
    return chunks


def build_chunks(raw_dir: Path = config.RAW_DIR) -> list[Chunk]:
    manifest = load_sources_manifest(raw_dir)
    all_chunks: list[Chunk] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED or path.name == "README.md":
            continue
        rel = path.relative_to(raw_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "general"
        if category not in config.CATEGORIES:
            category = "general"
        all_chunks.extend(chunk_file(path, category, manifest))
    return all_chunks


def save_chunks(chunks: list[Chunk], path: Path = config.CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_chars": config.CHUNK_CHARS,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "chunks": [c.to_dict() for c in chunks],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_chunks(path: Path = config.CHUNKS_PATH) -> tuple[list[Chunk], dict]:
    if not path.exists():
        return [], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in payload.get("chunks", [])]
    meta = {k: v for k, v in payload.items() if k != "chunks"}
    return chunks, meta
