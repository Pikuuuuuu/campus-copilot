"""Central configuration. Reads from environment variables, .env, or Streamlit secrets."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
DB_PATH = DATA_DIR / "app.db"
OFFICES_PATH = DATA_DIR / "offices.json"


def setting(name: str, default: str | None = None) -> str | None:
    """Env var first, then Streamlit secrets (for Streamlit Community Cloud)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


# --- Models ---------------------------------------------------------------
# Default setup costs nothing: Google AI Studio's free tier needs no credit card.
# Switch provider with one line in .env; the eval set shows what the switch costs.
LLM_PROVIDER = (setting("LLM_PROVIDER", "gemini") or "gemini").lower()

GEMINI_API_KEY = setting("GEMINI_API_KEY") or setting("GOOGLE_API_KEY")
GROQ_API_KEY = setting("GROQ_API_KEY")
OPENROUTER_API_KEY = setting("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = setting("ANTHROPIC_API_KEY")

DEFAULT_MODELS = {
    "gemini": ("gemini-2.5-flash-lite", "gemini-2.5-flash"),
    "groq": ("llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
    "openrouter": ("meta-llama/llama-3.3-70b-instruct:free", "meta-llama/llama-3.3-70b-instruct:free"),
    "anthropic": ("claude-haiku-4-5-20251001", "claude-sonnet-5"),
}
_answer_default, _judge_default = DEFAULT_MODELS.get(LLM_PROVIDER, DEFAULT_MODELS["gemini"])
ANSWER_MODEL = setting("ANSWER_MODEL", _answer_default)   # fast model answers students
JUDGE_MODEL = setting("JUDGE_MODEL", _judge_default)      # stronger model grades the eval set


def llm_api_key() -> str | None:
    return {
        "gemini": GEMINI_API_KEY,
        "groq": GROQ_API_KEY,
        "openrouter": OPENROUTER_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
    }.get(LLM_PROVIDER)


# --- Retrieval ------------------------------------------------------------
# Semantic search uses Google's free embedding endpoint and a local NumPy index.
# Pinecone is optional and only used when a key is present.
EMBED_MODEL = setting("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(setting("EMBED_DIM", "768"))

PINECONE_API_KEY = setting("PINECONE_API_KEY")
PINECONE_INDEX = setting("PINECONE_INDEX", "campus-copilot")
PINECONE_CLOUD = setting("PINECONE_CLOUD", "aws")
PINECONE_REGION = setting("PINECONE_REGION", "us-east-1")

CHUNK_CHARS = int(setting("CHUNK_CHARS", "900"))
CHUNK_OVERLAP = int(setting("CHUNK_OVERLAP", "150"))
TOP_K = int(setting("TOP_K", "6"))

# --- App ------------------------------------------------------------------
ADMIN_PASSWORD = setting("ADMIN_PASSWORD", "change-me")
MAX_QUESTION_CHARS = 1000

CATEGORIES = [
    "academics",
    "exams",
    "fees",
    "hostel",
    "placements",
    "scholarships",
    "general",
]
