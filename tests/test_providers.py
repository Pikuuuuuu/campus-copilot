"""Provider layer tests: payload shape, response parsing, retries. No network calls."""
import numpy as np
import pytest

from core import config, providers
from core.ingest import Chunk
from core.vectorstore import LocalVectorStore, normalise


class FakeResponse:
    def __init__(self, status=200, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload


@pytest.fixture
def capture(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "body": json})
        return FakeResponse(payload=fake_post.payload)

    fake_post.payload = {}
    monkeypatch.setattr(providers.requests, "post", fake_post)
    return calls, fake_post


def test_gemini_payload_and_parsing(capture, monkeypatch):
    calls, fake = capture
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "llm_api_key", lambda: "k")
    fake.payload = {"candidates": [{"content": {"parts": [{"text": '{"status": "answered"}'}]}}]}

    out = providers.chat("sys", "user", "gemini-2.5-flash-lite", max_tokens=50)

    assert out == '{"status": "answered"}'
    body = calls[0]["body"]
    assert calls[0]["headers"]["x-goog-api-key"] == "k"
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["temperature"] == 0


def test_groq_uses_openai_shape(capture, monkeypatch):
    calls, fake = capture
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "llm_api_key", lambda: "gk")
    fake.payload = {"choices": [{"message": {"content": "hello"}}]}

    assert providers.chat("sys", "user", "llama-3.3-70b-versatile", json_mode=False) == "hello"
    assert calls[0]["url"].endswith("/chat/completions")
    assert calls[0]["headers"]["Authorization"] == "Bearer gk"
    assert calls[0]["body"]["messages"][0]["role"] == "system"
    assert "response_format" not in calls[0]["body"]


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(config, "llm_api_key", lambda: None)
    with pytest.raises(providers.LLMError, match="No API key"):
        providers.chat("s", "u", "m")


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "llm_api_key", lambda: "k")
    monkeypatch.setattr(providers.time, "sleep", lambda _s: None)
    responses = [
        FakeResponse(429, text="rate limited", headers={"Retry-After": "1"}),
        FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}),
    ]
    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: responses.pop(0))

    assert providers.chat("s", "u", "m", json_mode=False) == "ok"


def test_client_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "llm_api_key", lambda: "k")
    monkeypatch.setattr(providers.time, "sleep", lambda _s: None)
    attempts = []

    def fake_post(*a, **k):
        attempts.append(1)
        return FakeResponse(400, text="bad request")

    monkeypatch.setattr(providers.requests, "post", fake_post)
    with pytest.raises(providers.LLMError):
        providers.chat("s", "u", "m")
    assert len(attempts) == 1


def test_local_vector_store_ranks_and_filters(monkeypatch, tmp_path):
    chunks = [
        Chunk("a", "attendance rule", "regs.pdf", "Regs", "academics", 1, ""),
        Chunk("b", "hostel mess timing", "hall.pdf", "Hall", "hostel", 2, ""),
    ]
    matrix = normalise(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    store = LocalVectorStore(["a", "b"], matrix, chunks)
    monkeypatch.setattr("core.vectorstore.embed",
                        lambda texts, task="RETRIEVAL_QUERY", **kw: normalise(np.array([[0.9, 0.1]], dtype=np.float32)))

    assert store.search("attendance", k=2)[0][0].id == "a"
    assert [c.id for c, _ in store.search("attendance", k=2, category="hostel")] == ["b"]


def test_stale_vector_file_is_rejected(tmp_path):
    path = tmp_path / "vectors.npz"
    np.savez_compressed(path, ids=np.array(["old-id"]), matrix=np.zeros((1, 2), dtype=np.float32))
    chunks = [Chunk("new-id", "t", "s", "T", "general", 1, "")]
    assert LocalVectorStore.load(chunks, path) is None
