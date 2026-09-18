"""One thin HTTP layer for every chat model provider.

Why not an SDK: the four providers below all speak plain JSON over HTTPS, so a single
`requests` dependency keeps the deploy small and makes the model a config choice rather
than a code change. That matters here because the free tiers move: if Gemini's limits
change tomorrow, switching to Groq is one line in .env, and the eval set tells us what
that switch cost in quality.

Providers:
  gemini      Google AI Studio  - free tier, no card required   (default)
  groq        Groq Cloud        - free tier, no card required
  openrouter  OpenRouter        - has :free models
  anthropic   Claude            - paid
"""
from __future__ import annotations

import time

import requests

from core import config

TIMEOUT = 90
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def _post(url: str, headers: dict, payload: dict, tries: int = 4) -> dict:
    """POST with backoff. Free tiers are rate limited per minute, so 429 is expected, not exceptional."""
    delay = 5.0
    last = ""
    for attempt in range(tries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        except requests.RequestException as exc:
            last = str(exc)
        else:
            if resp.status_code == 200:
                return resp.json()
            last = f"{resp.status_code}: {resp.text[:300]}"
            if resp.status_code not in RETRY_STATUS:
                break
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = max(delay, float(retry_after))
        if attempt < tries - 1:
            time.sleep(delay)
            delay *= 2
    raise LLMError(f"{url.split('/')[2]} request failed - {last}")


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
def _gemini(system: str, user: str, model: str, max_tokens: int, json_mode: bool, key: str) -> str:
    generation = {"maxOutputTokens": max_tokens, "temperature": 0}
    if json_mode:
        generation["responseMimeType"] = "application/json"
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": key, "Content-Type": "application/json"},
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        },
    )
    candidates = data.get("candidates") or []
    if not candidates:
        raise LLMError(f"Gemini returned no candidates: {str(data)[:200]}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def _openai_compatible(system: str, user: str, model: str, max_tokens: int, json_mode: bool,
                       key: str, base_url: str) -> str:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = _post(f"{base_url}/chat/completions", {"Authorization": f"Bearer {key}"}, payload)
    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"No choices returned: {str(data)[:200]}")
    return (choices[0]["message"].get("content") or "").strip()


def _anthropic(system: str, user: str, model: str, max_tokens: int, json_mode: bool, key: str) -> str:
    data = _post(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
        {"model": model, "max_tokens": max_tokens, "temperature": 0,
         "system": system, "messages": [{"role": "user", "content": user}]},
    )
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()


def chat(system: str, user: str, model: str, max_tokens: int = 900, json_mode: bool = True) -> str:
    provider = (config.LLM_PROVIDER or "gemini").lower()
    key = config.llm_api_key()
    if not key:
        raise LLMError(f"No API key set for provider '{provider}'. Add it to .env or Streamlit secrets.")
    if provider == "gemini":
        return _gemini(system, user, model, max_tokens, json_mode, key)
    if provider == "groq":
        return _openai_compatible(system, user, model, max_tokens, json_mode, key,
                                  "https://api.groq.com/openai/v1")
    if provider == "openrouter":
        return _openai_compatible(system, user, model, max_tokens, json_mode, key,
                                  "https://openrouter.ai/api/v1")
    if provider == "anthropic":
        return _anthropic(system, user, model, max_tokens, json_mode, key)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use gemini, groq, openrouter or anthropic.")
