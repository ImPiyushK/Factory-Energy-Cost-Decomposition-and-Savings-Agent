"""LLM provider abstraction (mandatory switching via LLM_PROVIDER / LLM_MODEL).

Providers: mock (template brief, no network), openai, anthropic, google, ollama.
Every hosted call is optional-at-runtime: transport errors raise and the caller
(the brief node) falls back to the deterministic template brief.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config import settings


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 1
    provider: str = ""
    model: str = ""
    error: str = ""


class BaseProvider:
    name = "base"

    def generate(self, system: str, user: str) -> LLMResult:
        raise NotImplementedError

    def _with_backoff(self, url: str, *, headers: dict | None = None, timeout: float,
                      payload: dict) -> dict:
        """POST with exponential backoff on 429/5xx (rate limits are real in demos)."""
        import time
        last_error = ""
        for attempt, delay in enumerate((0.0, 8.0, 20.0, 40.0)):
            if delay:
                time.sleep(delay)
            try:
                r = httpx.post(url, timeout=timeout, headers=headers or {}, json=payload)
            except httpx.HTTPStatusError as e:
                last_error = str(e)
                if e.response is not None and (e.response.status_code in (429,) or
                                               e.response.status_code >= 500):
                    continue
                raise
            if r.status_code in (429,) or r.status_code >= 500:
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"LLM provider unavailable after retries: {last_error}")


class MockProvider(BaseProvider):
    """MOCK_LLM=true: the deterministic template is produced by the brief node itself."""

    name = "mock"

    def generate(self, system: str, user: str) -> LLMResult:  # pragma: no cover
        raise RuntimeError("mock provider must be handled by the brief node, not called")


class OpenAIProvider(BaseProvider):
    name = "openai"
    url = "https://api.openai.com/v1/chat/completions"

    def generate(self, system: str, user: str) -> LLMResult:
        r = httpx.post(self.url, timeout=60, headers={
            "Authorization": f"Bearer {settings.openai_api_key}"},
            json={"model": settings.llm_model, "temperature": 0.0,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]})
        r.raise_for_status()
        d = r.json()
        msg = d["choices"][0]["message"]["content"]
        u = d.get("usage", {}) or {}
        return LLMResult(msg, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                         provider=self.name, model=settings.llm_model)


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def generate(self, system: str, user: str) -> LLMResult:
        r = httpx.post(self.url, timeout=60, headers={
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01"},
            json={"model": settings.llm_model, "max_tokens": 500, "temperature": 0.0,
                  "system": system, "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        d = r.json()
        msg = d["content"][0]["text"]
        u = d.get("usage", {}) or {}
        return LLMResult(msg, u.get("input_tokens", 0), u.get("output_tokens", 0),
                         provider=self.name, model=settings.llm_model)


class GoogleProvider(BaseProvider):
    name = "google"
    url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def generate(self, system: str, user: str) -> LLMResult:
        d = self._with_backoff(
            self.url.format(model=settings.llm_model) + f"?key={settings.gemini_api_key}",
            timeout=60,
            payload={"system_instruction": {"parts": [{"text": system}]},
                     "contents": [{"role": "user", "parts": [{"text": user}]}],
                     "generationConfig": {"temperature": 0.0}})
        msg = d["candidates"][0]["content"]["parts"][0]["text"]
        u = (d.get("usageMetadata", {}) or {})
        return LLMResult(msg, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0),
                         provider=self.name, model=settings.llm_model)


class OllamaProvider(BaseProvider):
    name = "ollama"

    def generate(self, system: str, user: str) -> LLMResult:
        r = httpx.post(f"{settings.ollama_base_url.rstrip('/')}/api/chat", timeout=120,
                       json={"model": settings.llm_model or "llama3.1:8b", "stream": False,
                             "options": {"temperature": 0.0},
                             "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": user}]})
        r.raise_for_status()
        d = r.json()
        msg = d["message"]["content"]
        return LLMResult(msg, d.get("prompt_eval_count", 0), d.get("eval_count", 0),
                         provider=self.name, model=settings.llm_model)


PROVIDERS: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider, "anthropic": AnthropicProvider,
    "google": GoogleProvider, "ollama": OllamaProvider, "mock": MockProvider,
}


def get_provider() -> BaseProvider:
    name = "mock" if settings.mock_llm else (settings.llm_provider or "mock").lower()
    cls = PROVIDERS.get(name, MockProvider)
    return cls()


def llm_reachable() -> dict:
    """Cheap readiness check for /health."""
    name = (settings.llm_provider or "mock").lower()
    if name == "mock" or settings.mock_llm:
        return {"reachable": True, "mode": "mock/template"}
    try:
        if name == "ollama":
            r = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=5)
            return {"reachable": r.status_code == 200, "mode": "local provider"}
        return {"reachable": True, "mode": f"{name} (network; checked at call time)"},
    except Exception as e:  # pragma: no cover
        return {"reachable": False, "error": str(e)}
