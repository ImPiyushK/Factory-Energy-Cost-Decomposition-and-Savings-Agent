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


class MockProvider(BaseProvider):
    """MOCK_LLM=true: the deterministic template is produced by the brief node itself."""

    name = "mock"

    def generate(self, system: str, user: str) -> LLMResult:  # pragma: no cover
        raise RuntimeError("mock provider must be handled by the brief node, not called")


class OpenAIProvider(BaseProvider):
    name = "openai"
    url = "https://api.openai.com/v1/chat/completions"

    def generate(self, system: str, user: str) -> LLMResult:
        import os
        r = httpx.post(self.url, timeout=45, headers={
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
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
        import os
        r = httpx.post(self.url, timeout=45, headers={
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
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
        import os
        key = os.environ.get("GEMINI_API_KEY", "")
        r = httpx.post(self.url.format(model=settings.llm_model) + f"?key={key}", timeout=45,
                       json={"system_instruction": {"parts": [{"text": system}]},
                             "contents": [{"role": "user", "parts": [{"text": user}]}],
                             "generationConfig": {"temperature": 0.0}})
        r.raise_for_status()
        d = r.json()
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
