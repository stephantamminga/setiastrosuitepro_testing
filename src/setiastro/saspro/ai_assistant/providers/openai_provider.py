# ai_assistant/providers/openai_provider.py
"""
OpenAI provider — supports OpenAI API and Azure OpenAI via endpoint override.
Mirrors OpenAIProvider.cs from michelebergo/nina.plugin.aiassistant.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base import AIRequest, AIResponse, IAIProvider, ProviderType

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"

_DEFAULT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]


class OpenAIProvider(IAIProvider):
    """
    Provider for OpenAI (and Azure OpenAI via endpoint override).
    Uses only stdlib urllib — no openai SDK required.
    """

    def __init__(self):
        self._api_key: Optional[str] = None
        self._model_id: str = "gpt-4o-mini"
        self._base_url: str = _DEFAULT_BASE_URL
        self._timeout: int = 60
        self._initialized: bool = False

    @property
    def provider_type(self) -> str:
        return ProviderType.OPENAI

    @property
    def display_name(self) -> str:
        return "OpenAI"

    @property
    def is_configured(self) -> bool:
        return self._initialized and bool(self._api_key)

    def initialize(self, config: Dict[str, Any]) -> bool:
        api_key = config.get("api_key", "").strip()
        if not api_key:
            logger.error("OpenAI: api_key is required")
            return False
        self._api_key = api_key
        self._model_id = config.get("model_id", "gpt-4o-mini") or "gpt-4o-mini"
        self._base_url = (config.get("endpoint_override") or _DEFAULT_BASE_URL).rstrip("/")
        self._timeout = int(config.get("timeout", 60))
        self._initialized = True
        logger.info("OpenAI provider initialised (model=%s, base=%s)", self._model_id, self._base_url)
        return True

    def update_model(self, model_id: Optional[str]) -> None:
        if model_id:
            self._model_id = model_id

    def send_request(self, request: AIRequest) -> AIResponse:
        if not self.is_configured:
            return AIResponse(success=False, error="OpenAI provider not initialised")

        messages: List[Dict[str, str]] = []

        system_prompt = request.system_prompt or (
            "You are Seti Astro Suite Pro Help Assistant. "
            "Answer using only the provided repository context."
        )
        messages.append({"role": "system", "content": system_prompt})

        for turn in (request.history or []):
            messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self._model_id,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        try:
            raw = self._post(f"{self._base_url}/chat/completions", payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error("OpenAI HTTP %s: %s", exc.code, body[:500])
            return AIResponse(success=False, error=f"HTTP {exc.code}: {body[:200]}")
        except Exception as exc:
            logger.error("OpenAI request failed: %s", exc)
            return AIResponse(success=False, error=str(exc))

        return self._parse_response(raw)

    def test_connection(self) -> bool:
        try:
            probe = AIRequest(prompt="Hello", max_tokens=5)
            return self.send_request(probe).success
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        if not self.is_configured:
            return list(_DEFAULT_MODELS)
        try:
            raw = self._get(f"{self._base_url}/models")
            data = json.loads(raw)
            models = sorted(
                m["id"]
                for m in data.get("data", [])
                if m.get("id", "").startswith("gpt")
            )
            return models if models else list(_DEFAULT_MODELS)
        except Exception as exc:
            logger.warning("OpenAI: could not fetch model list (%s)", exc)
            return list(_DEFAULT_MODELS)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"******",
            "Content-Type": "application/json",
        }

    def _post(self, url: str, payload: Dict) -> str:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return resp.read().decode("utf-8")

    def _get(self, url: str) -> str:
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return resp.read().decode("utf-8")

    def _parse_response(self, raw: str) -> AIResponse:
        try:
            data = json.loads(raw)
            choices = data.get("choices", [])
            content = choices[0]["message"]["content"] if choices else None
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            return AIResponse(
                success=True,
                content=content,
                model_used=data.get("model", self._model_id),
                tokens_used=prompt_tokens + completion_tokens,
                metadata={"provider": "OpenAI", "input_tokens": prompt_tokens, "output_tokens": completion_tokens},
            )
        except Exception as exc:
            return AIResponse(success=False, error=f"Parse error: {exc}")
