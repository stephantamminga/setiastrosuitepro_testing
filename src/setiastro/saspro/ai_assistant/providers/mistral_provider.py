# ai_assistant/providers/mistral_provider.py
"""
Mistral AI provider — direct Python translation of MistralProvider.cs from
michelebergo/nina.plugin.aiassistant.

API docs: https://docs.mistral.ai/api/
OpenAI-compatible endpoint: https://api.mistral.ai/v1
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base import AIRequest, AIResponse, IAIProvider, ProviderType

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.mistral.ai/v1"

_DEFAULT_MODELS = [
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "open-mistral-7b",
    "open-mixtral-8x7b",
    "codestral-latest",
]


class MistralProvider(IAIProvider):
    """
    Provider for Mistral AI API (OpenAI-compatible chat/completions).
    Uses only stdlib urllib — no extra dependencies.
    """

    def __init__(self):
        self._api_key: Optional[str] = None
        self._model_id: str = "mistral-large-latest"
        self._timeout: int = 60
        self._initialized: bool = False

    # ── identity ──────────────────────────────────────────────────────────

    @property
    def provider_type(self) -> str:
        return ProviderType.MISTRAL

    @property
    def display_name(self) -> str:
        return "Mistral AI"

    @property
    def is_configured(self) -> bool:
        return self._initialized and bool(self._api_key)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def initialize(self, config: Dict[str, Any]) -> bool:
        api_key = config.get("api_key", "").strip()
        if not api_key:
            logger.error("Mistral: api_key is required")
            return False
        self._api_key = api_key
        self._model_id = config.get("model_id", "mistral-large-latest") or "mistral-large-latest"
        self._timeout = int(config.get("timeout", 60))
        self._initialized = True
        logger.info("Mistral AI provider initialised (model=%s)", self._model_id)
        return True

    def update_model(self, model_id: Optional[str]) -> None:
        if model_id:
            self._model_id = model_id

    # ── core operations ───────────────────────────────────────────────────

    def send_request(self, request: AIRequest) -> AIResponse:
        if not self.is_configured:
            return AIResponse(success=False, error="Mistral provider not initialised")

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
            raw = self._post(f"{_BASE_URL}/chat/completions", payload)
        except Exception as exc:
            logger.error("Mistral request failed: %s", exc)
            return AIResponse(success=False, error=str(exc))

        return self._parse_response(raw)

    def test_connection(self) -> bool:
        try:
            probe = AIRequest(prompt="Hello, confirm you are working.", max_tokens=10)
            resp = self.send_request(probe)
            return resp.success
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        if not self.is_configured:
            return list(_DEFAULT_MODELS)
        try:
            raw = self._get(f"{_BASE_URL}/models")
            data = json.loads(raw)
            models = [
                m["id"]
                for m in data.get("data", [])
                if "embed" not in m.get("id", "").lower()
                   and m.get("id")
            ]
            return sorted(models) if models else list(_DEFAULT_MODELS)
        except Exception as exc:
            logger.warning("Mistral: could not fetch model list (%s), using defaults", exc)
            return list(_DEFAULT_MODELS)

    # ── helpers ───────────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
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
                metadata={
                    "provider": "Mistral",
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                },
            )
        except Exception as exc:
            logger.error("Mistral: failed to parse response: %s\nRaw: %s", exc, raw[:500])
            return AIResponse(success=False, error=f"Parse error: {exc}")
