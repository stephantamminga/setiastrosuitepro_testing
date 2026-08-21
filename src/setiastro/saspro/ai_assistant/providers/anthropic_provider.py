# ai_assistant/providers/anthropic_provider.py
"""
Anthropic (Claude) provider — mirrors AnthropicProvider.cs from
michelebergo/nina.plugin.aiassistant.

API docs: https://docs.anthropic.com/en/api/messages
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base import AIRequest, AIResponse, IAIProvider, ProviderType

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"

_DEFAULT_MODELS = [
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
]


class AnthropicProvider(IAIProvider):
    """
    Provider for Anthropic Claude API.
    Uses x-api-key + anthropic-version headers (not Bearer).
    """

    def __init__(self):
        self._api_key: Optional[str] = None
        self._model_id: str = "claude-3-5-sonnet-20241022"
        self._timeout: int = 60
        self._initialized: bool = False

    @property
    def provider_type(self) -> str:
        return ProviderType.ANTHROPIC

    @property
    def display_name(self) -> str:
        return "Anthropic (Claude)"

    @property
    def is_configured(self) -> bool:
        return self._initialized and bool(self._api_key)

    def initialize(self, config: Dict[str, Any]) -> bool:
        api_key = config.get("api_key", "").strip()
        if not api_key:
            logger.error("Anthropic: api_key is required")
            return False
        self._api_key = api_key
        self._model_id = config.get("model_id", "claude-3-5-sonnet-20241022") or "claude-3-5-sonnet-20241022"
        self._timeout = int(config.get("timeout", 60))
        self._initialized = True
        logger.info("Anthropic provider initialised (model=%s)", self._model_id)
        return True

    def update_model(self, model_id: Optional[str]) -> None:
        if model_id:
            self._model_id = model_id

    def send_request(self, request: AIRequest) -> AIResponse:
        if not self.is_configured:
            return AIResponse(success=False, error="Anthropic provider not initialised")

        system_prompt = request.system_prompt or (
            "You are Seti Astro Suite Pro Help Assistant. "
            "Answer using only the provided repository context."
        )

        # Anthropic messages API separates system from messages
        messages: List[Dict[str, str]] = []
        for turn in (request.history or []):
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self._model_id,
            "max_tokens": request.max_tokens,
            "system": system_prompt,
            "messages": messages,
        }

        try:
            raw = self._post(f"{_BASE_URL}/messages", payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error("Anthropic HTTP %s: %s", exc.code, body[:500])
            return AIResponse(success=False, error=f"HTTP {exc.code}: {body[:200]}")
        except Exception as exc:
            logger.error("Anthropic request failed: %s", exc)
            return AIResponse(success=False, error=str(exc))

        return self._parse_response(raw)

    def test_connection(self) -> bool:
        try:
            probe = AIRequest(prompt="Hello", max_tokens=10)
            return self.send_request(probe).success
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        # Anthropic does not expose a public /models endpoint in all tiers;
        # return the well-known list.
        return list(_DEFAULT_MODELS)

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self._api_key or "",
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _post(self, url: str, payload: Dict) -> str:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return resp.read().decode("utf-8")

    def _parse_response(self, raw: str) -> AIResponse:
        try:
            data = json.loads(raw)
            # Anthropic returns {"content": [{"type": "text", "text": "..."}], ...}
            content_blocks = data.get("content", [])
            text = " ".join(
                block.get("text", "") for block in content_blocks if block.get("type") == "text"
            ).strip()
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            return AIResponse(
                success=True,
                content=text or None,
                model_used=data.get("model", self._model_id),
                tokens_used=input_tokens + output_tokens,
                metadata={"provider": "Anthropic", "input_tokens": input_tokens, "output_tokens": output_tokens},
            )
        except Exception as exc:
            return AIResponse(success=False, error=f"Parse error: {exc}")
