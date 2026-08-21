# ai_assistant/providers/google_provider.py
"""
Google Gemini provider — mirrors GoogleProvider.cs from
michelebergo/nina.plugin.aiassistant.

API docs: https://ai.google.dev/api/generate-content
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base import AIRequest, AIResponse, IAIProvider, ProviderType

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_DEFAULT_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]


class GoogleProvider(IAIProvider):
    """
    Provider for Google Gemini via REST (no SDK required).
    API key passed as query param ?key=...
    """

    def __init__(self):
        self._api_key: Optional[str] = None
        self._model_id: str = "gemini-2.0-flash"
        self._timeout: int = 60
        self._initialized: bool = False

    @property
    def provider_type(self) -> str:
        return ProviderType.GOOGLE

    @property
    def display_name(self) -> str:
        return "Google (Gemini)"

    @property
    def is_configured(self) -> bool:
        return self._initialized and bool(self._api_key)

    def initialize(self, config: Dict[str, Any]) -> bool:
        api_key = config.get("api_key", "").strip()
        if not api_key:
            logger.error("Google: api_key is required")
            return False
        self._api_key = api_key
        self._model_id = config.get("model_id", "gemini-2.0-flash") or "gemini-2.0-flash"
        self._timeout = int(config.get("timeout", 60))
        self._initialized = True
        logger.info("Google Gemini provider initialised (model=%s)", self._model_id)
        return True

    def update_model(self, model_id: Optional[str]) -> None:
        if model_id:
            self._model_id = model_id

    def send_request(self, request: AIRequest) -> AIResponse:
        if not self.is_configured:
            return AIResponse(success=False, error="Google provider not initialised")

        system_prompt = request.system_prompt or (
            "You are Seti Astro Suite Pro Help Assistant. "
            "Answer using only the provided repository context."
        )

        # Build Gemini contents array
        contents: List[Dict] = []
        for turn in (request.history or []):
            role = "user" if turn.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": turn.content}]})
        contents.append({"role": "user", "parts": [{"text": request.prompt}]})

        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }

        url = f"{_BASE_URL}/models/{self._model_id}:generateContent?key={self._api_key}"

        try:
            raw = self._post(url, payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error("Google HTTP %s: %s", exc.code, body[:500])
            return AIResponse(success=False, error=f"HTTP {exc.code}: {body[:200]}")
        except Exception as exc:
            logger.error("Google request failed: %s", exc)
            return AIResponse(success=False, error=str(exc))

        return self._parse_response(raw)

    def test_connection(self) -> bool:
        try:
            probe = AIRequest(prompt="Hello", max_tokens=10)
            return self.send_request(probe).success
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        if not self.is_configured:
            return list(_DEFAULT_MODELS)
        try:
            url = f"{_BASE_URL}/models?key={self._api_key}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            models = sorted(
                m["name"].split("/")[-1]
                for m in data.get("models", [])
                if "gemini" in m.get("name", "").lower()
                   and "generateContent" in m.get("supportedGenerationMethods", [])
            )
            return models if models else list(_DEFAULT_MODELS)
        except Exception as exc:
            logger.warning("Google: could not fetch model list (%s)", exc)
            return list(_DEFAULT_MODELS)

    def _post(self, url: str, payload: Dict) -> str:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return resp.read().decode("utf-8")

    def _parse_response(self, raw: str) -> AIResponse:
        try:
            data = json.loads(raw)
            candidates = data.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text = " ".join(p.get("text", "") for p in parts).strip()
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            completion_tokens = usage.get("candidatesTokenCount", 0)
            return AIResponse(
                success=True,
                content=text or None,
                model_used=self._model_id,
                tokens_used=prompt_tokens + completion_tokens,
                metadata={"provider": "Google", "input_tokens": prompt_tokens, "output_tokens": completion_tokens},
            )
        except Exception as exc:
            return AIResponse(success=False, error=f"Parse error: {exc}")
