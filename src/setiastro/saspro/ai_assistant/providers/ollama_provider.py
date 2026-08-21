# ai_assistant/providers/ollama_provider.py
"""
Ollama local provider — mirrors OllamaProvider.cs from
michelebergo/nina.plugin.aiassistant.

Talks to a locally running Ollama server (http://localhost:11434).
No API key required.

API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base import AIRequest, AIResponse, IAIProvider, ProviderType

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"

_DEFAULT_MODELS = [
    "llama3.2",
    "llama3.1",
    "mistral",
    "gemma3",
    "phi4",
]


class OllamaProvider(IAIProvider):
    """
    Provider for locally running Ollama server.
    Uses the /api/chat endpoint (OpenAI-like messages format).
    """

    def __init__(self):
        self._base_url: str = _DEFAULT_BASE_URL
        self._model_id: str = "llama3.2"
        self._timeout: int = 120
        self._initialized: bool = False

    @property
    def provider_type(self) -> str:
        return ProviderType.OLLAMA

    @property
    def display_name(self) -> str:
        return "Ollama (Local)"

    @property
    def is_configured(self) -> bool:
        return self._initialized

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._base_url = (config.get("endpoint_override") or _DEFAULT_BASE_URL).rstrip("/")
        self._model_id = config.get("model_id", "llama3.2") or "llama3.2"
        self._timeout = int(config.get("timeout", 120))
        self._initialized = True
        logger.info("Ollama provider initialised (url=%s, model=%s)", self._base_url, self._model_id)
        return True

    def update_model(self, model_id: Optional[str]) -> None:
        if model_id:
            self._model_id = model_id

    def send_request(self, request: AIRequest) -> AIResponse:
        if not self.is_configured:
            return AIResponse(success=False, error="Ollama provider not initialised")

        system_prompt = request.system_prompt or (
            "You are Seti Astro Suite Pro Help Assistant. "
            "Answer using only the provided repository context."
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for turn in (request.history or []):
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self._model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        try:
            raw = self._post(f"{self._base_url}/api/chat", payload)
        except urllib.error.URLError as exc:
            msg = f"Cannot reach Ollama at {self._base_url}. Is it running? ({exc.reason})"
            logger.error(msg)
            return AIResponse(success=False, error=msg)
        except Exception as exc:
            logger.error("Ollama request failed: %s", exc)
            return AIResponse(success=False, error=str(exc))

        return self._parse_response(raw)

    def test_connection(self) -> bool:
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                pass
            return True
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            models = sorted(m["name"] for m in data.get("models", []) if m.get("name"))
            return models if models else list(_DEFAULT_MODELS)
        except Exception as exc:
            logger.warning("Ollama: could not fetch model list (%s)", exc)
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
            # Ollama /api/chat returns {"message": {"role": "assistant", "content": "..."}}
            message = data.get("message", {})
            content = message.get("content", "").strip() or None
            eval_count = data.get("eval_count", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            return AIResponse(
                success=True,
                content=content,
                model_used=data.get("model", self._model_id),
                tokens_used=prompt_eval_count + eval_count,
                metadata={
                    "provider": "Ollama",
                    "input_tokens": prompt_eval_count,
                    "output_tokens": eval_count,
                },
            )
        except Exception as exc:
            return AIResponse(success=False, error=f"Parse error: {exc}")
