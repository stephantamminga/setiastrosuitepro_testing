# ai_assistant/settings.py
"""
Persistent configuration for the AI assistant.

Uses QSettings (consistent with the rest of the app) under the
"ai_assistant" group key.  API keys are stored in plain text on disk
(same pattern as the app's existing credential storage) — users are
advised to use environment variables for sensitive deployments.
"""
from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import QSettings

from .providers.base import ProviderType


_GROUP = "ai_assistant"


class AISettings:
    """
    Thin wrapper around QSettings for AI assistant configuration.

    Usage:
        s = AISettings()
        s.provider = "mistral"
        s.api_key = "sk-..."
        config = s.to_provider_config()
    """

    def __init__(self, q_settings: Optional[QSettings] = None):
        self._qs = q_settings or QSettings("SetiaStro", "SetiaStroSuitePro")

    # ── helpers ───────────────────────────────────────────────────────────

    def _get(self, key: str, default=None):
        self._qs.beginGroup(_GROUP)
        try:
            return self._qs.value(key, default)
        finally:
            self._qs.endGroup()

    def _set(self, key: str, value):
        self._qs.beginGroup(_GROUP)
        try:
            self._qs.setValue(key, value)
        finally:
            self._qs.endGroup()

    # ── provider ──────────────────────────────────────────────────────────

    @property
    def provider(self) -> str:
        return self._get("provider", ProviderType.MISTRAL)

    @provider.setter
    def provider(self, value: str) -> None:
        self._set("provider", value)

    # ── model ─────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._get("model_id", "")

    @model_id.setter
    def model_id(self, value: str) -> None:
        self._set("model_id", value)

    # ── credentials ───────────────────────────────────────────────────────

    @property
    def api_key(self) -> str:
        """
        Return API key.  Preference order:
          1. Environment variable SASP_AI_API_KEY
          2. Stored QSettings value
        """
        return os.environ.get("SASP_AI_API_KEY") or self._get("api_key", "")

    @api_key.setter
    def api_key(self, value: str) -> None:
        self._set("api_key", value)

    @property
    def endpoint_override(self) -> str:
        return self._get("endpoint_override", "")

    @endpoint_override.setter
    def endpoint_override(self, value: str) -> None:
        self._set("endpoint_override", value)

    # ── generation params ─────────────────────────────────────────────────

    @property
    def temperature(self) -> float:
        raw = self._get("temperature", 0.3)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.3

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._set("temperature", float(value))

    @property
    def max_tokens(self) -> int:
        raw = self._get("max_tokens", 1024)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 1024

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self._set("max_tokens", int(value))

    # ── retrieval / grounding ─────────────────────────────────────────────

    @property
    def use_retrieval(self) -> bool:
        raw = self._get("use_retrieval", True)
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() not in ("false", "0", "no")

    @use_retrieval.setter
    def use_retrieval(self, value: bool) -> None:
        self._set("use_retrieval", value)

    @property
    def show_sources(self) -> bool:
        raw = self._get("show_sources", True)
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() not in ("false", "0", "no")

    @show_sources.setter
    def show_sources(self, value: bool) -> None:
        self._set("show_sources", value)

    @property
    def max_context_tokens(self) -> int:
        raw = self._get("max_context_tokens", 3000)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 3000

    @max_context_tokens.setter
    def max_context_tokens(self, value: int) -> None:
        self._set("max_context_tokens", int(value))

    @property
    def index_path(self) -> str:
        default = os.path.join(
            os.path.expanduser("~"), ".setiastro", "ai_index"
        )
        return self._get("index_path", default)

    @index_path.setter
    def index_path(self, value: str) -> None:
        self._set("index_path", value)

    # ── convenience ───────────────────────────────────────────────────────

    def to_provider_config(self) -> dict:
        """Build the config dict expected by IAIProvider.initialize()."""
        return {
            "api_key": self.api_key,
            "model_id": self.model_id or None,
            "endpoint_override": self.endpoint_override or None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def is_ready(self) -> bool:
        """True if at least provider + api_key are set (Ollama needs no key)."""
        if self.provider == ProviderType.OLLAMA:
            return True
        return bool(self.api_key)
