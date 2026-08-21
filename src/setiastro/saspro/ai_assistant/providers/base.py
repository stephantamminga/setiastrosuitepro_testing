# ai_assistant/providers/base.py
"""
Abstract base for AI providers — mirrors IAIProvider.cs from
michelebergo/nina.plugin.aiassistant.

Defines the dataclasses shared across all providers and the ABC that
every concrete provider must implement.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single turn (user or assistant) in a conversation."""
    role: str          # "user" | "assistant"
    content: str


@dataclass
class AIRequest:
    """Input to any provider's send_request()."""
    prompt: str
    system_prompt: Optional[str] = None
    history: List[ConversationTurn] = field(default_factory=list)
    temperature: float = 0.3
    max_tokens: int = 1024


@dataclass
class AIResponse:
    """Output from any provider's send_request()."""
    success: bool
    content: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: int = 0
    error: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider enum values (string keys used in QSettings)
# ---------------------------------------------------------------------------

class ProviderType:
    OPENAI = "openai"
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"

    ALL = [OPENAI, MISTRAL, ANTHROPIC, GOOGLE, OLLAMA]

    DISPLAY_NAMES = {
        OPENAI: "OpenAI",
        MISTRAL: "Mistral AI",
        ANTHROPIC: "Anthropic (Claude)",
        GOOGLE: "Google (Gemini)",
        OLLAMA: "Ollama (Local)",
    }


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class IAIProvider(abc.ABC):
    """
    Interface for AI providers.

    Mirrors IAIProvider.cs — every concrete provider must implement all
    abstract methods.  Non-abstract helpers are provided for convenience.
    """

    # ── identity ──────────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def provider_type(self) -> str:
        """One of ProviderType.* constants."""

    @property
    @abc.abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in UI."""

    @property
    @abc.abstractmethod
    def is_configured(self) -> bool:
        """True when the provider has been successfully initialised."""

    # ── lifecycle ─────────────────────────────────────────────────────────

    @abc.abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialise (or re-initialise) the provider from a config dict.

        Expected keys (all optional — provider decides what it needs):
          api_key, model_id, endpoint_override, temperature, max_tokens

        Returns True on success.
        """

    @abc.abstractmethod
    def update_model(self, model_id: Optional[str]) -> None:
        """
        Swap the active model without rebuilding the HTTP client.
        Mirrors UpdateModel() in the NINA plugin.
        """

    # ── core operations ───────────────────────────────────────────────────

    @abc.abstractmethod
    def send_request(self, request: AIRequest) -> AIResponse:
        """
        Send a request to the provider and return the response.

        Must *not* raise — errors should be returned via AIResponse.
        """

    @abc.abstractmethod
    def test_connection(self) -> bool:
        """Send a minimal probe request; return True if it succeeds."""

    @abc.abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Return a list of available model IDs.
        Falls back to a hardcoded default list if the API is unreachable.
        """
