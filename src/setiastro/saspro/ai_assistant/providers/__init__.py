# ai_assistant/providers/__init__.py
"""
AI provider registry.

Usage:
    from setiastro.saspro.ai_assistant.providers import get_provider, PROVIDER_CLASSES

    provider = get_provider("mistral")
    provider.initialize({"api_key": "..."})
"""
from __future__ import annotations

from typing import Dict, Type

from .base import IAIProvider, ProviderType, AIRequest, AIResponse, ConversationTurn
from .mistral_provider import MistralProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider
from .ollama_provider import OllamaProvider

PROVIDER_CLASSES: Dict[str, Type[IAIProvider]] = {
    ProviderType.OPENAI: OpenAIProvider,
    ProviderType.MISTRAL: MistralProvider,
    ProviderType.ANTHROPIC: AnthropicProvider,
    ProviderType.GOOGLE: GoogleProvider,
    ProviderType.OLLAMA: OllamaProvider,
}


def get_provider(provider_type: str) -> IAIProvider:
    """Return a new (uninitialised) provider instance for the given type key."""
    cls = PROVIDER_CLASSES.get(provider_type)
    if cls is None:
        raise ValueError(f"Unknown provider type: {provider_type!r}. "
                         f"Valid types: {list(PROVIDER_CLASSES)}")
    return cls()


__all__ = [
    "IAIProvider",
    "ProviderType",
    "AIRequest",
    "AIResponse",
    "ConversationTurn",
    "PROVIDER_CLASSES",
    "get_provider",
    "MistralProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "OllamaProvider",
]
