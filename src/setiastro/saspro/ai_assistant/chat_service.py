# ai_assistant/chat_service.py
"""
Orchestrator for the AI Help Chat.
Mirrors AIService.cs from michelebergo/nina.plugin.aiassistant.

Responsibilities:
- Manages a single active provider instance
- Applies rate limiting (token bucket, max 10 req/min)
- Retries on transient errors (up to 3 attempts, exponential back-off)
- Calls retriever when grounding is enabled
- Builds the final prompt via prompting.py
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .prompting import RetrievedChunk, build_prompt, format_sources
from .providers.base import AIResponse, ConversationTurn
from .settings import AISettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chat response
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    success: bool
    answer: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    model_used: Optional[str] = None
    tokens_used: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Simple token-bucket rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Allow up to *capacity* requests per *window_seconds*."""

    def __init__(self, capacity: int = 10, window_seconds: float = 60.0):
        self._capacity = capacity
        self._window = window_seconds
        self._timestamps: List[float] = []

    def allow(self) -> bool:
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < self._window]
        if len(self._timestamps) >= self._capacity:
            return False
        self._timestamps.append(now)
        return True

    def seconds_until_next(self) -> float:
        if not self._timestamps:
            return 0.0
        now = time.monotonic()
        oldest = min(self._timestamps)
        return max(0.0, self._window - (now - oldest))


# ---------------------------------------------------------------------------
# Chat service
# ---------------------------------------------------------------------------

class ChatService:
    """
    Orchestrates a single conversation session.

    Usage:
        service = ChatService(settings)
        service.load()           # initialise provider + retriever
        resp = service.ask("How do I calibrate flats?")
        print(resp.answer, resp.sources)
    """

    _MAX_RETRIES = 3
    _RETRY_BASE_DELAY = 1.0   # seconds

    def __init__(self, settings: Optional[AISettings] = None):
        self._settings = settings or AISettings()
        self._provider = None
        self._retriever = None
        self._history: List[ConversationTurn] = []
        self._rate_limiter = _RateLimiter()
        self._loaded = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Instantiate and initialise the configured provider, then optionally
        load the retriever.  Returns True on success.
        """
        from .providers import get_provider

        s = self._settings
        try:
            self._provider = get_provider(s.provider)
        except ValueError as exc:
            logger.error("ChatService.load: %s", exc)
            return False

        if not self._provider.initialize(s.to_provider_config()):
            logger.error("ChatService: provider failed to initialise (%s)", s.provider)
            return False

        if s.use_retrieval:
            self._load_retriever()

        self._loaded = True
        return True

    def reload(self) -> bool:
        """Re-initialise from current settings (e.g. after settings change)."""
        self._provider = None
        self._retriever = None
        self._loaded = False
        return self.load()

    def _load_retriever(self) -> None:
        from .retriever import KnowledgeRetriever

        retriever = KnowledgeRetriever(self._settings.index_path)
        if retriever.load():
            self._retriever = retriever
        else:
            logger.info("ChatService: retriever not available (index not built yet?)")

    # ── conversation ──────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        on_retrieval: Optional[Callable[[List[RetrievedChunk]], None]] = None,
    ) -> ChatResponse:
        """
        Process one user question and return a ChatResponse.

        *on_retrieval* is an optional callback invoked with the retrieved
        chunks before the LLM call (used by the UI to show sources early).
        """
        if not self._loaded:
            if not self.load():
                return ChatResponse(
                    success=False,
                    error="AI assistant is not configured. "
                          "Please open Settings (⚙) and enter an API key.",
                )

        if not self._provider or not self._provider.is_configured:
            return ChatResponse(
                success=False,
                error="AI provider not configured. Please check Settings.",
            )

        if not self._rate_limiter.allow():
            wait = self._rate_limiter.seconds_until_next()
            return ChatResponse(
                success=False,
                error=f"Rate limit reached. Please wait {wait:.0f}s before asking again.",
            )

        # Retrieve context
        chunks: List[RetrievedChunk] = []
        if self._settings.use_retrieval and self._retriever and self._retriever.is_ready():
            chunks = self._retriever.retrieve(question, top_k=5)
            if on_retrieval:
                on_retrieval(chunks)

        # Build prompt
        ai_request = build_prompt(
            question=question,
            chunks=chunks if chunks else None,
            history=list(self._history),
            max_context_tokens=self._settings.max_context_tokens,
        )
        ai_request.temperature = self._settings.temperature
        ai_request.max_tokens = self._settings.max_tokens

        # Call provider with retries
        response: AIResponse = self._call_with_retries(ai_request)

        if not response.success:
            return ChatResponse(success=False, error=response.error)

        # Update history
        self._history.append(ConversationTurn(role="user", content=question))
        self._history.append(ConversationTurn(role="assistant", content=response.content or ""))

        # Keep history bounded (last 20 turns)
        if len(self._history) > 20:
            self._history = self._history[-20:]

        return ChatResponse(
            success=True,
            answer=response.content,
            sources=format_sources(chunks) if self._settings.show_sources else [],
            model_used=response.model_used,
            tokens_used=response.tokens_used,
        )

    def clear_history(self) -> None:
        self._history.clear()

    # ── internal ──────────────────────────────────────────────────────────

    def _call_with_retries(self, request) -> AIResponse:
        last_error = "Unknown error"
        for attempt in range(self._MAX_RETRIES):
            try:
                resp = self._provider.send_request(request)
                if resp.success:
                    return resp
                last_error = resp.error or "Provider returned failure"
                # Don't retry on client errors (bad key, etc.)
                if any(code in (last_error or "") for code in ("401", "403", "400")):
                    return resp
            except Exception as exc:
                last_error = str(exc)
                logger.warning("ChatService attempt %d failed: %s", attempt + 1, exc)

            if attempt < self._MAX_RETRIES - 1:
                delay = self._RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)

        return AIResponse(success=False, error=f"Request failed after {self._MAX_RETRIES} attempts: {last_error}")
