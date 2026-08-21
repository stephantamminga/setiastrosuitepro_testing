# ai_assistant/prompting.py
"""
Prompt builder and token-budget manager.

Mirrors the prompt-builder component in michelebergo/nina.plugin.aiassistant
(Options.xaml.cs + AIChatVM.cs) but adapted for Seti Astro Suite Pro.

Key principle: keep the LLM strictly grounded in retrieved context so it
cannot invent menu names, settings, or behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .providers.base import AIRequest, ConversationTurn


# ---------------------------------------------------------------------------
# System prompt — strict grounding (mirrors NINA plugin pattern)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Seti Astro Suite Pro Help Assistant.

Rules you MUST follow at all times:
1. Answer questions using ONLY the repository context provided below.
2. If the context is insufficient to answer, say exactly:
   "I don't have enough information in the repository context to answer that."
3. NEVER invent menu names, settings, keyboard shortcuts, or application behaviour.
4. Prefer numbered steps for procedural answers.
5. Include source file paths when referencing specific functionality,
   e.g. "See `src/setiastro/saspro/calibration.py`."
6. Keep answers concise and actionable.
"""


# ---------------------------------------------------------------------------
# Retrieved chunk (populated by retriever.py)
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single chunk of repository text returned by the retriever."""
    text: str
    source_file: str
    score: float = 0.0


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (safe over-estimate)."""
    return max(1, len(text) // 4)


def build_prompt(
    question: str,
    chunks: Optional[List[RetrievedChunk]],
    history: Optional[List[ConversationTurn]] = None,
    max_context_tokens: int = 3000,
) -> AIRequest:
    """
    Assemble an AIRequest from the user question, retrieved chunks, and
    conversation history.

    The retrieved chunks are formatted as:

        [Source: <source_file>]
        <chunk text>

    They are included in order of descending relevance score, trimmed when
    the total would exceed *max_context_tokens*.

    Parameters
    ----------
    question:
        The raw user question.
    chunks:
        Retrieved context chunks (may be None or empty if retrieval is disabled).
    history:
        Previous conversation turns for multi-turn context.
    max_context_tokens:
        Hard upper bound on context tokens injected into the prompt.
    """
    context_parts: List[str] = []
    token_budget = max_context_tokens

    if chunks:
        for chunk in chunks:
            header = f"[Source: {chunk.source_file}]"
            body = chunk.text.strip()
            block = f"{header}\n{body}"
            cost = _approx_tokens(block)
            if cost > token_budget:
                # Include a truncated version if at least 50 tokens remain
                if token_budget > 50:
                    chars = token_budget * 4
                    block = f"{header}\n{body[:chars]}…"
                    context_parts.append(block)
                break
            context_parts.append(block)
            token_budget -= cost

    if context_parts:
        context_section = (
            "---\nREPOSITORY CONTEXT:\n\n"
            + "\n\n".join(context_parts)
            + "\n---"
        )
        full_prompt = f"{context_section}\n\nQUESTION: {question}"
    else:
        full_prompt = question

    return AIRequest(
        prompt=full_prompt,
        system_prompt=SYSTEM_PROMPT,
        history=history or [],
    )


def format_sources(chunks: Optional[List[RetrievedChunk]]) -> List[str]:
    """Return deduplicated source file paths from a chunk list."""
    if not chunks:
        return []
    seen: set = set()
    result: List[str] = []
    for chunk in chunks:
        if chunk.source_file not in seen:
            seen.add(chunk.source_file)
            result.append(chunk.source_file)
    return result
