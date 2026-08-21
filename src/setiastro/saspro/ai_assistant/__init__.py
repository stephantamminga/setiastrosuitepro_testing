# ai_assistant/__init__.py
"""
AI Help Chat for Seti Astro Suite Pro.

Quick usage from the main window:

    from setiastro.saspro.ai_assistant import open_help_chat
    open_help_chat(repo_root="/path/to/repo", parent=main_window)
"""
from __future__ import annotations

from typing import Optional


def open_help_chat(repo_root: str, parent=None):
    """
    Open (or raise) the AI Help Chat dialog.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root.  Used by the indexer to
        locate source files and docs.
    parent:
        Optional parent QWidget (main window).
    """
    from .chat_panel import HelpChatDialog
    return HelpChatDialog.open_or_raise(repo_root=repo_root, parent=parent)


__all__ = ["open_help_chat"]
