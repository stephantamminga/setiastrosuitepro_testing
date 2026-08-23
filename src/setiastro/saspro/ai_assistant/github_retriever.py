"""On-demand retrieval of relevant files from the public GitHub repository."""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import List

from .prompting import RetrievedChunk

logger = logging.getLogger(__name__)


class GitHubRepositoryRetriever:
    """Fetch small, filename-relevant source excerpts without a local index."""

    _API_URL = "https://api.github.com/repos/setiastro/setiastrosuitepro"
    _RAW_URL = "https://raw.githubusercontent.com/setiastro/setiastrosuitepro"
    _MAX_FILES = 4
    _MAX_FILE_CHARS = 12_000
    _ALLOWED_SUFFIXES = (".py", ".md", ".rst", ".json")

    def __init__(self):
        self._branch = "main"
        self._paths: List[str] | None = None

    def load(self) -> bool:
        """Load the repository's file tree once for later filename matching."""
        try:
            metadata = self._get_json(self._API_URL)
            self._branch = metadata.get("default_branch", "main")
            tree = self._get_json(
                f"{self._API_URL}/git/trees/{self._branch}?recursive=1"
            )
            self._paths = [
                item["path"]
                for item in tree.get("tree", [])
                if item.get("type") == "blob"
                and item.get("path", "").endswith(self._ALLOWED_SUFFIXES)
            ]
            return bool(self._paths)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            logger.warning("GitHub retriever unavailable: %s", exc)
            self._paths = []
            return False

    def is_ready(self) -> bool:
        return bool(self._paths)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if not self.is_ready():
            return []

        candidates = self._rank_paths(query)[:min(top_k, self._MAX_FILES)]
        chunks: List[RetrievedChunk] = []
        for score, path in candidates:
            try:
                text = self._get_text(f"{self._RAW_URL}/{self._branch}/{path}")
            except (OSError, urllib.error.URLError) as exc:
                logger.warning("Could not fetch GitHub source %s: %s", path, exc)
                continue
            if text.strip():
                chunks.append(RetrievedChunk(
                    text=text[:self._MAX_FILE_CHARS],
                    source_file=path,
                    score=score,
                ))
        return chunks

    def _rank_paths(self, query: str) -> List[tuple[float, str]]:
        query_terms = self._terms(query)
        scored: List[tuple[float, str]] = []
        for path in self._paths or []:
            path_terms = self._terms(path)
            compact_path = re.sub(r"[^a-z0-9]", "", path.lower())
            score = sum(1.0 for term in query_terms if term in path_terms)
            score += sum(2.0 for term in query_terms if len(term) >= 4 and term in compact_path)
            if score:
                scored.append((score, path))
        return sorted(scored, key=lambda item: (-item[0], item[1]))

    @staticmethod
    def _terms(value: str) -> set[str]:
        terms = set(re.findall(r"[a-z0-9]+", value.lower()))
        expanded = set(terms)
        for term in terms:
            if len(term) > 4 and term.endswith("ing"):
                expanded.add(term[:-3])
            elif len(term) > 3 and term.endswith("s"):
                expanded.add(term[:-1])
        return expanded

    @staticmethod
    def _get_json(url: str) -> dict:
        return json.loads(GitHubRepositoryRetriever._get_text(url))

    @staticmethod
    def _get_text(url: str) -> str:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "SetiAstroSuitePro"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read().decode("utf-8")