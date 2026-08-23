# ai_assistant/indexer.py
"""
Knowledge-base builder for the AI assistant.

Indexes repository content into overlapping text chunks, embeds them with
sentence-transformers (all-MiniLM-L6-v2), and persists to a FAISS flat
index + JSON chunk store under ~/.setiastro/ai_index/.

Falls back gracefully when sentence-transformers / faiss are not installed
(the chat still works without retrieval in that case).
"""
from __future__ import annotations

import ast
import enum
import hashlib
import json
import logging
import os
import pathlib
import re
import threading
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status enum (consumed by UI)
# ---------------------------------------------------------------------------

class IndexStatus(enum.Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"
    UNAVAILABLE = "unavailable"   # sentence-transformers / faiss not installed


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class IndexedChunk:
    text: str
    source_file: str
    chunk_index: int


# ---------------------------------------------------------------------------
# Helper: extract meaningful text from Python source
# ---------------------------------------------------------------------------

def _extract_python_symbols(source: str, max_body_lines: int = 30) -> str:
    """
    Extract module docstring + all function/class signatures with their
    docstrings.  Body lines are truncated to keep chunk sizes small.
    """
    lines: List[str] = []
    source_lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source[:2000]

    # Module docstring
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        lines.append(mod_doc)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Signature line
            try:
                sig_line = source_lines[node.lineno - 1].strip()
            except IndexError:
                sig_line = f"def/class {node.name}"

            doc = ast.get_docstring(node) or ""
            entry = sig_line
            if doc:
                entry += "\n    " + doc.replace("\n", "\n    ")
            end_line = min(
                getattr(node, "end_lineno", node.lineno),
                node.lineno + max_body_lines - 1,
            )
            excerpt = "\n".join(source_lines[node.lineno - 1:end_line]).strip()
            if excerpt and excerpt != sig_line:
                entry += "\n" + excerpt
            lines.append(entry)

    return "\n\n".join(lines)


def _extract_tooltips(source: str) -> str:
    """Pull setToolTip / setWhatsThis / setStatusTip strings from source."""
    pattern = re.compile(
        r'\.set(?:ToolTip|WhatsThis|StatusTip)\s*\(\s*(?:self\.tr\s*\()?'
        r'["\']([^"\']{8,})["\']',
        re.IGNORECASE,
    )
    found = list(dict.fromkeys(pattern.findall(source)))
    return "\n".join(found[:100]) if found else ""


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def _chunk_text(text: str, source_file: str, chunk_size: int = 300,
                overlap: int = 50) -> List[IndexedChunk]:
    """
    Split *text* into ~chunk_size token chunks with *overlap* token overlap.
    Uses word-based splitting (1 word ≈ 1.3 tokens is close enough).
    """
    words = text.split()
    if not words:
        return []
    chunks: List[IndexedChunk] = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(IndexedChunk(text=chunk_text, source_file=source_file, chunk_index=idx))
        idx += 1
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# File collector
# ---------------------------------------------------------------------------

def _collect_files(repo_root: str) -> List[Tuple[str, str]]:
    """
    Return list of (path, file_type) pairs to index.
    file_type: "python" | "markdown" | "json"
    """
    root = pathlib.Path(repo_root)
    result: List[Tuple[str, str]] = []

    # README / docs
    for pattern in ("README.md", "README.rst", "docs/**/*.md", "docs/**/*.rst"):
        for p in root.glob(pattern):
            result.append((str(p), "markdown"))

    # updates.json (changelog)
    updates = root / "updates.json"
    if updates.exists():
        result.append((str(updates), "json"))

    # Python sources — signatures + docstrings only
    saspro = root / "src" / "setiastro" / "saspro"
    if saspro.exists():
        for p in sorted(saspro.rglob("*.py")):
            if "__pycache__" not in str(p):
                result.append((str(p), "python"))

    return result


# ---------------------------------------------------------------------------
# Hasher (staleness check)
# ---------------------------------------------------------------------------

def _hash_files(files: List[Tuple[str, str]]) -> str:
    h = hashlib.sha256()
    h.update(b"ai-index-format-v2")
    for path, _ in sorted(files):
        try:
            stat = os.stat(path)
            h.update(f"{path}:{stat.st_size}:{stat.st_mtime}".encode())
        except OSError:
            pass
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class KnowledgeIndexer:
    """
    Manages the FAISS-backed knowledge index for the AI assistant.

    Usage:
        indexer = KnowledgeIndexer(repo_root="/path/to/repo",
                                   index_path="~/.setiastro/ai_index")
        indexer.build(on_progress=lambda msg: print(msg))
        # later:
        vecs, chunks = indexer.load()
    """

    _CHUNKS_FILE = "chunks.json"
    _INDEX_FILE = "index.faiss"
    _HASH_FILE = "source_hash.txt"
    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, repo_root: str, index_path: str):
        self.repo_root = repo_root
        self.index_path = os.path.expanduser(index_path)
        self.status: IndexStatus = IndexStatus.PENDING
        self._lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────

    def is_stale(self) -> bool:
        """Return True if the index needs rebuilding."""
        hash_path = os.path.join(self.index_path, self._HASH_FILE)
        if not os.path.exists(hash_path):
            return True
        if not os.path.exists(os.path.join(self.index_path, self._INDEX_FILE)):
            return True
        try:
            with open(hash_path) as f:
                stored = f.read().strip()
            files = _collect_files(self.repo_root)
            return stored != _hash_files(files)
        except Exception:
            return True

    def build(
        self,
        on_progress: Optional[Callable[[str], None]] = None,
        force: bool = False,
    ) -> IndexStatus:
        """
        Build (or rebuild) the index.  Thread-safe; may be called from a
        background QThread.

        Returns the final IndexStatus.
        """
        with self._lock:
            if not force and not self.is_stale():
                self.status = IndexStatus.READY
                return self.status

            self.status = IndexStatus.INDEXING
            _emit(on_progress, "Checking dependencies…")

            try:
                import numpy as np  # noqa: F401
                from sentence_transformers import SentenceTransformer
                import faiss  # noqa: F401
            except ImportError as exc:
                logger.warning("AI indexer: optional deps missing (%s). "
                               "Retrieval disabled.", exc)
                self.status = IndexStatus.UNAVAILABLE
                return self.status

            try:
                _emit(on_progress, "Collecting repository files…")
                files = _collect_files(self.repo_root)
                _emit(on_progress, f"Found {len(files)} files to index.")

                chunks: List[IndexedChunk] = []
                for path, ftype in files:
                    try:
                        raw = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        continue
                    rel = os.path.relpath(path, self.repo_root)
                    if ftype == "python":
                        symbols = _extract_python_symbols(raw)
                        tips = _extract_tooltips(raw)
                        text = "\n\n".join(filter(None, [symbols, tips]))
                    elif ftype == "json":
                        text = raw[:4000]
                    else:
                        text = raw
                    chunks.extend(_chunk_text(text, rel))

                _emit(on_progress, f"Created {len(chunks)} chunks. Embedding…")

                model = SentenceTransformer(self._MODEL_NAME)
                texts = [c.text for c in chunks]
                embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)

                import numpy as np
                import faiss
                embeddings_np = np.array(embeddings, dtype="float32")
                faiss.normalize_L2(embeddings_np)
                dim = embeddings_np.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings_np)

                os.makedirs(self.index_path, exist_ok=True)
                faiss.write_index(index, os.path.join(self.index_path, self._INDEX_FILE))
                with open(os.path.join(self.index_path, self._CHUNKS_FILE), "w") as f:
                    json.dump([asdict(c) for c in chunks], f)
                source_hash = _hash_files(files)
                with open(os.path.join(self.index_path, self._HASH_FILE), "w") as f:
                    f.write(source_hash)

                _emit(on_progress, f"Index ready ({len(chunks)} chunks, dim={dim}).")
                self.status = IndexStatus.READY
                return self.status

            except Exception as exc:
                logger.error("Indexer failed: %s", exc, exc_info=True)
                self.status = IndexStatus.ERROR
                _emit(on_progress, f"Indexing error: {exc}")
                return self.status

    def load(self):
        """
        Load the FAISS index and chunk list from disk.

        Returns (faiss_index, chunks: List[IndexedChunk]) or (None, []).
        """
        index_file = os.path.join(self.index_path, self._INDEX_FILE)
        chunks_file = os.path.join(self.index_path, self._CHUNKS_FILE)
        if not (os.path.exists(index_file) and os.path.exists(chunks_file)):
            return None, []
        try:
            import faiss
            index = faiss.read_index(index_file)
            with open(chunks_file) as f:
                raw = json.load(f)
            chunks = [IndexedChunk(**c) for c in raw]
            return index, chunks
        except Exception as exc:
            logger.error("Could not load FAISS index: %s", exc)
            return None, []


def _emit(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb:
        try:
            cb(msg)
        except Exception:
            pass
