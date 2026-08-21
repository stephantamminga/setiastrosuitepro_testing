# ai_assistant/chat_panel.py
"""
AI Help Chat panel — PyQt6 UI.
Mirrors AIChatView.xaml + AIChatVM.cs from michelebergo/nina.plugin.aiassistant,
adapted for PyQt6 and the Seti Astro Suite Pro dark theme.

Architecture:
  HelpChatDialog           — main floating dialog
    ├─ _SettingsPanel      — provider/key configuration sub-dialog
    ├─ _IndexWorker        — QThread for background indexing
    └─ _AskWorker          — QThread for background API calls
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QTimer,
)
from PyQt6.QtGui import QFont, QKeySequence, QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpacerItem, QTextEdit, QVBoxLayout, QWidget,
    QMessageBox, QGroupBox, QFormLayout,
)

from .settings import AISettings
from .providers.base import ProviderType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class _AskWorker(QThread):
    """Calls ChatService.ask() off the main thread."""

    finished = pyqtSignal(object)   # ChatResponse
    retrieved = pyqtSignal(list)    # List[RetrievedChunk] — optional early callback

    def __init__(self, service, question: str, parent=None):
        super().__init__(parent)
        self._service = service
        self._question = question

    def run(self):
        from .chat_service import ChatResponse
        try:
            resp = self._service.ask(
                self._question,
                on_retrieval=lambda chunks: self.retrieved.emit(chunks),
            )
        except Exception as exc:
            from .chat_service import ChatResponse
            resp = ChatResponse(success=False, error=str(exc))
        self.finished.emit(resp)


class _IndexWorker(QThread):
    """Runs KnowledgeIndexer.build() off the main thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str)   # final IndexStatus value string

    def __init__(self, repo_root: str, index_path: str, parent=None):
        super().__init__(parent)
        self._repo_root = repo_root
        self._index_path = index_path

    def run(self):
        from .indexer import KnowledgeIndexer, IndexStatus
        indexer = KnowledgeIndexer(self._repo_root, self._index_path)
        status = indexer.build(
            on_progress=lambda msg: self.progress.emit(msg),
            force=True,
        )
        self.finished.emit(status.value)


class _ModelsWorker(QThread):
    """Fetches available model list from the provider."""

    finished = pyqtSignal(list)

    def __init__(self, provider, parent=None):
        super().__init__(parent)
        self._provider = provider

    def run(self):
        try:
            models = self._provider.get_available_models()
        except Exception:
            models = []
        self.finished.emit(models)


class _TestWorker(QThread):
    """Tests provider connection."""

    finished = pyqtSignal(bool, str)

    def __init__(self, provider, parent=None):
        super().__init__(parent)
        self._provider = provider

    def run(self):
        try:
            ok = self._provider.test_connection()
            self.finished.emit(ok, "" if ok else "Connection test failed")
        except Exception as exc:
            self.finished.emit(False, str(exc))


# ---------------------------------------------------------------------------
# Message bubble widget
# ---------------------------------------------------------------------------

class _MessageBubble(QWidget):
    """A single chat message displayed in the transcript."""

    def __init__(self, text: str, is_user: bool, sources: Optional[List[str]] = None,
                 parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        # Message text
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        label.setFont(QFont("Arial", 10))

        if is_user:
            label.setStyleSheet(
                "background-color: #1e4d7a; color: #e8f4fd; "
                "border-radius: 8px; padding: 8px 12px;"
            )
            label.setAlignment(Qt.AlignmentFlag.AlignRight)
        else:
            label.setStyleSheet(
                "background-color: #2a2a3e; color: #d0d0e8; "
                "border-radius: 8px; padding: 8px 12px;"
            )
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(label, alignment=(
            Qt.AlignmentFlag.AlignRight if is_user else Qt.AlignmentFlag.AlignLeft
        ))

        # Sources (assistant only)
        if not is_user and sources:
            toggle_btn = QPushButton(f"📎 {len(sources)} source(s)")
            toggle_btn.setFlat(True)
            toggle_btn.setStyleSheet("color: #7090b0; font-size: 9pt; text-align: left;")
            toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)

            sources_label = QLabel("\n".join(f"  • {s}" for s in sources))
            sources_label.setStyleSheet("color: #556688; font-size: 9pt;")
            sources_label.setWordWrap(True)
            sources_label.setVisible(False)

            toggle_btn.clicked.connect(lambda: sources_label.setVisible(not sources_label.isVisible()))

            layout.addWidget(toggle_btn, alignment=Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(sources_label, alignment=Qt.AlignmentFlag.AlignLeft)


# ---------------------------------------------------------------------------
# Settings panel (sub-dialog)
# ---------------------------------------------------------------------------

class _SettingsPanel(QDialog):
    """Provider / key / model configuration dialog."""

    settings_saved = pyqtSignal()

    def __init__(self, settings: AISettings, repo_root: str, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._repo_root = repo_root
        self._provider_obj = None
        self._models_worker: Optional[_ModelsWorker] = None
        self._test_worker: Optional[_TestWorker] = None
        self._index_worker: Optional[_IndexWorker] = None
        self.setWindowTitle("AI Assistant Settings")
        self.setMinimumWidth(480)
        self._build_ui()
        self._populate()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form_group = QGroupBox("Provider Configuration")
        form = QFormLayout(form_group)
        form.setSpacing(8)

        self._provider_combo = QComboBox()
        for ptype in ProviderType.ALL:
            self._provider_combo.addItem(ProviderType.DISPLAY_NAMES[ptype], ptype)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        form.addRow("Provider:", self._provider_combo)

        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        form.addRow("Model:", self._model_combo)

        self._fetch_models_btn = QPushButton("↺ Fetch models")
        self._fetch_models_btn.clicked.connect(self._fetch_models)
        form.addRow("", self._fetch_models_btn)

        self._key_edit = QLineEdit()
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_edit.setPlaceholderText("API key (or set SASP_AI_API_KEY env var)")
        form.addRow("API Key:", self._key_edit)

        self._endpoint_edit = QLineEdit()
        self._endpoint_edit.setPlaceholderText("Leave empty for default endpoint")
        form.addRow("Endpoint override:", self._endpoint_edit)

        layout.addWidget(form_group)

        # Generation params
        gen_group = QGroupBox("Generation")
        gen_form = QFormLayout(gen_group)

        self._temp_edit = QLineEdit()
        self._temp_edit.setPlaceholderText("0.3")
        gen_form.addRow("Temperature:", self._temp_edit)

        self._tokens_edit = QLineEdit()
        self._tokens_edit.setPlaceholderText("1024")
        gen_form.addRow("Max tokens:", self._tokens_edit)

        self._ctx_edit = QLineEdit()
        self._ctx_edit.setPlaceholderText("3000")
        gen_form.addRow("Max context tokens:", self._ctx_edit)

        layout.addWidget(gen_group)

        # Behaviour
        beh_group = QGroupBox("Behaviour")
        beh_layout = QVBoxLayout(beh_group)
        self._use_retrieval_cb = QCheckBox("Use retrieval grounding (recommended)")
        self._show_sources_cb = QCheckBox("Show source file references below answers")
        beh_layout.addWidget(self._use_retrieval_cb)
        beh_layout.addWidget(self._show_sources_cb)
        layout.addWidget(beh_group)

        # Index management
        idx_group = QGroupBox("Knowledge Index")
        idx_layout = QVBoxLayout(idx_group)
        self._index_status_label = QLabel("Status: unknown")
        self._index_status_label.setStyleSheet("color: #aaaaaa;")
        idx_layout.addWidget(self._index_status_label)
        self._index_progress = QProgressBar()
        self._index_progress.setRange(0, 0)
        self._index_progress.setVisible(False)
        idx_layout.addWidget(self._index_progress)
        self._reindex_btn = QPushButton("🔄 Re-index repository now")
        self._reindex_btn.clicked.connect(self._start_reindex)
        idx_layout.addWidget(self._reindex_btn)
        layout.addWidget(idx_group)

        # Test / Save
        btn_row = QHBoxLayout()
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._test_connection)
        btn_row.addWidget(self._test_btn)
        btn_row.addStretch()
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _populate(self):
        s = self._settings
        # Provider combo
        for i in range(self._provider_combo.count()):
            if self._provider_combo.itemData(i) == s.provider:
                self._provider_combo.setCurrentIndex(i)
                break
        # Model
        if s.model_id:
            self._model_combo.addItem(s.model_id)
            self._model_combo.setCurrentText(s.model_id)
        self._key_edit.setText(s.api_key)
        self._endpoint_edit.setText(s.endpoint_override)
        self._temp_edit.setText(str(s.temperature))
        self._tokens_edit.setText(str(s.max_tokens))
        self._ctx_edit.setText(str(s.max_context_tokens))
        self._use_retrieval_cb.setChecked(s.use_retrieval)
        self._show_sources_cb.setChecked(s.show_sources)
        self._update_index_status_label()

    def _on_provider_changed(self, _idx: int):
        self._model_combo.clear()

    def _fetch_models(self):
        ptype = self._provider_combo.currentData()
        key = self._key_edit.text().strip()
        from .providers import get_provider
        try:
            p = get_provider(ptype)
            p.initialize({
                "api_key": key,
                "endpoint_override": self._endpoint_edit.text().strip() or None,
            })
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc))
            return
        self._fetch_models_btn.setEnabled(False)
        self._models_worker = _ModelsWorker(p, self)
        self._models_worker.finished.connect(self._on_models_fetched)
        self._models_worker.start()

    def _on_models_fetched(self, models: list):
        current = self._model_combo.currentText()
        self._model_combo.clear()
        for m in models:
            self._model_combo.addItem(m)
        if current:
            idx = self._model_combo.findText(current)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
            else:
                self._model_combo.setCurrentText(current)
        self._fetch_models_btn.setEnabled(True)

    def _test_connection(self):
        ptype = self._provider_combo.currentData()
        key = self._key_edit.text().strip()
        from .providers import get_provider
        try:
            p = get_provider(ptype)
            ok = p.initialize({
                "api_key": key,
                "model_id": self._model_combo.currentText().strip() or None,
                "endpoint_override": self._endpoint_edit.text().strip() or None,
            })
            if not ok:
                QMessageBox.warning(self, "Test Failed", "Provider initialisation failed.")
                return
        except Exception as exc:
            QMessageBox.warning(self, "Test Failed", str(exc))
            return
        self._test_btn.setEnabled(False)
        self._test_btn.setText("Testing…")
        self._test_worker = _TestWorker(p, self)
        self._test_worker.finished.connect(self._on_test_finished)
        self._test_worker.start()

    def _on_test_finished(self, ok: bool, error: str):
        self._test_btn.setEnabled(True)
        self._test_btn.setText("Test Connection")
        if ok:
            QMessageBox.information(self, "Success", "✅ Connection successful!")
        else:
            QMessageBox.warning(self, "Failed", f"❌ {error or 'Connection test failed'}")

    def _start_reindex(self):
        self._reindex_btn.setEnabled(False)
        self._index_progress.setVisible(True)
        self._index_status_label.setText("🔄 Indexing…")
        index_path = self._settings.index_path
        self._index_worker = _IndexWorker(self._repo_root, index_path, self)
        self._index_worker.progress.connect(self._index_status_label.setText)
        self._index_worker.finished.connect(self._on_index_finished)
        self._index_worker.start()

    def _on_index_finished(self, status_value: str):
        self._reindex_btn.setEnabled(True)
        self._index_progress.setVisible(False)
        self._update_index_status_label(status_value)

    def _update_index_status_label(self, status_value: Optional[str] = None):
        from .indexer import KnowledgeIndexer, IndexStatus
        if status_value:
            status = status_value
        else:
            indexer = KnowledgeIndexer(self._repo_root, self._settings.index_path)
            if not indexer.is_stale():
                status = IndexStatus.READY.value
            else:
                status = IndexStatus.PENDING.value
        icon = {"ready": "✅", "pending": "⏳", "indexing": "🔄",
                "error": "❌", "unavailable": "⚠️"}.get(status, "❓")
        self._index_status_label.setText(f"{icon} {status.capitalize()}")

    def _save(self):
        s = self._settings
        s.provider = self._provider_combo.currentData()
        s.model_id = self._model_combo.currentText().strip()
        s.api_key = self._key_edit.text().strip()
        s.endpoint_override = self._endpoint_edit.text().strip()
        try:
            s.temperature = float(self._temp_edit.text() or 0.3)
        except ValueError:
            pass
        try:
            s.max_tokens = int(self._tokens_edit.text() or 1024)
        except ValueError:
            pass
        try:
            s.max_context_tokens = int(self._ctx_edit.text() or 3000)
        except ValueError:
            pass
        s.use_retrieval = self._use_retrieval_cb.isChecked()
        s.show_sources = self._show_sources_cb.isChecked()
        self.settings_saved.emit()
        self.accept()


# ---------------------------------------------------------------------------
# Main chat dialog
# ---------------------------------------------------------------------------

class HelpChatDialog(QDialog):
    """
    Main AI Help Chat window.

    Create once and reuse (non-modal, show/raise):
        self._help_chat = HelpChatDialog(repo_root, parent=self)
        self._help_chat.show()
    """

    def __init__(self, repo_root: str, parent=None):
        super().__init__(parent)
        self._repo_root = repo_root
        self._settings = AISettings()
        self._service = None
        self._ask_worker: Optional[_AskWorker] = None
        self._settings_panel: Optional[_SettingsPanel] = None
        self._bubble_container_layout: Optional[QVBoxLayout] = None
        self._scroll_area: Optional[QScrollArea] = None

        self.setWindowTitle("Seti Astro Help Chat (AI)")
        self.setMinimumSize(520, 620)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._build_ui()
        self._init_service()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Header bar ────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("🤖 Seti Astro Help Chat")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #d0d0e8;")
        hdr.addWidget(title)
        hdr.addStretch()

        self._index_label = QLabel("⏳ Index: pending")
        self._index_label.setStyleSheet("color: #888; font-size: 9pt;")
        hdr.addWidget(self._index_label)

        gear_btn = QPushButton("⚙")
        gear_btn.setFixedSize(28, 28)
        gear_btn.setToolTip("Settings")
        gear_btn.clicked.connect(self._open_settings)
        hdr.addWidget(gear_btn)

        root.addLayout(hdr)

        # ── Transcript area ───────────────────────────────────────────────
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setStyleSheet(
            "QScrollArea { border: 1px solid #333; background: #1a1a2e; }"
        )

        bubble_container = QWidget()
        bubble_container.setStyleSheet("background: #1a1a2e;")
        self._bubble_container_layout = QVBoxLayout(bubble_container)
        self._bubble_container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._bubble_container_layout.setSpacing(6)
        self._bubble_container_layout.setContentsMargins(6, 6, 6, 6)
        # Add spacer so messages stay at top when few
        self._bubble_container_layout.addStretch(1)

        self._scroll_area.setWidget(bubble_container)
        root.addWidget(self._scroll_area, 1)

        # ── Status bar ────────────────────────────────────────────────────
        status_row = QHBoxLayout()
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #7090b0; font-size: 9pt;")
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        self._model_label = QLabel("")
        self._model_label.setStyleSheet("color: #556677; font-size: 8pt;")
        status_row.addWidget(self._model_label)
        root.addLayout(status_row)

        # ── Input area ────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333;")
        root.addWidget(sep)

        input_row = QHBoxLayout()
        self._input_edit = QTextEdit()
        self._input_edit.setPlaceholderText("Ask a question about Seti Astro Suite Pro… (Enter to send, Shift+Enter for newline)")
        self._input_edit.setFixedHeight(72)
        self._input_edit.installEventFilter(self)
        self._input_edit.setStyleSheet(
            "QTextEdit { background: #1e1e2e; color: #d0d0e8; "
            "border: 1px solid #444; border-radius: 4px; padding: 4px; }"
        )
        input_row.addWidget(self._input_edit, 1)

        send_btn = QPushButton("Send")
        send_btn.setFixedWidth(70)
        send_btn.setFixedHeight(72)
        send_btn.clicked.connect(self._send)
        send_btn.setStyleSheet(
            "QPushButton { background: #1e4d7a; color: white; border-radius: 4px; }"
            "QPushButton:hover { background: #2a6aaa; }"
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        input_row.addWidget(send_btn)
        self._send_btn = send_btn

        root.addLayout(input_row)

        # Clear history button
        clear_row = QHBoxLayout()
        clear_btn = QPushButton("Clear conversation")
        clear_btn.setFlat(True)
        clear_btn.setStyleSheet("color: #556677; font-size: 8pt;")
        clear_btn.clicked.connect(self._clear_conversation)
        clear_row.addWidget(clear_btn)
        clear_row.addStretch()
        root.addLayout(clear_row)

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent
        if obj is self._input_edit and event.type() == QEvent.Type.KeyPress:
            key_event: QKeyEvent = event
            if (key_event.key() == Qt.Key.Key_Return
                    and not (key_event.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
                self._send()
                return True
        return super().eventFilter(obj, event)

    # ── Service initialisation ────────────────────────────────────────────

    def _init_service(self):
        from .chat_service import ChatService
        self._service = ChatService(self._settings)
        self._update_model_label()
        self._update_index_label()

        if not self._settings.is_ready():
            QTimer.singleShot(300, self._open_settings_first_run)

    def _open_settings_first_run(self):
        QMessageBox.information(
            self, "Welcome to AI Help Chat",
            "To get started, please configure your AI provider and API key in the Settings dialog.\n\n"
            "Supported providers: OpenAI, Mistral AI, Anthropic, Google Gemini, Ollama (local)."
        )
        self._open_settings()

    # ── Settings ──────────────────────────────────────────────────────────

    def _open_settings(self):
        panel = _SettingsPanel(self._settings, self._repo_root, parent=self)
        panel.settings_saved.connect(self._on_settings_saved)
        panel.exec()

    def _on_settings_saved(self):
        self._service = None
        from .chat_service import ChatService
        self._service = ChatService(self._settings)
        self._update_model_label()
        self._update_index_label()

    def _update_model_label(self):
        provider_name = ProviderType.DISPLAY_NAMES.get(self._settings.provider, self._settings.provider)
        model = self._settings.model_id or "default"
        self._model_label.setText(f"{provider_name} · {model}")

    def _update_index_label(self):
        try:
            from .indexer import KnowledgeIndexer, IndexStatus
            indexer = KnowledgeIndexer(self._repo_root, self._settings.index_path)
            if not indexer.is_stale():
                self._index_label.setText("✅ Index: ready")
                self._index_label.setStyleSheet("color: #4a9; font-size: 9pt;")
            else:
                self._index_label.setText("⏳ Index: pending")
                self._index_label.setStyleSheet("color: #888; font-size: 9pt;")
        except Exception:
            self._index_label.setText("⚠ Index: unavailable")
            self._index_label.setStyleSheet("color: #a84; font-size: 9pt;")

    # ── Sending messages ──────────────────────────────────────────────────

    def _send(self):
        question = self._input_edit.toPlainText().strip()
        if not question:
            return
        if self._ask_worker and self._ask_worker.isRunning():
            return

        self._input_edit.clear()
        self._add_bubble(question, is_user=True)
        self._set_thinking(True)

        if not self._service:
            self._on_settings_saved()

        self._ask_worker = _AskWorker(self._service, question, self)
        self._ask_worker.finished.connect(self._on_answer)
        self._ask_worker.start()

    def _on_answer(self, response):
        self._set_thinking(False)
        if response.success:
            self._add_bubble(
                response.answer or "(no response)",
                is_user=False,
                sources=response.sources,
            )
            if response.model_used:
                self._model_label.setText(
                    f"{ProviderType.DISPLAY_NAMES.get(self._settings.provider, '')} · {response.model_used}"
                )
        else:
            self._add_bubble(
                f"⚠ Error: {response.error}",
                is_user=False,
            )

    # ── Conversation helpers ──────────────────────────────────────────────

    def _add_bubble(self, text: str, is_user: bool, sources: Optional[List[str]] = None):
        layout = self._bubble_container_layout
        if layout is None:
            return
        # Remove the trailing stretch before adding content
        stretch_item = layout.itemAt(layout.count() - 1)
        if stretch_item and stretch_item.spacerItem():
            layout.removeItem(stretch_item)

        bubble = _MessageBubble(text, is_user, sources, parent=self._scroll_area.widget())
        layout.addWidget(bubble)
        layout.addStretch(1)

        # Scroll to bottom after adding
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        if self._scroll_area:
            sb = self._scroll_area.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _set_thinking(self, thinking: bool):
        self._send_btn.setEnabled(not thinking)
        self._input_edit.setEnabled(not thinking)
        self._status_label.setText("⟳ Thinking…" if thinking else "")

    def _clear_conversation(self):
        if self._service:
            self._service.clear_history()
        layout = self._bubble_container_layout
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            layout.addStretch(1)

    # ── Convenience class method ──────────────────────────────────────────

    @classmethod
    def open_or_raise(cls, repo_root: str, parent=None):
        """
        Open a new HelpChatDialog or raise the existing one.
        Store the instance on *parent* as ``_help_chat_dialog``.
        """
        dlg: Optional[HelpChatDialog] = getattr(parent, "_help_chat_dialog", None)
        if dlg is None or not dlg.isVisible():
            dlg = cls(repo_root=repo_root, parent=parent)
            if parent is not None:
                parent._help_chat_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return dlg
