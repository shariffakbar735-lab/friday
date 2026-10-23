"""FRIDAY local-first desktop assistant with optional hardware-aware inference controls."""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QSettings, QThread, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMainWindow, QMessageBox, QPushButton,
                               QTextBrowser, QVBoxLayout, QWidget)

from app.ai.cloud_client import CloudConfigurationError, OpenAICompatibleClient
from app.ai.hardware import HardwareSnapshot, detect_hardware, estimate_model_fits, recommend_model
from app.ai.ollama_client import Model, OllamaClient, OllamaError
from app.ai.profiles import PROFILES
from app.memory.storage import ConversationStore


def bounded_context(messages: list[dict[str, str]], max_chars: int) -> list[dict[str, str]]:
    """Keep the newest contiguous message suffix within a conservative char budget."""
    selected: list[dict[str, str]] = []
    used = 0
    for message in reversed(messages):
        size = len(message.get("content", ""))
        if used + size > max_chars:
            break
        selected.append(message)
        used += size
    return list(reversed(selected))


def prompt_char_budget(num_ctx: int) -> int:
    # Leave a quarter of the model context for the system prompt and tokenization variation.
    return max(256, num_ctx * 3 // 4)


def is_gpu_initialization_error(message: str) -> bool:
    lowered = message.lower()
    return any(word in lowered for word in ("cuda", "gpu", "vram", "driver", "offload", "failed to initialize"))


class StartupWorker(QThread):
    initialized = Signal(object, object, str)

    def __init__(self, client: OllamaClient) -> None:
        super().__init__()
        self.client = client

    def run(self) -> None:
        snapshot = detect_hardware(self.client)
        models: list[Model] = []
        error = ""
        try:
            models = self.client.list_models()
        except OllamaError as exc:
            error = str(exc)
        self.initialized.emit(snapshot, models, error)


class ChatWorker(QThread):
    chunk = Signal(str)
    failed = Signal(str, str)
    finished_text = Signal(str)

    def __init__(self, client: OllamaClient, model: str, messages: list[dict[str, str]],
                 options: dict[str, int | float], backend: str = "local") -> None:
        super().__init__()
        self.client, self.model, self.messages = client, model, messages
        self.options, self.backend = options, backend
        self.stop_event = threading.Event()

    def run(self) -> None:
        pieces: list[str] = []
        try:
            if self.backend == "cloud":
                cloud = OpenAICompatibleClient()
                if not cloud.configured:
                    raise CloudConfigurationError("Cloud provider settings are incomplete. Choose Local Ollama or configure the FRIDAY_CLOUD_* environment variables.")
                stream = cloud.stream_chat(self.messages, self.stop_event,
                                           max_tokens=int(self.options["num_predict"]))
            else:
                stream = self.client.stream_chat(self.model, self.messages, self.stop_event, self.options)
            for piece in stream:
                pieces.append(piece)
                self.chunk.emit(piece)
            answer = "".join(pieces)
            if answer.strip():
                self.finished_text.emit(answer)
            else:
                self.failed.emit("The model completed without visible answer text. Try a larger output budget or another installed model.", answer)
        except Exception as exc:
            partial = "".join(pieces)
            if self.stop_event.is_set():
                self.finished_text.emit(partial)
            else:
                self.failed.emit(str(exc), partial)

    def stop(self) -> None:
        self.stop_event.set()


class BenchmarkWorker(QThread):
    completed = Signal(float, float, str)
    failed = Signal(str)

    def __init__(self, client: OllamaClient, model: str, options: dict[str, int | float]) -> None:
        super().__init__()
        self.client, self.model, self.options = client, model, options

    def run(self) -> None:
        started = time.perf_counter()
        first_chunk: float | None = None
        pieces: list[str] = []
        try:
            for text in self.client.stream_chat(
                self.model, [{"role": "user", "content": "Reply with one short sentence confirming you are ready."}],
                options=self.options,
            ):
                if first_chunk is None:
                    first_chunk = time.perf_counter() - started
                pieces.append(text)
            elapsed = time.perf_counter() - started
            answer = "".join(pieces)
            if not answer.strip():
                raise RuntimeError("The model completed without visible answer text.")
            self.completed.emit(elapsed, first_chunk or elapsed, answer)
        except Exception as exc:
            self.failed.emit(str(exc))


class FridayWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FRIDAY — Local AI Assistant")
        self.resize(1120, 760)
        self.settings = QSettings("FRIDAY", "FRIDAY")
        data_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "FRIDAY"
        self.store = ConversationStore(data_dir / "friday.sqlite3")
        self.client = OllamaClient()
        self.conversation_id = self.store.create_conversation()
        self.worker: ChatWorker | None = None
        self.startup_worker: StartupWorker | None = None
        self.benchmark_worker: BenchmarkWorker | None = None
        self.snapshot: HardwareSnapshot | None = None
        self.installed_models: list[Model] = []
        self.streamed_text = ""
        self.active_response_conversation_id: str | None = None
        self.active_messages: list[dict[str, str]] = []
        self.last_failed_conversation_id: str | None = None
        self._cloud_confirmed = False
        self.new_button: QPushButton | None = None
        self.cpu_retry_button: QPushButton | None = None
        self._backend_guard = False
        self._build_ui()
        self._start_discovery()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        sidebar = QVBoxLayout()
        brand = QLabel("FRIDAY")
        brand.setStyleSheet("font-size: 24px; font-weight: 700; color: #79d7ff")
        sidebar.addWidget(brand)
        self.status = QLabel("Checking hardware and Ollama…")
        self.status.setWordWrap(True)
        sidebar.addWidget(self.status)

        sidebar.addWidget(QLabel("AI provider"))
        self.backend = QComboBox()
        self.backend.addItem("Local — Ollama", "local")
        self.backend.addItem("Optional cloud — OpenAI-compatible", "cloud")
        selected_backend = self.settings.value("provider", "local")
        self.backend.setCurrentIndex(max(0, self.backend.findData(selected_backend)))
        self.backend.currentIndexChanged.connect(self._backend_changed)
        sidebar.addWidget(self.backend)

        sidebar.addWidget(QLabel("Performance profile"))
        self.profile = QComboBox()
        for key, item in PROFILES.items():
            self.profile.addItem(item.display_name, key)
        saved_profile = self.settings.value("profile", "balanced")
        self.profile.setCurrentIndex(max(0, self.profile.findData(saved_profile)))
        self.profile.currentIndexChanged.connect(self._profile_changed)
        sidebar.addWidget(self.profile)

        sidebar.addWidget(QLabel("Installed local model"))
        self.models = QComboBox()
        self.models.setEnabled(False)
        self.models.currentTextChanged.connect(self._model_changed)
        sidebar.addWidget(self.models)
        self.hardware_button = QPushButton("Hardware & setup")
        self.hardware_button.clicked.connect(self._show_hardware)
        sidebar.addWidget(self.hardware_button)
        self.benchmark_button = QPushButton("Run local benchmark")
        self.benchmark_button.clicked.connect(self._run_benchmark)
        sidebar.addWidget(self.benchmark_button)
        self.benchmark_result = QLabel("")
        self.benchmark_result.setWordWrap(True)
        sidebar.addWidget(self.benchmark_result)

        self.new_button = QPushButton("+ New conversation")
        self.new_button.clicked.connect(self._new_conversation)
        sidebar.addWidget(self.new_button)
        self.history = QListWidget()
        self.history.itemClicked.connect(self._open_conversation)
        sidebar.addWidget(self.history, 1)
        layout.addLayout(sidebar, 1)

        chat = QVBoxLayout()
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(True)
        chat.addWidget(self.transcript, 1)
        controls = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Message FRIDAY…")
        self.input.returnPressed.connect(self._send)
        controls.addWidget(self.input, 1)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._send)
        controls.addWidget(self.send_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        controls.addWidget(self.stop_button)
        chat.addLayout(controls)
        self.cpu_retry_button = QPushButton("Retry last request using CPU only")
        self.cpu_retry_button.setVisible(False)
        self.cpu_retry_button.clicked.connect(self._retry_cpu)
        chat.addWidget(self.cpu_retry_button)
        layout.addLayout(chat, 4)
        self.setCentralWidget(root)
        self.setStyleSheet("QMainWindow,QWidget{background:#111722;color:#e8eef7}"
                           "QTextBrowser,QLineEdit,QListWidget,QComboBox{background:#192332;"
                           "border:1px solid #344258;border-radius:6px;padding:8px}"
                           "QPushButton{background:#216c91;border:0;border-radius:6px;padding:9px 14px}"
                           "QPushButton:disabled{background:#374151;color:#aab2bf}")
        self._refresh_history()
        self._render_messages()

    def _start_discovery(self) -> None:
        if self.startup_worker is not None and self.startup_worker.isRunning():
            return
        self.send_button.setEnabled(False)
        self.models.setEnabled(False)
        self.startup_worker = StartupWorker(self.client)
        self.startup_worker.initialized.connect(self._initialization_done)
        self.startup_worker.finished.connect(self._startup_finished)
        self.startup_worker.start()

    def _startup_finished(self) -> None:
        self.startup_worker = None

    def _initialization_done(self, snapshot: HardwareSnapshot, models: list[Model], error: str) -> None:
        self.snapshot = snapshot
        self.installed_models = models
        if self.backend.currentData() == "cloud" and not OpenAICompatibleClient().configured:
            self._reset_backend_to_local()
            self.settings.setValue("provider", "local")
        self.models.clear()
        if self.backend.currentData() == "cloud" and OpenAICompatibleClient().configured:
            self.models.addItem(OpenAICompatibleClient().model)
        else:
            self.models.addItems([model.name for model in models])
        fits = estimate_model_fits(snapshot, models, self._profile_key())
        suggested = recommend_model(fits, self._profile_key())
        saved_model = str(self.settings.value("model", "") or "")
        installed_names = {model.name for model in models}
        preferred = saved_model if saved_model in installed_names else suggested
        if not preferred and models:
            preferred = "qwen3:4b" if "qwen3:4b" in installed_names else models[0].name
        selected = self.models.findText(str(preferred or ""))
        if selected >= 0:
            self.models.setCurrentIndex(selected)
        self.models.setEnabled(bool(models) and self.backend.currentData() == "local")
        self._model_changed(self.models.currentText())
        if not snapshot.ollama_running:
            self.status.setText("Ollama is not responding; chat history and setup remain available.")
            self.status.setToolTip(error or "Start Ollama to enable local chat.")
        else:
            model_line = f"{len(models)} local model(s)" if models else "Ollama running; no local models installed"
            self.status.setText(f"{snapshot.ram_total_bytes / 1024**3:.1f} GB RAM · " if snapshot.ram_total_bytes else "RAM unknown · "
                                f"{snapshot.gpu_name or 'CPU-only'} · {model_line}")
        if self.backend.currentData() == "local":
            self.send_button.setEnabled(bool(models) and snapshot.ollama_running)
        else:
            self.send_button.setEnabled(OpenAICompatibleClient().configured)

    def _profile_key(self) -> str:
        return str(self.profile.currentData() or "balanced")

    def _profile_changed(self, _index: int) -> None:
        key = self._profile_key()
        self.settings.setValue("profile", key)
        self._model_changed(self.models.currentText())

    def _model_changed(self, model_name: str) -> None:
        if not self.snapshot:
            return
        if self.backend.currentData() == "local":
            self.settings.setValue("model", model_name)
        fits = estimate_model_fits(self.snapshot, self.installed_models, self._profile_key())
        suggested = recommend_model(fits, self._profile_key())
        candidate = next((item for item in fits if item.name == model_name), None)
        if candidate:
            state = "estimated fit" if candidate.fits_estimate else "may exceed available memory" if candidate.fits_estimate is False else "fit unknown"
            self.models.setToolTip(f"{candidate.explanation}\nRecommendation: {suggested or 'none of the installed models fit the advisory estimate'}")
            self.status.setToolTip(f"Selected model {model_name}: {state}. {candidate.explanation}")

    def _backend_changed(self, index: int) -> None:
        if self._backend_guard:
            return
        provider = self.backend.itemData(index)
        if provider == "cloud":
            cloud = OpenAICompatibleClient()
            if not cloud.configured:
                QMessageBox.information(self, "Optional cloud setup",
                    "Cloud mode is not configured. Set FRIDAY_CLOUD_BASE_URL, FRIDAY_CLOUD_MODEL, and "
                    "FRIDAY_CLOUD_API_KEY in your Windows environment, then restart FRIDAY. "
                    "Cloud mode is optional; local Ollama remains the default.")
                self._reset_backend_to_local()
                return
            if not self._confirm_cloud_use():
                self._reset_backend_to_local()
                return
            self.settings.setValue("provider", "cloud")
            self.models.setEnabled(False)
            self.models.clear()
            self.models.addItem(cloud.model)
            self.send_button.setEnabled(True)
        else:
            self._cloud_confirmed = False
            self.settings.setValue("provider", "local")
            self._start_discovery()

    def _confirm_cloud_use(self) -> bool:
        if self._cloud_confirmed:
            return True
        answer = QMessageBox.question(self, "Use optional cloud AI?",
            "Cloud mode sends the messages in this conversation to the configured provider. "
            "The provider may retain or process them under its own terms. This is not local/private processing. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        self._cloud_confirmed = answer == QMessageBox.Yes
        return self._cloud_confirmed

    def _reset_backend_to_local(self) -> None:
        self._cloud_confirmed = False
        self._backend_guard = True
        self.backend.setCurrentIndex(self.backend.findData("local"))
        self._backend_guard = False

    def _show_hardware(self) -> None:
        if not self.snapshot:
            QMessageBox.information(self, "FRIDAY hardware check", "Hardware check is running in the background.")
            return
        snapshot = self.snapshot
        fits = estimate_model_fits(snapshot, self.installed_models, self._profile_key())
        recommendation = recommend_model(fits, self._profile_key())
        lines = [
            f"System RAM: {snapshot.ram_total_bytes / 1024**3:.1f} GB total; "
            f"{snapshot.ram_available_bytes / 1024**3:.1f} GB currently available" if snapshot.ram_total_bytes and snapshot.ram_available_bytes else "System RAM: unavailable",
            f"CPU architecture: {snapshot.cpu_architecture}; logical cores: {snapshot.cpu_count}",
            f"GPU: {snapshot.gpu_name or 'not detected (CPU-only supported)'}",
            f"GPU type: {snapshot.gpu_type or 'none detected'}; reported VRAM: " + (f"{snapshot.vram_bytes / 1024**3:.1f} GB" if snapshot.vram_bytes else "unknown"),
            f"Ollama installed: {'yes' if snapshot.ollama_installed else 'no'}; service responding: {'yes' if snapshot.ollama_running else 'no'}",
            f"Installed-model recommendation for {PROFILES[self._profile_key()].display_name}: {recommendation or 'no confident recommendation from available installed models'}",
            "Model fit estimates use Ollama-reported disk size plus a conservative runtime/context allowance; they are advisory, not a compatibility guarantee. No model is downloaded automatically.",
            "Before manually installing a model, check its Ollama Library page for the published download size and leave additional free disk space for temporary downloads and other model layers. Actual RAM/VRAM needs depend on quantization, context, OS, and runtime offload.",
            "Installed Ollama models and their reported disk sizes: " + ("; ".join(f"{m.name}: {m.size_bytes / 1024**3:.2f} GB" if m.size_bytes else f"{m.name}: size unavailable" for m in self.installed_models) if self.installed_models else "none detected"),
            "Local mode keeps ordinary chat on this PC. Optional cloud mode is disabled until configured and explicitly selected; if used, conversation messages are sent to that provider.",
            "If Ollama is missing, install the official Windows Ollama app, then start it and choose an already available model. Benchmark only runs for an installed local model.",
        ]
        if snapshot.notes:
            lines.extend(["", "Diagnostics:", *snapshot.notes])
        lines.extend(["", "Profile settings bound context and output length; they do not promise response speed."])
        QMessageBox.information(self, "FRIDAY hardware & setup", "\n\n".join(lines))

    def _run_benchmark(self) -> None:
        if self.worker is not None or self.benchmark_worker is not None:
            return
        if not self.snapshot or not self.snapshot.ollama_running or not self.models.currentText() or self.backend.currentData() != "local":
            QMessageBox.information(self, "FRIDAY benchmark", "Start Ollama and select an installed local model before benchmarking.")
            return
        profile = PROFILES[self._profile_key()]
        self._set_busy(True)
        self.benchmark_result.setText("Running one short local prompt in a background thread…")
        self.benchmark_worker = BenchmarkWorker(self.client, self.models.currentText(),
            {"num_ctx": profile.num_ctx, "num_predict": min(profile.num_predict, 48), "temperature": 0.2})
        self.benchmark_worker.completed.connect(self._benchmark_complete)
        self.benchmark_worker.failed.connect(self._benchmark_failed)
        self.benchmark_worker.finished.connect(self._benchmark_stopped)
        self.benchmark_worker.start()

    def _benchmark_complete(self, elapsed: float, first_chunk: float, text: str) -> None:
        self.benchmark_result.setText(f"Measured: first output {first_chunk:.2f}s; total {elapsed:.2f}s; {len(text)} characters. One short run; not a speed guarantee.")

    def _benchmark_failed(self, message: str) -> None:
        self.benchmark_result.setText("Benchmark failed: " + message)

    def _benchmark_stopped(self) -> None:
        self.benchmark_worker = None
        self._set_busy(self.worker is not None)

    def _can_send(self) -> bool:
        if self.backend.currentData() == "cloud":
            return OpenAICompatibleClient().configured
        return bool(self.models.currentText()) and bool(self.snapshot and self.snapshot.ollama_running)

    def _set_busy(self, busy: bool) -> None:
        self.backend.setEnabled(not busy)
        self.profile.setEnabled(not busy)
        self.models.setEnabled(not busy and self.backend.currentData() == "local" and bool(self.installed_models))
        self.history.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.send_button.setEnabled(not busy and self._can_send())
        self.benchmark_button.setEnabled(not busy)
        if self.new_button:
            self.new_button.setEnabled(not busy)

    def _refresh_history(self) -> None:
        self.history.clear()
        for convo in self.store.list_conversations():
            self.history.addItem(convo["title"])
            self.history.item(self.history.count() - 1).setData(256, convo["id"])

    def _render_messages(self) -> None:
        parts = []
        for msg in self.store.get_messages(self.conversation_id):
            label = "You" if msg["role"] == "user" else "FRIDAY"
            body = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            parts.append(f"<p><b>{label}</b><br>{body}</p>")
        if self.streamed_text:
            body = (self.streamed_text.replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;").replace("\n", "<br>"))
            parts.append(f"<p><b>FRIDAY</b><br>{body}</p>")
        self.transcript.setHtml("".join(parts))

    def _new_conversation(self) -> None:
        self.conversation_id = self.store.create_conversation()
        self._refresh_history()
        self._render_messages()

    def _open_conversation(self, item) -> None:
        self.conversation_id = item.data(256)
        self._render_messages()

    def _send(self) -> None:
        text = self.input.text().strip()
        if not text or self.worker is not None or self.benchmark_worker is not None:
            return
        provider = self.backend.currentData()
        profile = PROFILES[self._profile_key()]
        char_budget = prompt_char_budget(profile.num_ctx)
        if len(text) > char_budget:
            QMessageBox.warning(self, "FRIDAY context limit",
                                f"This message exceeds the conservative {char_budget}-character input limit for the {profile.display_name} profile. Shorten it or choose a larger context profile.")
            return
        self.store.add_message(self.conversation_id, "user", text)
        self.input.clear()
        self._render_messages()
        if provider == "local" and (not self.models.currentText() or not self.snapshot or not self.snapshot.ollama_running):
            QMessageBox.information(self, "FRIDAY local AI", "Conversation history is available, but local inference is unavailable. Start/install Ollama and select an installed model, or explicitly configure the optional cloud provider.")
            self._refresh_history()
            return
        if provider == "cloud" and not OpenAICompatibleClient().configured:
            QMessageBox.warning(self, "FRIDAY cloud setup", "Cloud settings are incomplete; the message has been saved locally but not sent.")
            self._refresh_history()
            return
        if provider == "cloud" and not self._confirm_cloud_use():
            self._refresh_history()
            return
        messages = self.store.get_messages(self.conversation_id, limit=40)
        messages = bounded_context(messages, char_budget)
        self._start_generation(messages, provider, cpu_only=False)

    def _start_generation(self, messages: list[dict[str, str]], provider: str, cpu_only: bool) -> None:
        if self.worker is not None or self.benchmark_worker is not None:
            return
        profile = PROFILES[self._profile_key()]
        prompt_messages = [{"role": "system", "content": profile.system_instruction}, *messages]
        options: dict[str, int | float] = {"num_ctx": profile.num_ctx, "num_predict": profile.num_predict, "temperature": 0.2}
        if cpu_only and provider == "local":
            options["num_gpu"] = 0
        model = self.models.currentText() if provider == "local" else OpenAICompatibleClient().model
        self.streamed_text = ""
        self.active_response_conversation_id = self.conversation_id
        self.active_messages = messages
        self.last_failed_conversation_id = None
        self.worker = ChatWorker(self.client, model, prompt_messages, options, provider)
        self.worker.chunk.connect(self._append_chunk)
        self.worker.failed.connect(self._show_error)
        self.worker.finished_text.connect(self._finish_response)
        self.worker.finished.connect(self._worker_stopped)
        self._set_busy(True)
        self.stop_button.setEnabled(True)
        self.cpu_retry_button.setVisible(False)
        self.worker.start()

    def _retry_cpu(self) -> None:
        if self.worker is None and self.benchmark_worker is None and self.active_messages:
            if self.last_failed_conversation_id != self.conversation_id:
                QMessageBox.information(self, "FRIDAY CPU retry", "Return to the conversation where the GPU error occurred before retrying.")
                return
            budget = prompt_char_budget(PROFILES[self._profile_key()].num_ctx)
            if self.active_messages and len(self.active_messages[-1].get("content", "")) > budget:
                QMessageBox.information(self, "FRIDAY CPU retry", "The last prompt exceeds the selected profile's conservative context budget. Choose a larger profile before retrying.")
                return
            self._start_generation(bounded_context(self.active_messages, budget), "local", cpu_only=True)

    def _append_chunk(self, chunk: str) -> None:
        self.streamed_text += chunk
        self._render_messages()

    def _show_error(self, message: str, partial_text: str = "") -> None:
        gpu_error = is_gpu_initialization_error(message)
        self.cpu_retry_button.setVisible(gpu_error and self.backend.currentData() == "local")
        target = self.active_response_conversation_id
        if target:
            self.last_failed_conversation_id = target
            if partial_text.strip():
                self.store.add_message(target, "assistant", partial_text)
                self.streamed_text = ""
                self._render_messages()
                self._refresh_history()
        QMessageBox.warning(self, "FRIDAY — inference error", message + (
            "\n\nIf this is a GPU/driver initialization problem, you can explicitly retry the last request on CPU. CPU mode may be substantially slower." if gpu_error else ""))

    def _finish_response(self, text: str) -> None:
        if text.strip():
            target = self.active_response_conversation_id or self.conversation_id
            self.store.add_message(target, "assistant", text)
        self.streamed_text = ""
        self._render_messages()
        self._refresh_history()

    def _stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self.stop_button.setEnabled(False)

    def _worker_stopped(self) -> None:
        self.worker = None
        self.streamed_text = ""
        self.active_response_conversation_id = None
        self._render_messages()
        self._set_busy(self.benchmark_worker is not None)
        self.stop_button.setEnabled(False)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("FRIDAY")
    window = FridayWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
