# FRIDAY — Local AI Desktop Assistant

FRIDAY is a local-first Windows desktop assistant. Local chat uses Ollama; cloud inference is an optional OpenAI-compatible backend that is used only after it is configured and explicitly selected in the app. Conversation history stays in the local SQLite database.

## Requirements and run

- Windows 10/11 and Python 3.11+
- PySide6 and psutil (`requirements.txt`)
- Ollama is optional for opening FRIDAY and viewing history, but required for local AI responses

```powershell
py -0p
py -3.12 -m venv .venv
py -3.12 setup_friday.py
.\.venv\Scripts\python.exe -m app.main
```

`py -0p` lists Python versions registered with the Windows Python Launcher. The first command creates `.venv` with the selected interpreter; the setup script then installs requirements by invoking the venv's Python as a child process. If you use another supported Python 3.11+ version, replace `3.12` in both commands with that version (for example, `py -3.11`). The setup script reuses an existing `.venv` and refuses to replace an existing directory that lacks a venv interpreter. No activation or global package installation is required. `.gitignore` excludes generated environments, private `.env` files, local databases, and common model weights.

If Ollama is missing, install its official Windows application and start it. FRIDAY checks whether Ollama responds and lists installed models; it never pulls a model automatically. Install models separately and only after reviewing the Ollama Library download size and the free storage on your machine.

Chat data is stored at `%LOCALAPPDATA%\FRIDAY\friday.sqlite3`. Ollama calls default to `http://127.0.0.1:11434`.

## Local Windows process inspection

Ask FRIDAY to list or inspect running processes/applications (for example, “list my running applications”). FRIDAY reads process name, PID, executable path, and resident memory when Windows permits, using `psutil`. This operation is read-only and does not require administrator rights for ordinary listings. Fields blocked by Windows are reported as unavailable; processes that exit during collection are omitted. Large inventories are bounded to fit the selected model context and identify when the shown rows are truncated. FRIDAY routes these requests to local Ollama only—even if Cloud is selected—and never sends process metadata to a cloud provider. If local Ollama is unavailable or process inspection fails, FRIDAY reports that condition and does not ask a model to claim it succeeded.

## Hardware-aware setup

`Hardware & setup` reports system RAM and currently available RAM, CPU architecture/logical core count, GPU type/name and driver-reported VRAM when available, and whether Ollama is installed/responding. GPU probing is best effort; a missing GPU or driver does not prevent the app/history from opening. Windows adapter-reported VRAM may be inaccurate. If a probe fails, the UI records a diagnostic and continues.

For installed models, FRIDAY uses Ollama's reported model-file size plus a conservative context/runtime allowance to present an advisory fit estimate and recommendation. These are heuristics, not compatibility guarantees or vendor requirements: real use varies with quantization, context/KV cache, operating system, Ollama build, and GPU offload. An unknown measurement is shown as unknown. It does not fetch a model catalog or download models. Before installing an uninstalled model, check that model's published Ollama Library pull size; leave additional free disk for temporary downloads and other model layers. The size displayed for installed models is Ollama's reported on-disk size, not a guaranteed RAM/VRAM requirement.

The built-in one-prompt local benchmark measures first output latency, total elapsed time, and output character count on the selected model/profile. Run it on the target machine for an observed sample; one run is not a throughput guarantee. CPU inference may be slower and depends on the model and hardware.

## Performance profiles and explicit CPU retry

Choose **Fast**, **Balanced**, or **Low-resource** in the UI. They bound Ollama context (`num_ctx`) and output (`num_predict`) and adjust response style; the displayed settings are not promises of speed. Fast prefers concise answers, Balanced allows more context/output, and Low-resource uses the most conservative context/output budget. Recommendations consider available RAM, reported model size, and VRAM when measurable; the selected local model remains user-configurable.

Ollama can run CPU-only. If an inference error appears to involve GPU/CUDA/driver initialization, FRIDAY exposes a **Retry last request using CPU only** action. It makes a single explicit retry with `num_gpu: 0`; FRIDAY does not silently switch or repeatedly retry. CPU behavior is runtime/model dependent.

## Optional cloud provider

Local Ollama is the default and needs no cloud key. Cloud mode supports OpenAI-compatible chat-completions endpoints when all of these environment variables are configured before starting FRIDAY:

- `FRIDAY_CLOUD_BASE_URL` — API base, for example an OpenAI-compatible `/v1` endpoint
- `FRIDAY_CLOUD_MODEL` — provider model identifier
- `FRIDAY_CLOUD_API_KEY` — provider key

FRIDAY does not save the key in its settings or send conversation data while Local is selected. Selecting Cloud requires an in-app confirmation explaining that conversation messages go to the configured provider and are subject to its retention/processing terms. The provider may charge for usage; check its terms. Cloud is never required for offline operation.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite includes local mock-server tests for Ollama and the OpenAI-compatible streaming client, SQLite persistence, unavailable-runtime errors, selectable profile bounds, simulated no-GPU/driver-failure cases, and mocked Windows process-inspection results and failures. Simulating absent GPU validates graceful detection paths, **not real CPU inference performance**. A real GPU/CPU inference benchmark requires a running Ollama instance and installed model on each physical configuration.

## Windows distribution (experimental first build)

For an initial private/technical release, use the PyInstaller **one-folder** build. It bundles Python and the Qt/runtime dependencies so recipients do not need Python, while leaving files inspectable for troubleshooting. A one-file executable is not used because it unpacks dependencies at launch and is harder to diagnose. Builds are Windows/Python/architecture-specific; create a separate build on each target architecture. No application icon asset exists in this project, so the build uses the default executable icon.

Build from a Windows machine with Python 3.12 and an internet connection for pinned build dependencies:

```powershell
py -3.12 -m venv .venv
py -3.12 setup_friday.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build_windows.ps1
```

The `-ExecutionPolicy Bypass` applies only to this PowerShell process; it does not change the machine or user execution-policy setting. If `.venv` already exists but cannot launch, preserve it by renaming/archiving it first, then create a fresh `.venv` with the verified Python 3.12 interpreter before running setup.

The build script installs `requirements-build.txt` into the project venv and chooses a timestamped distribution root `dist\FRIDAY_<timestamp>` and work directory `build\pyinstaller_<timestamp>`. It prints both paths and refuses a timestamp collision; existing output directories are not deleted or overwritten. The executable is inside that root's `FRIDAY` one-folder directory. Direct build-tool versions are pinned; pip-resolved transitive dependencies are not yet locked, so builds are not guaranteed byte-for-byte reproducible. Both output trees are ignored by Git. Zip the complete generated `FRIDAY` directory to distribute it; do not send only the executable. PyInstaller's official documentation describes the one-folder layout and platform-specific build requirement. This is a portable app folder, **not an installer**, and it is not code-signed.

Recipients must install Ollama separately and install at least one compatible model themselves. FRIDAY does not bundle Ollama or model weights and never downloads a model automatically. Ollama's current Windows requirements and installer instructions are on its [Windows documentation](https://docs.ollama.com/windows). Start Ollama before local chat. If it is unavailable, history and setup remain accessible, while local inference is disabled. Before a first model pull, users should review that model's published download size and their available storage; actual memory/performance varies with model quantization, context, and hardware.

Conversation history remains under `%LOCALAPPDATA%\FRIDAY\friday.sqlite3`, outside the application directory. Removing the portable app folder does not remove that database; back it up or remove it separately if desired. Normal operation does not require administrator privileges. Cloud mode remains optional and requires a user-configured key plus an in-app privacy confirmation. Do not place provider keys in build settings or redistribute them.

Before public distribution, resolve licensing for the FRIDAY application and inventory every bundled dependency and Qt component. Qt open-source distribution has LGPL/GPL notice, replacement/relinking, and source-availability requirements; the applicable terms depend on the exact components and distribution method. This project does not yet contain a complete third-party license bundle or an application license. Obtain an appropriate licensing review and add notices before public release. PyInstaller's own license exception permits bundling but does not override dependency licenses. Also add a chosen version/release notes, test on a clean supported Windows installation and each target architecture, and obtain/sign with a trusted code-signing certificate if SmartScreen reputation is needed. Building alone does not establish trust or eliminate antivirus false positives.

The executable build is experimental until it has been smoke-tested on a separate clean Windows machine. The automated source tests and an on-device packaged smoke test do not prove compatibility on all Windows systems. A signed installer, automatic updates, telemetry, and model bundling are not part of this project.
