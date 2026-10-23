# FRIDAY — Local AI Desktop Assistant

FRIDAY is a local-first Windows desktop assistant. Local chat uses Ollama; cloud inference is an optional OpenAI-compatible backend that is used only after it is configured and explicitly selected in the app. Conversation history stays in the local SQLite database.

## Requirements and run

- Windows 10/11 and Python 3.11+
- PySide6 (`requirements.txt`)
- Ollama is optional for opening FRIDAY and viewing history, but required for local AI responses

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.main
```

If Ollama is missing, install its official Windows application and start it. FRIDAY checks whether Ollama responds and lists installed models; it never pulls a model automatically. Install models separately and only after reviewing the Ollama Library download size and the free storage on your machine.

Chat data is stored at `%LOCALAPPDATA%\FRIDAY\friday.sqlite3`. Ollama calls default to `http://127.0.0.1:11434`.

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
python -m unittest discover -s tests -v
```

The suite includes local mock-server tests for Ollama and the OpenAI-compatible streaming client, SQLite persistence, unavailable-runtime errors, selectable profile bounds, and simulated no-GPU/driver-failure cases. Simulating absent GPU validates graceful detection paths, **not real CPU inference performance**. A real GPU/CPU inference benchmark requires a running Ollama instance and installed model on each physical configuration.
