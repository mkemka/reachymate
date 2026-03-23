---
title: Reachy Mini Conversation App
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: ReachyConvoMate — realtime voice, Gradio dashboard, optional receptionist gate
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# ReachyConvoMate (Reachy Mini conversation app)

**Package:** `reachy-convo-mate` · **CLI:** `reachy-convo-mate`

Conversational app for the Reachy Mini robot combining OpenAI’s realtime APIs, vision pipelines, choreographed motion, and a Gradio dashboard (live camera, chat log, personality editor, optional doctor-practice mode). Optional **receptionist** mode adds face + voice enrollment and verification (InsightFace, Whisper, YOLO).

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Architecture

The app follows a layered architecture connecting the user, AI services, and robot hardware:

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## Overview
- Real-time audio conversation loop powered by the OpenAI realtime API and `fastrtc` for low-latency streaming.
- Vision processing uses gpt-realtime by default (when camera tool is used), with optional local vision processing using SmolVLM2 model running on-device (CPU/GPU/MPS) via `--local-vision` flag.
- Layered motion system queues primary moves (dances, emotions, goto poses, breathing) while blending speech-reactive wobble and face-tracking.
- Async tool dispatch integrates robot motion, camera capture, and optional face-tracking through a Gradio web UI with live transcripts; register the app with the Reachy Mini daemon (see [`install.md`](install.md)).
- Optional **receptionist** profile and `--receptionist` flag: enroll and verify visitors with face embeddings and a spoken passphrase; data under `receptionist_data/` (requires `pip install -e ".[receptionist]"` and **ffmpeg** on `PATH` for Whisper).

## Additional docs
- **[`install.md`](install.md)** — install as a daemon-discovered app (`reachy_mini_apps` entry point) and open the settings UI.
- **[`docs/microphone-and-camera.md`](docs/microphone-and-camera.md)** — where mic/camera live (robot vs PC), daemon + app startup, and troubleshooting “camera offline”.

## Installation

> [!IMPORTANT]
> Requires **Python 3.10+** (3.12 recommended; matches `pyproject.toml` and tooling) and [uv](https://docs.astral.sh/uv/). Windows support is experimental.

### Quickstart

```bash
# 1. Create a virtual environment
uv venv --python 3.12.1
source .venv/bin/activate

# 2. Install the Reachy Mini daemon (SDK + app server)
uv pip install reachy-mini

# 3. Clone and install this repo in editable mode
git clone https://github.com/mkemka/reachymate.git
cd reachymate
uv pip install -e .

# 4. Start the Reachy Mini daemon
reachy-mini
```

Once the daemon is running, open http://127.0.0.1:7860/ to access the settings dashboard. Enter your OpenAI API key on first launch -- it will be saved for subsequent runs.

### Optional extras

```bash
uv pip install -e ".[reachy_mini_wireless]"  # Wireless Reachy Mini (GStreamer)
uv pip install -e ".[local_vision]"          # Local vision (PyTorch/SmolVLM2)
uv pip install -e ".[yolo_vision]"           # YOLO face tracking
uv pip install -e ".[mediapipe_vision]"      # MediaPipe face tracking
uv pip install -e ".[all_vision]"            # All vision extras
uv pip install -e ".[receptionist]"          # Receptionist: InsightFace + Whisper + YOLO face stack
uv pip install -e ".[dev]"                   # Dev tools (pytest, ruff, mypy)
```

## Optional dependency groups

| Extra | Purpose | Notes |
|-------|---------|-------|
| `reachy_mini_wireless` | Wireless Reachy Mini with GStreamer support. | Required for wireless versions of Reachy Mini, includes GStreamer dependencies.
| `local_vision` | Run the local VLM (SmolVLM2) through PyTorch/Transformers. | GPU recommended; ensure compatible PyTorch builds for your platform.
| `yolo_vision` | YOLOv8 tracking via `ultralytics` and `supervision`. | CPU friendly; supports the `--head-tracker yolo` option.
| `mediapipe_vision` | Lightweight landmark tracking with MediaPipe. | Works on CPU; enables `--head-tracker mediapipe`.
| `all_vision` | Convenience alias installing every vision extra. | Install when you want the flexibility to experiment with every provider.
| `receptionist` | Face + voice gate for the receptionist profile (`--receptionist`). | Pulls InsightFace, ONNX Runtime, Whisper, SciPy, Torch, Ultralytics, Supervision; Whisper needs **ffmpeg** on `PATH`. |
| `dev` | Developer tooling (`pytest`, `ruff`). | Add on top of either base or `all_vision` environments.

## Configuration

1. Copy `.env.example` to `.env`.
2. Fill in the required values, notably the OpenAI API key.

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required. Grants access to the OpenAI realtime endpoint.
| `MODEL_NAME` | Override the realtime model (defaults to `gpt-realtime`). Used for both conversation and vision (unless `--local-vision` flag is used).
| `HF_HOME` | Cache directory for local Hugging Face downloads (only used with `--local-vision` flag, defaults to `./cache`).
| `HF_TOKEN` | Optional token for Hugging Face models (only used with `--local-vision` flag, falls back to `huggingface-cli login`).
| `LOCAL_VISION_MODEL` | Hugging Face model path for local vision processing (only used with `--local-vision` flag, defaults to `HuggingFaceTB/SmolVLM2-2.2B-Instruct`).

Profile and receptionist-related variables are documented in [`.env.example`](.env.example), including `REACHY_MINI_CUSTOM_PROFILE` (e.g. `receptionist`), `RECEPTIONIST_MODE`, `YOLO_FACE_MODEL`, `WHISPER_MODEL`, `RECEPTIONIST_VOICE_THRESHOLD`, `INSIGHTFACE_MODEL_NAME`, and `RECEPTIONIST_WAKE_PHRASES`.

## Running the app

Activate your virtual environment, ensure the Reachy Mini robot (or simulator) is reachable, then launch:

```bash
reachy-convo-mate
```

By default, the app runs in console mode for direct audio interaction. Use the `--gradio` flag to launch a web UI served locally at http://127.0.0.1:7860/ (required when running in simulation mode). With a camera attached, vision is handled by the gpt-realtime model when the camera tool is used. For local vision processing, use the `--local-vision` flag to process frames periodically using the SmolVLM2 model. Additionally, you can enable face tracking via YOLO or MediaPipe pipelines depending on the extras you installed. When using `--gradio`, the dashboard can include a **Doctor** panel for case-based practice (cases ship in-repo; see `medical.yaml` / `doctor_cases.py`).

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--head-tracker {yolo,mediapipe}` | `None` | Select a face-tracking backend when a camera is available. YOLO is implemented locally, MediaPipe comes from the `reachy_mini_toolbox` package. Requires the matching optional extra. |
| `--no-camera` | `False` | Run without camera capture or face tracking. |
| `--local-vision` | `False` | Use local vision model (SmolVLM2) for periodic image processing instead of gpt-realtime vision. Requires `local_vision` extra to be installed. |
| `--gradio` | `False` | Launch the Gradio web UI. Without this flag, runs in console mode. Required when running in simulation mode. |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |
| `--wireless-version` | `False` | Use GStreamer backend for wireless version of the robot. Requires `reachy_mini_wireless` extra to be installed. |
| `--on-device` | `False` | Use when the conversation app runs on the same machine as the Reachy Mini daemon. |
| `--receptionist` | `False` | Enable receptionist biometric gate (face + voice + wake phrases). Requires `receptionist` extra and a camera-capable setup. |


### Examples
- Run on hardware with MediaPipe face tracking:

  ```bash
  reachy-convo-mate --head-tracker mediapipe
  ```

- Run with local vision processing (requires `local_vision` extra):

  ```bash
  reachy-convo-mate --local-vision
  ```

- Run with wireless support (requires `reachy_mini_wireless` extra and daemon started with `--wireless-version`):

  ```bash
  reachy-convo-mate --wireless-version
  ```

- Disable the camera pipeline (audio-only conversation):

  ```bash
  reachy-convo-mate --no-camera
  ```

- Receptionist profile with biometric gate (install `receptionist` extra, set profile, ensure ffmpeg for Whisper):

  ```bash
  set REACHY_MINI_CUSTOM_PROFILE=receptionist
  reachy-convo-mate --gradio --receptionist
  ```
  On Unix: `export REACHY_MINI_CUSTOM_PROFILE=receptionist` before the same command.

### Troubleshooting

- Timeout error:
If you get an error like this:
  ```bash
  TimeoutError: Timeout while waiting for connection with the server.
  ```
It probably means that the Reachy Mini's daemon isn't running. Install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) and start the daemon.

## LLM tools exposed to the assistant

| Tool | Action | Dependencies |
|------|--------|--------------|
| `move_head` | Queue a head pose change (left/right/up/down/front). | Core install only. |
| `camera` | Capture the latest camera frame and send it to gpt-realtime for vision analysis. | Requires camera worker; uses gpt-realtime vision by default. |
| `head_tracking` | Enable or disable face-tracking offsets (not facial recognition - only detects and tracks face position). | Camera worker with configured head tracker. |
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Core install only. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face assets. | Needs `HF_TOKEN` for the recorded emotions dataset. |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `do_nothing` | Explicitly remain idle. | Core install only. |
| `receptionist_enroll` | Capture face + voice passphrase for a named visitor. | `--receptionist` and `receptionist` extra. |
| `receptionist_verify` | Verify visitor against enrolled face + passphrase. | `--receptionist` and `receptionist` extra. |
| `receptionist_list` | List enrolled visitors. | `--receptionist` and `receptionist` extra. |
| `receptionist_delete` | Remove an enrollment by `person_id`. | `--receptionist` and `receptionist` extra. |

## Using custom profiles
Create custom profiles with dedicated instructions and enabled tools! 

Set `REACHY_MINI_CUSTOM_PROFILE=<name>` to load `src/reachy_mini_conversation_app/profiles/<name>/` (see `.env.example`). If unset, the `default` profile is used.

Each profile requires two files: `instructions.txt` (prompt text) and `tools.txt` (list of allowed tools), and optionally contains custom tools implementations.

### Custom instructions
Write plain-text prompts in `instructions.txt`. To reuse shared prompt pieces, add lines like:
```
[passion_for_lobster_jokes]
[identities/witty_identity]
```
Each placeholder pulls the matching file under `src/reachy_mini_conversation_app/prompts/` (nested paths allowed). See `src/reachy_mini_conversation_app/profiles/example/` for a reference layout.

### Enabling tools
List enabled tools in `tools.txt`, one per line; prefix with `#` to comment out. For example:

```
play_emotion
# move_head

# My custom tool defined locally
sweep_look
```
Tools are resolved first from Python files in the profile folder (custom tools), then from the shared library `src/reachy_mini_conversation_app/tools/` (e.g., `dance`, `head_tracking`). 

### Custom tools
On top of built-in tools found in the shared library, you can implement custom tools specific to your profile by adding Python files in the profile folder. 
Custom tools must subclass `reachy_mini_conversation_app.tools.core_tools.Tool` (see `profiles/example/sweep_look.py`).

### Edit personalities from the UI
When running with `--gradio`, open the “Personality” accordion:
- Select among available profiles (folders under `src/reachy_mini_conversation_app/profiles/`) or the built‑in default.
- Click “Apply” to update the current session instructions live.
- Create a new personality by entering a name and instructions text; it stores files under `profiles/<name>/` and copies `tools.txt` from the `default` profile.

Note: The “Personality” panel updates the conversation instructions. Tool sets are loaded at startup from `tools.txt` and are not hot‑reloaded.




## Development workflow
- Install the dev group extras: `uv sync --group dev` or `pip install -e .[dev]`.
- Run formatting and linting: `ruff check .`.
- Execute the test suite: `pytest`.
- When iterating on robot motions, keep the control loop responsive => offload blocking work using the helpers in `tools.py`.

## License
Apache 2.0
