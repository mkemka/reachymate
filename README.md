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




---

## Receptionist Mode — Complete Setup Guide

Receptionist mode turns Reachy Mini into a biometric access gate for coworking spaces or any venue that needs face + voice check-in tied to a balance ("Roo points").

### What it does

| Subsystem | Technology | What it does |
|-----------|-----------|--------------|
| **Face detection** | YOLO26x (Ultralytics) | Detects faces in Reachy's camera feed |
| **Face embedding** | InsightFace `buffalo_l` | 512-d L2-normalised embedding for identity matching |
| **Voice transcription** | OpenAI Whisper `base.en` | Transcribes spoken passphrase and yes/no intent |
| **State machine** | Deterministic controller | 3-second check-in budget, no LLM in critical path |
| **Ledger** | Local SQLite (WAL mode) | Thread-safe, idempotent Roo points deductions |
| **Dashboard** | FastAPI + vanilla JS | Live camera, identity card, enroll + members tabs |

**Check-in flow (≤ 3 seconds):**
1. Reachy detects a face in the camera (autonomous background loop, polls every 0.5 s).
2. InsightFace embeds the face and looks up the closest match.
3. *High confidence (≥ 0.90):* skip voice, ask yes/no in 1.0 s.
4. *Medium confidence (0.35–0.90):* also transcribe voice passphrase.
5. If identity confirmed **and** visitor says yes: deduct 1 Roo point (idempotent), unlock session.
6. A wake phrase (default: `"hey reachy"`) then lets the visitor talk to the assistant.

**Non-negotiables:**
- No cloud in the critical path. All ML inference runs locally.
- Points are only deducted after identity AND intent are confirmed.
- Ledger deductions are idempotent via a unique `interaction_id` (no double-charge on retry).

---

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | 3.12 recommended |
| [uv](https://docs.astral.sh/uv/) | Package manager |
| **ffmpeg** on `PATH` | Required by Whisper for audio decode |
| A camera | Reachy Mini's built-in camera, or any webcam |
| OpenAI API key | For the conversation layer (not used in check-in itself) |

**Install ffmpeg:**

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
# Or download from https://ffmpeg.org/download.html and add to PATH
```

---

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/ShriabhayS/reachymate.git
cd reachymate

# 2. Create virtual environment
uv venv --python 3.12
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# 3. Install with receptionist extras
uv pip install -e ".[receptionist]"

# (Optional) also install YOLO head tracker for better face detection
uv pip install -e ".[receptionist,yolo_vision]"
```

The `receptionist` extra installs: `insightface`, `onnxruntime`, `openai-whisper`, `scipy`, `torch`, `ultralytics`, `supervision`.

> **First run:** InsightFace will automatically download `buffalo_l` (~280 MB) to `~/.insightface/`. YOLO26x (~119 MB) is downloaded on first use. Whisper `base.en` (~140 MB) downloads the first time it transcribes audio. All models are cached and reused on subsequent runs.

---

### Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Key receptionist variables (all optional — defaults work out of the box):

```env
# ── Receptionist mode ────────────────────────────────────────
RECEPTIONIST_MODE=true                # auto-enabled by --receptionist flag

# Profiles / Models
YOLO_FACE_MODEL=yolo26x.pt            # most accurate; use yolo26n.pt for speed
WHISPER_MODEL=base.en                 # English-only; try small.en for accuracy
INSIGHTFACE_MODEL_NAME=buffalo_l      # best open-source face embed model

# Thresholds (cosine similarity, 0–1)
RECEPTIONIST_FACE_THRESHOLD=0.35      # minimum similarity to consider a match
RECEPTIONIST_VOICE_THRESHOLD=0.60     # minimum voice passphrase similarity
FACE_CONFIDENCE_HIGH_THRESHOLD=0.90   # skip voice check above this score

# Session / points
RECEPTIONIST_SESSION_TTL_S=300        # seconds before session expires (5 min)
RECEPTIONIST_BUFFER_SECONDS=6         # seconds of audio buffer to keep
CHECK_IN_COST_POINTS=1                # Roo points deducted per check-in
LEDGER_INITIAL_BALANCE=100            # starting balance for new members

# Wake phrases (comma-separated, any triggers conversation)
RECEPTIONIST_WAKE_PHRASES=hey reachy,hello reachy,hi reachy
```

---

### Running the app

**Recommended (dashboard + YOLO tracker):**

```bash
python -m reachy_mini_conversation_app.main \
  --receptionist \
  --gradio \
  --head-tracker yolo
```

**Headless (no Gradio, API-only):**

```bash
python -m reachy_mini_conversation_app.main --receptionist
```

**Debug mode (verbose logging):**

```bash
python -m reachy_mini_conversation_app.main --receptionist --gradio --debug
```

Once running, open:
- **Settings page:** `http://localhost:7860/`
- **Receptionist dashboard:** `http://localhost:7860/receptionist/`

---

### Receptionist Dashboard

The dashboard at `http://localhost:7860/receptionist/` shows:

| Panel | What it shows |
|-------|--------------|
| **Left — Reachy's View** | Live camera feed (refreshes every second) with corner brackets overlay |
| **Left — Identity Card** | Current check-in state (IDLE / SCANNING / RECOGNISED / DENIED), recognised person's name, and their Roo points balance |
| **Left — Conversation Log** | Live transcript of everything said to / by Reachy |
| **Right — Enroll tab** | Enroll a new member: type their name, position them in front of the camera, have them say their passphrase, click Enroll |
| **Right — Members tab** | List all enrolled members with their Roo point balances; add points or delete members |

The dashboard polls the backend automatically — no manual refresh needed.

---

### Enrolling a new member

1. Open `http://localhost:7860/receptionist/` and click the **Enroll** tab.
2. Position the person in front of Reachy's camera — their face should be clearly visible.
3. Have them say their passphrase out loud for 2–4 seconds (any phrase works; it's stored as a voice fingerprint).
4. Type their display name in the **Display Name** field.
5. Click **Capture + Enroll**.

Reachy simultaneously:
- Captures the current camera frame and runs InsightFace to extract a 512-d face embedding.
- Transcribes the last few seconds of audio with Whisper to capture the passphrase.
- Saves both to the enrollment store (SQLite + `receptionist_data/`).
- Creates a ledger entry with `LEDGER_INITIAL_BALANCE` Roo points.

**Tips:**
- Enroll 2–3 different angles (straight-on, slight left/right) by enrolling the same person twice with the same name. The second enrollment replaces the first.
- Good lighting improves face embedding quality significantly.
- Speak clearly at normal volume — Whisper handles accents well.

---

### Adding Roo points (admin)

In the **Members** tab, find the member card and use the **+ Add Points** form.
Via API directly:

```bash
curl -X POST http://localhost:7860/receptionist/people/<person_id>/add_points \
  -H "Content-Type: application/json" \
  -d '{"points": 50}'
```

---

### Cross-workstation portability

The app is designed to run identically on any machine that has the repo and dependencies installed:

1. **Clone the same repo** on the new workstation.
2. **Copy your `.env` file** (contains your OpenAI API key and settings).
3. **Copy `receptionist_data/`** (enrollment store — SQLite DB + embeddings). This folder is created in your working directory when you first run `--receptionist`.
4. Install dependencies: `uv pip install -e ".[receptionist,yolo_vision]"`
5. Start the app.

The enrollment store is a single SQLite file (`receptionist_data/enrollment.db`) plus small numpy embedding files. All ML models download automatically on first use and cache in standard OS locations (`~/.insightface/`, `~/.cache/whisper/`, Ultralytics `~/ultralytics/`).

> **Shared store tip:** If multiple machines share a NFS mount or cloud-synced folder (e.g. Dropbox), point `RECEPTIONIST_DATA_DIR` to that shared path and all machines see the same enrollments in real-time.

---

### File structure (receptionist-specific)

```
src/reachy_mini_conversation_app/
├── receptionist/
│   ├── __init__.py
│   ├── face_embed.py          # InsightFace buffalo_l — 512-d face embeddings
│   ├── whisper_voice.py       # Whisper transcription (passphrase + intent)
│   ├── audio_intent.py        # Bounded yes/no detector (1.0 s window)
│   ├── controller.py          # Deterministic 3-second check-in state machine
│   ├── gate.py                # Session management, audio buffer, wake-phrase gate
│   ├── ledger.py              # Local SQLite Roo-points ledger (idempotent)
│   ├── stack.py               # Wires all components together at startup
│   └── store.py               # Enrollment store (face + voice per person)
├── receptionist_dashboard.py  # FastAPI routes: /receptionist/* REST + HTML page
├── tools/
│   └── ledger_balance.py      # LLM tool: query/list Roo point balances
├── profiles/receptionist/
│   ├── instructions.txt       # System prompt: silent until addressed, etc.
│   └── tools.txt              # Enabled tools for this profile
└── static/
    ├── receptionist.html      # Dashboard HTML (served at /receptionist/)
    ├── receptionist.css       # MLAI.AU-inspired brutalist styles
    └── receptionist.js        # Dashboard JS (polling, enroll, members)

receptionist_data/             # Created at runtime (gitignored)
├── enrollment.db              # SQLite: members + faces + voices + ledger
└── embeddings/                # Numpy face embedding arrays
```

---

### Troubleshooting

**"No face embedding" on enroll:**
- Ensure Reachy's camera is connected and the feed shows in the dashboard.
- Make sure the person's face is clearly visible and well-lit.
- The YOLO model needs to detect a face first — try `--head-tracker yolo`.

**"Voice not understood" on enroll:**
- Verify `ffmpeg` is on your `PATH`: `ffmpeg -version`.
- Speak louder and more clearly for 3–4 seconds.
- Try `WHISPER_MODEL=small.en` for better accuracy at the cost of speed.

**Face not recognised (DENIED) when it should match:**
- Lower `RECEPTIONIST_FACE_THRESHOLD` (e.g. `0.28`) to make matching more lenient.
- Re-enroll with better lighting or a closer shot.
- Check `FACE_CONFIDENCE_HIGH_THRESHOLD` — if set too high, voice check kicks in.

**Points not deducted / session not unlocking:**
- Enable debug logging: `--debug` flag, look for `Autonomous check-in` log lines.
- Verify Roo points balance is > 0 in the Members tab.

**Dashboard shows "Receptionist Mode Inactive":**
- The app must be started with `--receptionist` flag.
- If using `reachy-convo-mate` CLI, set `RECEPTIONIST_MODE=true` in `.env`.

---

## Development workflow
- Install the dev group extras: `uv sync --group dev` or `pip install -e .[dev]`.
- Run formatting and linting: `ruff check .`.
- Execute the test suite: `pytest`.
- When iterating on robot motions, keep the control loop responsive => offload blocking work using the helpers in `tools.py`.

## License
Apache 2.0
