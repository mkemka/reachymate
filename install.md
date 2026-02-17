# ReachyConvoMate -- Install for Reachy Mini Daemon

## Prerequisites

- Reachy Mini SDK installed (`reachy_mini >= 1.2.3rc1`)
- Reachy Mini daemon running (`reachy-mini-daemon`)
- Python 3.10+

## Install the app

```bash
# From the repo root
pip install -e .

# Or with vision extras
pip install -e ".[all_vision]"
```

The package registers itself as a daemon app via the `reachy_mini_apps` entry point in `pyproject.toml`. No extra config needed.

## Verify daemon sees it

Restart the daemon so it discovers the new entry point:

```bash
sudo systemctl restart reachy-mini-daemon
```

Or if running manually:

```bash
reachy-mini-daemon
```

ReachyConvoMate should appear in the daemon dashboard under **Applications**. Click it to start.

## Access the settings UI

Once the app is running, the daemon serves its settings page at:

```
http://<reachy-ip>:7860/
```

The configuration screen includes:

- **Reachy's view** -- live camera feed (~2 fps), with a "Take photo" button to capture and save snapshots
- **Chat log** -- live iOS-style transcript of the conversation between the user and Reachy
- **Personality studio** -- prompt editor, tools, and voice selection

## CLI (standalone, without daemon)

```bash
reachy-convo-mate
```

## OpenAI API key

Three options (in priority order):

1. **Settings UI** -- paste the key in the web form at `http://<reachy-ip>:7860/`
2. **Environment** -- `export OPENAI_API_KEY=sk-...` before starting the daemon
3. **`.env` file** -- place a `.env` in the app instance path with `OPENAI_API_KEY=sk-...`

The settings UI persists the key to the instance `.env` automatically.

## Troubleshooting

**Camera feed shows "Camera offline"** -- the camera worker hasn't started yet or `--no-camera` was passed. Ensure the robot's camera is connected and the app was not launched with `--no-camera`.

**Chat log stays empty** -- the realtime session hasn't connected. Check that the OpenAI API key is valid and the daemon logs show `Realtime session updated successfully`.

**App doesn't appear in daemon** -- run `pip show reachy-convo-mate` to confirm it's installed in the same Python environment as the daemon. The entry point `reachy_mini_apps` must be discoverable.
