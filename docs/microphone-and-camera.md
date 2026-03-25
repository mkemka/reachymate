# Getting microphone and camera working

In this app, **microphone and camera are on the Reachy Mini robot** (your Reachy Mini Lite), not on your PC. Audio and video are sent from the robot to the daemon over USB, then the conversation app uses them.

## Checklist

### 1. Robot connected
- Reachy Mini Lite is powered on and connected to your PC via **USB-C**.
- On Windows, the robot may show up in Device Manager (e.g. under “Portable Devices” or “Sound, video and game controllers” if it exposes USB audio).

### 2. Daemon running and connected
- In a **first terminal**, start the daemon with the venv activated:
  ```powershell
  cd C:\Users\shria\Desktop\reachymate
  .\.venv\Scripts\Activate.ps1
  reachy-mini-daemon
  ```
- Leave it running. You should see it connect to the robot (no timeout error). If you see “Timeout while waiting for connection”, the robot is not detected — check USB cable and port, and try another USB port if needed.

### 3. Start the conversation app
- In a **second terminal**:
  ```powershell
  cd C:\Users\shria\Desktop\reachymate
  .\.venv\Scripts\Activate.ps1
  reachy-convo-mate --gradio
  ```
- When the app starts, it connects to the daemon and then starts the robot’s **recording** (microphone) and **playback** (speaker). The camera worker also starts and pulls frames from the robot’s camera. You do not need to click anything in the browser to “start” mic or camera — they start when the app launches (after the API key is set).

### 4. Open the web UI
- Open the URL printed in the terminal (e.g. http://127.0.0.1:7860/) in your browser.
- If the API key was not in `.env`, enter it in the settings page when prompted. After that, the stream (mic + speaker + camera pipeline) is already running.

## What you should see

- **Camera**: The “Reachy’s camera” / camera feed area in the UI should show the robot’s view. If it shows “Camera offline”, no frames are coming from the robot (see troubleshooting below).
- **Microphone**: You talk to the robot; your voice is captured by the **robot’s microphones**, sent to the app, then to OpenAI. The robot’s **speaker** plays the assistant’s voice. There is no browser permission for mic — it’s all server-side from the robot.

## If the camera stays “offline”

- Confirm the **daemon** is running and connected (no timeout at startup).
- Confirm the **conversation app** was started **after** the daemon (so it connects to an already-connected robot).
- Restart in this order: stop the conversation app (Ctrl+C) → leave daemon running → start `reachy-convo-mate --gradio` again.
- On Windows, if the robot uses USB video, ensure no other app has exclusive access to the camera device.

## If the microphone or speaker doesn’t work

- Same as above: daemon must be running and connected, then start the conversation app.
- On Windows, the robot may appear as a **USB audio device**. In **Sound settings** (or “Manage sound devices”), check that the Reachy Mini device is present and not disabled. You do not need to set it as the default device for the app to use it — the daemon talks to the robot directly over USB.
- If the daemon or robot SDK needs specific Windows drivers or permissions, refer to the official Reachy Mini / Reachy Mini Lite docs for your setup.

## Summary

| Item        | Where it comes from        | What you need to do |
|------------|----------------------------|----------------------|
| Microphone | Robot’s mics (Reachy Lite) | Daemon + robot connected, then run app |
| Speaker    | Robot’s speaker             | Same as above        |
| Camera     | Robot’s head camera        | Same as above        |

No browser permissions for mic/camera are required; everything goes through the daemon and the robot.
