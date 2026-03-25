"""Bounded yes/no audio intent detector.

Reads a slice of audio that arrives *after* a prompt is given (so it does not
accidentally match audio spoken before the prompt) and classifies it as one of:

  confirmed = "yes"     — affirmative keyword detected
  confirmed = "no"      — negative keyword detected
  confirmed = "timeout" — window elapsed with no clear signal

The detector is intentionally simple and local-first: Whisper transcribes the
audio and a keyword lookup decides the intent. No cloud calls.
"""

from __future__ import annotations

import time
import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from reachy_mini_conversation_app.receptionist.whisper_voice import WhisperTranscriber

logger = logging.getLogger(__name__)

# Keyword sets (lower-case, normalised by Whisper)
_AFFIRMATIVE: frozenset[str] = frozenset(
    {
        "yes", "yeah", "yep", "yup", "sure", "okay", "ok",
        "check", "in", "confirm", "go", "correct", "right",
        "indeed", "absolutely", "aye", "affirmative",
    }
)
_NEGATIVE: frozenset[str] = frozenset(
    {
        "no", "nope", "nah", "stop", "cancel", "negative",
        "dont", "not", "never", "reject",
    }
)

# Minimum audio to attempt transcription (seconds)
_MIN_AUDIO_S = 0.3


class AudioIntentDetector:
    """Detect yes/no intent from a bounded audio window.

    Parameters
    ----------
    transcriber:
        A :class:`~reachy_mini_conversation_app.receptionist.whisper_voice.WhisperTranscriber`
        instance (already warm / loaded).
    """

    def __init__(self, transcriber: "WhisperTranscriber") -> None:
        self.transcriber = transcriber

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def detect(
        self,
        audio_buffer: NDArray[np.int16],
        sample_rate: int,
        window_seconds: float = 1.0,
        deadline_monotonic: float | None = None,
    ) -> dict[str, str]:
        """Classify intent from ``audio_buffer``.

        Parameters
        ----------
        audio_buffer:
            Full recent audio (int16 mono). The detector only uses the last
            ``window_seconds`` of audio to respect the timing budget.
        sample_rate:
            Sample rate of ``audio_buffer`` in Hz.
        window_seconds:
            How many seconds from the tail of ``audio_buffer`` to inspect.
        deadline_monotonic:
            If ``time.monotonic()`` is already past this value, returns
            ``"timeout"`` immediately without transcribing.

        Returns
        -------
        dict with key ``"confirmed"`` → ``"yes"``, ``"no"``, or ``"timeout"``.
        """
        if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
            logger.debug("AudioIntent: deadline already past — timeout")
            return {"confirmed": "timeout"}

        # Crop to last `window_seconds`
        n_samples = int(window_seconds * sample_rate)
        window = audio_buffer[-n_samples:] if len(audio_buffer) > n_samples else audio_buffer

        min_samples = int(_MIN_AUDIO_S * sample_rate)
        if len(window) < min_samples:
            logger.debug("AudioIntent: audio too short (%d samples) — timeout", len(window))
            return {"confirmed": "timeout"}

        phrase = self.transcriber.transcribe_pcm(window, sample_rate, min_seconds=_MIN_AUDIO_S)
        if phrase is None:
            logger.debug("AudioIntent: transcription returned None — timeout")
            return {"confirmed": "timeout"}

        words = set(phrase.lower().split())
        logger.debug("AudioIntent: heard=%r words=%s", phrase, words)

        if words & _AFFIRMATIVE:
            return {"confirmed": "yes"}
        if words & _NEGATIVE:
            return {"confirmed": "no"}
        return {"confirmed": "timeout"}

    def detect_from_gate_live(
        self,
        gate: object,
        window_seconds: float = 1.0,
    ) -> dict[str, str]:
        """Detect intent by reading *new* audio that arrives after this call.

        Records the current audio buffer length as a baseline, sleeps for
        ``window_seconds`` to let new audio accumulate, then transcribes only the
        newly captured audio.  This avoids classifying speech that was spoken
        *before* the prompt was given.

        Parameters
        ----------
        gate:
            A :class:`~reachy_mini_conversation_app.receptionist.gate.ReceptionistGate`
            instance (duck-typed to avoid circular imports).
        window_seconds:
            How long to listen (seconds).

        Returns
        -------
        dict with key ``"confirmed"`` → ``"yes"``, ``"no"``, or ``"timeout"``.
        """
        # Snapshot current buffer length (samples already present before prompt)
        try:
            baseline_offset: int = len(gate.get_recent_audio())  # type: ignore[attr-defined]
            sample_rate: int = int(gate.input_sample_rate)  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning("AudioIntent: gate access error: %s", e)
            return {"confirmed": "timeout"}

        deadline = time.monotonic() + window_seconds
        time.sleep(window_seconds)

        try:
            full_audio: NDArray[np.int16] = gate.get_recent_audio()  # type: ignore[attr-defined]
        except Exception:
            return {"confirmed": "timeout"}

        new_audio = full_audio[baseline_offset:]
        min_samples = int(_MIN_AUDIO_S * sample_rate)
        if len(new_audio) < min_samples:
            logger.debug("AudioIntent live: only %d new samples — timeout", len(new_audio))
            return {"confirmed": "timeout"}

        phrase = self.transcriber.transcribe_pcm(new_audio, sample_rate, min_seconds=_MIN_AUDIO_S)
        if phrase is None:
            return {"confirmed": "timeout"}

        words = set(phrase.lower().split())
        logger.debug("AudioIntent live: heard=%r words=%s", phrase, words)

        if words & _AFFIRMATIVE:
            return {"confirmed": "yes"}
        if words & _NEGATIVE:
            return {"confirmed": "no"}
        return {"confirmed": "timeout"}
