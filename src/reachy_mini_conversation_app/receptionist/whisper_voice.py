"""Local speech recognition with OpenAI Whisper for enrollment and voice intent detection.

Whisper: https://github.com/openai/whisper  (MIT licence)
Default model: base.en — English-only, fast, ~74M parameters, ~1 GB VRAM.
Alternatives  : tiny.en (fastest), small.en (better accuracy), turbo (best GPU),
                medium.en, large (multilingual).

Install: pip install '.[receptionist]'  (openai-whisper + ffmpeg on PATH)
ffmpeg : https://ffmpeg.org/download.html
"""

from __future__ import annotations
import re
import logging

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

_TARGET_SR = 16000


def _normalize_phrase(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


class WhisperTranscriber:
    """Transcribe short mic buffers; used to store and match a spoken passphrase."""

    def __init__(self, model_size: str = "base", device: str | None = None) -> None:
        try:
            import whisper  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "openai-whisper is required for receptionist voice (phrase) recognition. "
                "Install with: pip install '.[receptionist]' (requires ffmpeg on PATH). "
                "See https://github.com/openai/whisper",
            ) from e

        self._mod = whisper
        dev = device
        if dev is None:
            try:
                import torch

                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"
        self._model = whisper.load_model(model_size, device=dev)
        logger.info("Whisper model %s on %s", model_size, dev)

    def transcribe_pcm(
        self,
        samples_int16: NDArray[np.int16],
        sample_rate: int,
        min_seconds: float = 0.8,
    ) -> str | None:
        """Return normalized transcript text, or None if audio is too short / empty."""
        if samples_int16 is None or len(samples_int16) < int(sample_rate * min_seconds):
            return None
        try:
            from scipy.signal import resample  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("scipy is required for Whisper resampling") from e

        x = samples_int16.astype(np.float32).reshape(-1) / 32768.0
        if sample_rate != _TARGET_SR:
            new_len = max(int(len(x) * _TARGET_SR / sample_rate), 1)
            x = resample(x, new_len).astype(np.float32)

        # Whisper expects float32 mono 16 kHz; clip to avoid overload
        x = np.clip(x, -1.0, 1.0)

        try:
            result = self._mod.transcribe(
                self._model,
                x,
                fp16=False,
                language="en",
                verbose=False,
            )
        except Exception as e:
            logger.warning("Whisper transcribe failed: %s", e)
            return None

        text = (result.get("text") or "").strip() if isinstance(result, dict) else ""
        if not text:
            return None
        norm = _normalize_phrase(text)
        return norm if norm else None


def phrase_similarity(a: str, b: str) -> float:
    """0–1 score between two normalized phrases."""
    from difflib import SequenceMatcher

    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def try_whisper_transcriber(model_size: str | None = None, device: str | None = None) -> WhisperTranscriber | None:
    try:
        from reachy_mini_conversation_app.config import config as _cfg

        size = model_size or getattr(_cfg, "WHISPER_MODEL", "base")
        return WhisperTranscriber(model_size=size, device=device)
    except ImportError:
        return None
    except Exception as e:
        logger.error("Failed to initialize Whisper: %s", e)
        return None
