"""Persist enrolled face embeddings and Whisper-derived voice phrases per person."""

from __future__ import annotations
import re
import json
import time
import logging
from typing import Any, Iterator
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from reachy_mini_conversation_app.receptionist.whisper_voice import phrase_similarity


logger = logging.getLogger(__name__)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower()).strip("-")
    return s or "person"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class EnrolledPerson:
    """One enrolled visitor."""

    person_id: str
    display_name: str
    face_embedding: np.ndarray
    voice_phrase: str
    created_ts: float


class EnrollmentStore:
    """Filesystem-backed store: ``<root>/people/<id>/{face.npy,voice.txt,meta.json}``."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root = Path(root_dir)
        self.people_dir = self.root / "people"
        self.people_dir.mkdir(parents=True, exist_ok=True)

    def _person_dir(self, person_id: str) -> Path:
        return self.people_dir / person_id

    def list_people(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in sorted(self.people_dir.iterdir()):
            if not p.is_dir():
                continue
            meta_path = p / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["person_id"] = p.name
                if "voice_phrase_preview" not in meta:
                    vt = p / "voice.txt"
                    if vt.exists():
                        text = vt.read_text(encoding="utf-8").strip()
                        meta["voice_phrase_preview"] = (text[:24] + "…") if len(text) > 24 else text
                out.append(meta)
            except Exception as e:
                logger.warning("Skipping %s: %s", p, e)
        return out

    def get(self, person_id: str) -> EnrolledPerson | None:
        d = self._person_dir(person_id)
        face_p = d / "face.npy"
        voice_p = d / "voice.txt"
        meta_p = d / "meta.json"
        if not (face_p.exists() and voice_p.exists() and meta_p.exists()):
            return None
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            phrase = voice_p.read_text(encoding="utf-8").strip()
            if not phrase:
                return None
            return EnrolledPerson(
                person_id=person_id,
                display_name=str(meta.get("display_name", person_id)),
                face_embedding=np.load(face_p),
                voice_phrase=phrase,
                created_ts=float(meta.get("created_ts", 0)),
            )
        except Exception as e:
            logger.error("Failed to load person %s: %s", person_id, e)
            return None

    def iter_enrolled(self) -> Iterator[EnrolledPerson]:
        for row in self.list_people():
            pid = row.get("person_id")
            if isinstance(pid, str):
                person = self.get(pid)
                if person is not None:
                    yield person

    def upsert(
        self,
        display_name: str,
        face_embedding: np.ndarray,
        voice_phrase: str,
        person_id: str | None = None,
    ) -> str:
        pid = person_id or _slug(display_name)
        base = pid
        n = 0
        while (self._person_dir(pid)).exists() and person_id is None:
            n += 1
            pid = f"{base}-{n}"
        d = self._person_dir(pid)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "face.npy", face_embedding.astype(np.float32))
        (d / "voice.txt").write_text(voice_phrase.strip(), encoding="utf-8")
        meta = {
            "display_name": display_name.strip(),
            "created_ts": time.time(),
            "voice_phrase_preview": (voice_phrase.strip()[:24] + "…")
            if len(voice_phrase.strip()) > 24
            else voice_phrase.strip(),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # Remove legacy embedding file if present
        legacy = d / "voice.npy"
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:
                pass
        logger.info("Enrolled person %s (%s)", pid, display_name)
        return pid

    def delete(self, person_id: str) -> bool:
        d = self._person_dir(person_id)
        if not d.exists():
            return False
        for f in d.iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        try:
            d.rmdir()
        except OSError:
            pass
        return True

    def best_match_dual(
        self,
        face_emb: np.ndarray | None,
        live_phrase: str | None,
        face_threshold: float,
        phrase_threshold: float,
    ) -> tuple[str | None, float, dict[str, Any]]:
        """Match face cosine + Whisper phrase similarity to the same enrolled person."""
        if face_emb is None or not live_phrase:
            return None, 0.0, {"reason": "missing_face_or_phrase"}

        best_id: str | None = None
        best_min = -1.0
        debug: dict[str, Any] = {}

        for person in self.iter_enrolled():
            cf = _cosine(face_emb, person.face_embedding)
            cv = phrase_similarity(live_phrase, person.voice_phrase)
            debug[f"{person.person_id}_face"] = cf
            debug[f"{person.person_id}_phrase"] = cv
            if cf >= face_threshold and cv >= phrase_threshold:
                joint = min(cf, cv)
                if joint > best_min:
                    best_min = joint
                    best_id = person.person_id

        if best_id is None:
            return None, 0.0, debug
        return best_id, best_min, debug
