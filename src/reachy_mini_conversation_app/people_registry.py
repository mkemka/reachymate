"""On-disk registry for optional voice and face samples per person.

Data lives under ``<instance_path>/people_registry/``:

- ``index.json`` — list of person records (metadata only).
- ``<person_id>/`` — ``face_*.jpg``, ``voice_*.<ext>``, and optional ``notes.txt``.

This is a simple file-based store for enrollment workflows, not a full biometric
database. Downstream systems can sync or index these files as needed.
"""

from __future__ import annotations
import re
import json
import uuid
from typing import Any
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import field, dataclass


def _safe_segment(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip())[:80]
    return s or "person"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class PersonRecord:
    """One enrolled person."""

    id: str
    display_name: str
    created_at: str
    faces: list[str] = field(default_factory=list)
    voices: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for API and index storage."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "faces": list(self.faces),
            "voices": list(self.voices),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PersonRecord:
        """Build a record from index or API data."""
        return cls(
            id=str(d["id"]),
            display_name=str(d.get("display_name", "")),
            created_at=str(d.get("created_at", "")),
            faces=[str(x) for x in d.get("faces", [])],
            voices=[str(x) for x in d.get("voices", [])],
        )


class PeopleRegistry:
    """File-backed people registry rooted at ``base_dir``."""

    def __init__(self, base_dir: Path) -> None:
        """Use ``base_dir`` as the registry root (created if missing)."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        """Path to the main ``index.json`` file."""
        return self.base_dir / "index.json"

    def _load_rows(self) -> list[dict[str, Any]]:
        if not self.index_path.is_file():
            return []
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        return []

    def _save_rows(self, rows: list[dict[str, Any]]) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.index_path)

    def _row_by_id(self, person_id: str) -> dict[str, Any] | None:
        for r in self._load_rows():
            if str(r.get("id")) == person_id:
                return r
        return None

    def _upsert_row(self, record: PersonRecord) -> None:
        rows = self._load_rows()
        d = record.to_dict()
        found = False
        for i, r in enumerate(rows):
            if str(r.get("id")) == record.id:
                rows[i] = d
                found = True
                break
        if not found:
            rows.append(d)
        self._save_rows(rows)

    def list_people(self) -> list[dict[str, Any]]:
        """Return all person rows (metadata) from the index."""
        return list(self._load_rows())

    def get_person(self, person_id: str) -> PersonRecord | None:
        """Return the person with ``person_id``, or ``None``."""
        r = self._row_by_id(person_id)
        return PersonRecord.from_dict(r) if r else None

    def create_person(self, display_name: str) -> PersonRecord:
        """Create a new person directory and index entry."""
        pid = str(uuid.uuid4())
        person_dir = self.base_dir / pid
        person_dir.mkdir(parents=True, exist_ok=False)
        slug = _safe_segment(display_name)
        rec = PersonRecord(
            id=pid,
            display_name=(display_name or "").strip() or slug,
            created_at=_utc_now_iso(),
        )
        self._upsert_row(rec)
        (person_dir / "README.txt").write_text(
            "Face images: face_*.jpg\nVoice clips: voice_*.{webm,wav,mp3,...}\n",
            encoding="utf-8",
        )
        return rec

    def add_face(self, person_id: str, jpeg_bytes: bytes) -> str:
        """Store a JPEG face image; returns the new filename (e.g. ``face_001.jpg``)."""
        rec = self.get_person(person_id)
        if rec is None:
            raise KeyError("unknown_person")
        person_dir = self.base_dir / person_id
        if not person_dir.is_dir():
            raise KeyError("unknown_person")
        n = len(rec.faces) + 1
        fname = f"face_{n:03d}.jpg"
        path = person_dir / fname
        path.write_bytes(jpeg_bytes)
        rec.faces.append(fname)
        self._upsert_row(rec)
        return fname

    def add_voice(self, person_id: str, data: bytes, ext: str) -> str:
        """Store a voice clip; returns the new filename."""
        rec = self.get_person(person_id)
        if rec is None:
            raise KeyError("unknown_person")
        person_dir = self.base_dir / person_id
        if not person_dir.is_dir():
            raise KeyError("unknown_person")
        ext = re.sub(r"[^a-zA-Z0-9.]+", "", ext or "bin")[:10] or "bin"
        if ext.startswith("."):
            ext = ext[1:]
        n = len(rec.voices) + 1
        fname = f"voice_{n:03d}.{ext}"
        path = person_dir / fname
        path.write_bytes(data)
        rec.voices.append(fname)
        self._upsert_row(rec)
        return fname

    def file_path(self, person_id: str, filename: str) -> Path | None:
        """Resolve a stored file path if it belongs to the person, else ``None``."""
        if not re.match(r"^[a-f0-9-]{36}$", person_id):
            return None
        rec = self.get_person(person_id)
        if rec is None:
            return None
        if filename not in rec.faces and filename not in rec.voices:
            return None
        p = (self.base_dir / person_id / filename).resolve()
        root = (self.base_dir / person_id).resolve()
        if not str(p).startswith(str(root)) or not p.is_file():
            return None
        return p


def get_registry_dir(instance_path: str | None) -> Path | None:
    """Return ``<instance_path>/people_registry`` or ``None`` if no instance path."""
    if not instance_path:
        return None
    return Path(instance_path) / "people_registry"
