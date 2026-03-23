"""HTTP routes for the on-disk people registry (face + voice enrollment)."""

from __future__ import annotations
import base64
import logging
from typing import Callable, Optional

from fastapi import FastAPI

from reachy_mini_conversation_app.people_registry import PeopleRegistry, get_registry_dir


logger = logging.getLogger(__name__)


def mount_people_registry_routes(
    app: FastAPI,
    get_instance_path: Callable[[], Optional[str]],
) -> None:
    """Register people registry endpoints."""
    try:
        from fastapi import Request, Response
        from pydantic import BaseModel
        from fastapi.responses import FileResponse, JSONResponse
    except Exception:  # pragma: no cover
        return

    class CreatePayload(BaseModel):
        display_name: str

    def _registry() -> PeopleRegistry | None:
        inst = get_instance_path()
        d = get_registry_dir(inst)
        if d is None:
            return None
        return PeopleRegistry(d)

    @app.get("/people/registry")
    def _list() -> dict:  # type: ignore
        reg = _registry()
        if reg is None:
            return JSONResponse({"ok": False, "error": "no_instance_path"}, status_code=503)  # type: ignore
        return {"ok": True, "people": reg.list_people()}

    @app.post("/people/registry/create")
    def _create(payload: CreatePayload) -> dict:  # type: ignore
        reg = _registry()
        if reg is None:
            return JSONResponse({"ok": False, "error": "no_instance_path"}, status_code=503)  # type: ignore
        name = (payload.display_name or "").strip()
        if not name:
            return JSONResponse({"ok": False, "error": "empty_name"}, status_code=400)  # type: ignore
        try:
            rec = reg.create_person(name)
            return {"ok": True, "person": rec.to_dict()}
        except Exception as e:
            logger.warning("people create failed: %s", e)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore

    @app.post("/people/registry/{person_id}/face")
    async def _face(person_id: str, request: Request) -> dict:  # type: ignore
        reg = _registry()
        if reg is None:
            return JSONResponse({"ok": False, "error": "no_instance_path"}, status_code=503)  # type: ignore
        ct = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        jpeg: bytes | None = None
        if ct == "application/json":
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            if isinstance(raw, dict) and raw.get("image_b64"):
                try:
                    jpeg = base64.b64decode(str(raw["image_b64"]), validate=False)
                except Exception:
                    return JSONResponse({"ok": False, "error": "bad_b64"}, status_code=400)  # type: ignore
        elif ct == "image/jpeg":
            jpeg = await request.body()
        if not jpeg:
            return JSONResponse({"ok": False, "error": "expected_json_b64_or_jpeg"}, status_code=400)  # type: ignore
        if len(jpeg) > 15 * 1024 * 1024:
            return JSONResponse({"ok": False, "error": "too_large"}, status_code=400)  # type: ignore
        try:
            fname = reg.add_face(person_id, jpeg)
            return {"ok": True, "file": fname, "person": reg.get_person(person_id).to_dict()}  # type: ignore
        except KeyError:
            return JSONResponse({"ok": False, "error": "unknown_person"}, status_code=404)  # type: ignore
        except Exception as e:
            logger.warning("face upload failed: %s", e)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore

    @app.post("/people/registry/{person_id}/voice")
    async def _voice(person_id: str, request: Request) -> dict:  # type: ignore
        reg = _registry()
        if reg is None:
            return JSONResponse({"ok": False, "error": "no_instance_path"}, status_code=503)  # type: ignore
        ct = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        ext = "webm"
        if "webm" in ct:
            ext = "webm"
        elif "wav" in ct:
            ext = "wav"
        elif "mpeg" in ct or "mp3" in ct:
            ext = "mp3"
        elif "ogg" in ct:
            ext = "ogg"
        data = await request.body()
        if not data or len(data) < 256:
            return JSONResponse({"ok": False, "error": "empty_or_too_small"}, status_code=400)  # type: ignore
        if len(data) > 30 * 1024 * 1024:
            return JSONResponse({"ok": False, "error": "too_large"}, status_code=400)  # type: ignore
        try:
            fname = reg.add_voice(person_id, data, ext)
            return {"ok": True, "file": fname, "person": reg.get_person(person_id).to_dict()}  # type: ignore
        except KeyError:
            return JSONResponse({"ok": False, "error": "unknown_person"}, status_code=404)  # type: ignore
        except Exception as e:
            logger.warning("voice upload failed: %s", e)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore

    @app.get("/people/registry/{person_id}/file/{filename}")
    def _file(person_id: str, filename: str) -> Response:  # type: ignore
        reg = _registry()
        if reg is None:
            return Response(status_code=503)
        p = reg.file_path(person_id, filename)
        if p is None:
            return Response(status_code=404)
        lower = filename.lower()
        if lower.endswith((".jpg", ".jpeg")):
            media = "image/jpeg"
        elif lower.endswith(".webm"):
            media = "audio/webm"
        elif lower.endswith(".wav"):
            media = "audio/wav"
        elif lower.endswith(".mp3"):
            media = "audio/mpeg"
        elif lower.endswith(".ogg"):
            media = "audio/ogg"
        else:
            media = "application/octet-stream"
        return FileResponse(str(p), media_type=media)
