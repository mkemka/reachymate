"""FastAPI routes for the Reachy Receptionist dashboard.

Mounted into the Reachy Mini Apps settings server so the same URL that serves
the settings page also serves the receptionist live view, enrollment, and
member management endpoints.

All routes are prefixed with ``/receptionist/``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

_STATIC_DIR = Path(__file__).parent / "static"

if TYPE_CHECKING:
    from reachy_mini_conversation_app.receptionist.gate import ReceptionistGate

logger = logging.getLogger(__name__)


def mount_receptionist_dashboard(
    app: FastAPI,
    get_gate: Callable[[], Optional["ReceptionistGate"]],
    get_loop: Callable[[], Optional[asyncio.AbstractEventLoop]],
) -> None:
    """Register all ``/receptionist/*`` routes on *app*.

    Parameters
    ----------
    app:       FastAPI instance (the settings server).
    get_gate:  Callable that returns the current ``ReceptionistGate`` (or None).
    get_loop:  Callable that returns the running asyncio event loop (or None).
    """

    try:
        from pydantic import BaseModel
    except ImportError:
        logger.warning("Pydantic not available — receptionist dashboard routes not mounted")
        return

    _NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}

    # ------------------------------------------------------------------
    # HTML page
    # ------------------------------------------------------------------

    @app.get("/receptionist")
    @app.get("/receptionist/")
    def _receptionist_page() -> FileResponse:
        """Serve the receptionist dashboard HTML page."""
        return FileResponse(str(_STATIC_DIR / "receptionist.html"), headers=_NO_CACHE)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _gate() -> "ReceptionistGate | None":
        return get_gate()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @app.get("/receptionist/status")
    def _status() -> JSONResponse:
        """Current recognition state for the live dashboard panel."""
        g = _gate()
        if g is None:
            return JSONResponse({"state": "DISABLED", "message": "Receptionist mode not active", "led_state": "off"})
        return JSONResponse(g.get_current_status())

    # ------------------------------------------------------------------
    # People / enrollment
    # ------------------------------------------------------------------

    @app.get("/receptionist/people")
    def _people() -> JSONResponse:
        """All enrolled members with their Roo point balances."""
        g = _gate()
        if g is None:
            return JSONResponse({"ok": False, "people": [], "error": "receptionist_disabled"})
        people: list[dict[str, Any]] = []
        for row in g.store.list_people():
            pid = row.get("person_id")
            points: int | None = None
            if g.ledger is not None and pid:
                points = g.ledger.get_balance(pid)
            people.append({**row, "roo_points": points})
        return JSONResponse({"ok": True, "people": people, "count": len(people)})

    class EnrollPayload(BaseModel):
        display_name: str

    @app.post("/receptionist/enroll")
    async def _enroll(payload: EnrollPayload) -> JSONResponse:
        """Enroll a new person using the current camera frame + recent audio."""
        g = _gate()
        if g is None:
            return JSONResponse({"ok": False, "error": "receptionist_disabled"}, status_code=503)
        name = (payload.display_name or "").strip()
        if not name:
            return JSONResponse({"ok": False, "error": "display_name required"}, status_code=400)
        loop = get_loop()
        if loop is None:
            result = await asyncio.get_event_loop().run_in_executor(None, g.enroll_from_buffers, name, None)
        else:
            result = await asyncio.get_event_loop().run_in_executor(None, g.enroll_from_buffers, name, None)
        return JSONResponse(result)

    @app.delete("/receptionist/people/{person_id}")
    def _delete_person(person_id: str) -> JSONResponse:
        """Remove an enrolled person (does not delete ledger history)."""
        g = _gate()
        if g is None:
            return JSONResponse({"ok": False, "error": "receptionist_disabled"}, status_code=503)
        ok = g.store.delete(person_id)
        if not ok:
            return JSONResponse({"ok": False, "error": "person_not_found"}, status_code=404)
        return JSONResponse({"ok": True, "deleted": person_id})

    # ------------------------------------------------------------------
    # Ledger management
    # ------------------------------------------------------------------

    class AddPointsPayload(BaseModel):
        points: int

    @app.post("/receptionist/people/{person_id}/add_points")
    def _add_points(person_id: str, payload: AddPointsPayload) -> JSONResponse:
        """Admin: add Roo points to a member's balance."""
        g = _gate()
        if g is None or g.ledger is None:
            return JSONResponse({"ok": False, "error": "ledger_not_available"}, status_code=503)
        pts = int(payload.points)
        if pts <= 0:
            return JSONResponse({"ok": False, "error": "points must be a positive integer"}, status_code=400)
        ok = g.ledger.add_points(person_id, pts)
        if not ok:
            return JSONResponse({"ok": False, "error": "member_not_found"}, status_code=404)
        balance = g.ledger.get_balance(person_id)
        return JSONResponse({"ok": True, "person_id": person_id, "new_balance": balance})

    @app.get("/receptionist/people/{person_id}/transactions")
    def _transactions(person_id: str, limit: int = 20) -> JSONResponse:
        """Recent ledger transactions for a member."""
        g = _gate()
        if g is None or g.ledger is None:
            return JSONResponse({"ok": False, "error": "ledger_not_available"}, status_code=503)
        txs = g.ledger.get_transactions(person_id, limit=limit)
        return JSONResponse({"ok": True, "transactions": txs})

    logger.info("Receptionist dashboard routes mounted (/receptionist/*)")
