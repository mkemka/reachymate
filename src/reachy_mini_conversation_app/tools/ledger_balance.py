"""Tool: ledger_balance — query a member's point balance from the local ledger."""

from __future__ import annotations

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class LedgerBalance(Tool):
    """Query the local ledger for a member's current point balance and recent transactions."""

    name = "ledger_balance"
    description = (
        "Look up a member's current point balance (and optional recent transactions) from the "
        "local ledger. Use the person_id returned by receptionist_enroll or receptionist_verify. "
        "If person_id is omitted, lists all members with their balances."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "person_id": {
                "type": "string",
                "description": "Person ID from enrollment. Omit to list all members.",
            },
            "include_transactions": {
                "type": "boolean",
                "description": "If true, include the last 10 transactions for this member.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        gate = getattr(deps, "receptionist_gate", None)
        if gate is None:
            return {"error": "receptionist mode is not enabled"}

        ledger = getattr(gate, "ledger", None)
        if ledger is None:
            return {"error": "ledger is not initialised (check config and dependencies)"}

        person_id = (kwargs.get("person_id") or "").strip() or None
        include_tx = bool(kwargs.get("include_transactions", False))

        if person_id:
            member = ledger.get_member(person_id)
            if member is None:
                return {"ok": False, "error": "member_not_found", "person_id": person_id}
            result: Dict[str, Any] = {"ok": True, **member}
            if include_tx:
                result["transactions"] = ledger.get_transactions(person_id, limit=10)
            logger.info("ledger_balance: %s → %s pts", person_id, member.get("points"))
            return result
        else:
            members = ledger.list_members()
            logger.info("ledger_balance: listing %d members", len(members))
            return {"ok": True, "members": members, "count": len(members)}
