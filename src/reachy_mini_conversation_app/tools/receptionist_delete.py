import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class ReceptionistDelete(Tool):
    """Remove an enrolled person by person_id."""

    name = "receptionist_delete"
    description = "Delete a stored face+voice enrollment by person_id (from receptionist_list)."
    parameters_schema = {
        "type": "object",
        "properties": {
            "person_id": {"type": "string", "description": "Directory id under receptionist_data/people/"},
        },
        "required": ["person_id"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        if deps.receptionist_gate is None:
            return {"error": "receptionist mode is not enabled"}
        pid = (kwargs.get("person_id") or "").strip()
        if not pid:
            return {"error": "person_id required"}
        ok = deps.receptionist_gate.store.delete(pid)
        logger.info("receptionist_delete %s -> %s", pid, ok)
        return {"ok": ok, "person_id": pid}
