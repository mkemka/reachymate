import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class ReceptionistEnroll(Tool):
    """Link a person's face (YOLO crop + InsightFace) and spoken passphrase (Whisper) on disk."""

    name = "receptionist_enroll"
    description = (
        "Enroll a visitor: they face the camera and speak the same passphrase they will use at check-in "
        "(2–4 seconds). Stores an InsightFace embedding plus the Whisper-normalized phrase under person_id."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "display_name": {
                "type": "string",
                "description": "Human-readable name for this person",
            },
            "replace_person_id": {
                "type": "string",
                "description": "If set, overwrite this existing person_id instead of creating a new one",
            },
        },
        "required": ["display_name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        if deps.receptionist_gate is None:
            return {"error": "receptionist mode is not enabled"}
        name = (kwargs.get("display_name") or "").strip()
        if not name:
            return {"error": "display_name is required"}
        replace = kwargs.get("replace_person_id")
        replace_id = replace.strip() if isinstance(replace, str) and replace.strip() else None
        result = await asyncio.to_thread(deps.receptionist_gate.enroll_from_buffers, name, replace_id)
        logger.info("receptionist_enroll: %s", result)
        return result
