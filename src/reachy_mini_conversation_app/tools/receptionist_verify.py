import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class ReceptionistVerify(Tool):
    """Match InsightFace embedding + Whisper phrase against enrolled profiles and unlock a session."""

    name = "receptionist_verify"
    description = (
        "Verify identity: caller faces the camera and speaks their enrolled passphrase. "
        "If YOLO/InsightFace face and Whisper phrase both match the same person, unlocks the session; "
        "they must then say a wake phrase before the assistant speaks aloud."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        if deps.receptionist_gate is None:
            return {"error": "receptionist mode is not enabled"}
        result = await asyncio.to_thread(deps.receptionist_gate.verify_from_buffers)
        logger.info("receptionist_verify: %s", result)
        return result
