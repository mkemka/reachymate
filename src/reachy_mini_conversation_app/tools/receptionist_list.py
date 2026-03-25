import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class ReceptionistList(Tool):
    """List enrolled people (ids and display names)."""

    name = "receptionist_list"
    description = "List visitors enrolled for receptionist face+voice verification."
    parameters_schema = {"type": "object", "properties": {}}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        if deps.receptionist_gate is None:
            return {"error": "receptionist mode is not enabled"}
        rows = deps.receptionist_gate.store.list_people()
        logger.info("receptionist_list: %d people", len(rows))
        return {"people": rows}
