import posixpath

from harmonyagent.agents.exceptions.agent_exceptions import UnknownToolCalled
from harmonyagent.domain_model import Tool


def _resolve_path(path: str, cwd: str) -> str:
    """Resolve a path against cwd: empty/relative paths become absolute under cwd."""
    if not path:
        return cwd
    if not posixpath.isabs(path):
        return posixpath.join(cwd, path)
    return path


def pick_tool(tool_name: str, tool_registry: dict[str, Tool]) -> Tool:
    for registry_tool_name in tool_registry:
        if tool_name.startswith(registry_tool_name):
            return tool_registry[registry_tool_name]
    raise UnknownToolCalled(tool_name, list(tool_registry.keys()))
