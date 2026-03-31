from typing import Any, Protocol

from openai_harmony import ToolDescription
from rich.console import Console

Console().print("👋 This is [bold green]mini-swe-agent[/bold green] version [bold green] [/bold green].\n")

# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any
    cost: float
    n_calls: int

    def query(self, prompt_text: str, stop_token_ids: list[int], max_tokens: int) -> dict: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Agent(Protocol):
    """Protocol for agents."""

    model: Model
    env: Environment
    messages: list[dict[str, str]]
    config: Any

    def run(self, task: str) -> tuple[str, str]: ...


class Tool(Protocol):
    "Tool for agents."

    name: str
    definition: Any
    args_names: list[str]
    desc: ToolDescription

    def use(self, tool_args: dict) -> Any: ...
