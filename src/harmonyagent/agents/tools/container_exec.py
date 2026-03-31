import json
import subprocess

from openai_harmony import ToolDescription, ToolNamespaceConfig

from harmonyagent.agents.exceptions.agent_exceptions import (
    ExecutionTimeoutError,
    ToolCallArgParsingError,
    UnknownToolCallArg,
)
from harmonyagent.domain_model import Environment

container_exec = ToolDescription.new(
    "exec",
    'Runs a shell command and returns its output.\n- The arguments to `shell` will be passed to execvp(). Most terminal commands should be prefixed with ["bash", "-lc"].\n- Always set the `workdir` param when using the shell function. Do not use `cd` unless absolutely necessary.',
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The command to execute",
            },
        },
        "required": ["command"],
    },
)


class ContainerExec:
    def __init__(self, env: Environment):
        self.env = env
        self.name = "container.exec"
        self.definition = ""
        self.args_names = ["command", "cmd", "commands"]

    def use(self, tool_args: str) -> dict:
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            raise ToolCallArgParsingError("shell", tool_args)

        if "command" in tool_args:
            bash_command = " ".join(tool_args["command"])
        elif "cmd" in tool_args:
            bash_command = " ".join(tool_args["cmd"])
        elif "commands" in tool_args:
            bash_command = " ".join(tool_args["commands"])
        else:
            raise UnknownToolCallArg("shell", list(tool_args.keys()), self.args_names)
        bash_command = bash_command.removeprefix("bash -lc ")
        try:
            return self.env.execute(bash_command)
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(output=output, cmd=str(e.cmd), timeout=e.timeout)


CONTAINER_NS = ToolNamespaceConfig(
    name="container",
    description="Container tools.",
    tools=[container_exec],
)
