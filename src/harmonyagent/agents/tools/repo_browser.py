# Discovered in-distribution tools for gpt-oss model.
# Each tool is defined based on schema sampling results.
# Import these into sampler.py with --tools wip.log_analyzer.discovered_tools

import json
import subprocess

from openai_harmony import ToolDescription, ToolNamespaceConfig

from harmonyagent.agents.exceptions.agent_exceptions import (
    ExecutionTimeoutError,
    ToolCallArgParsingError,
    UnknownToolCallArg,
)
from harmonyagent.agents.tools.utils import _resolve_path
from harmonyagent.domain_model import Environment

print_tree = ToolDescription.new(
    "print_tree",
    "Prints a tree view of the repository file structure.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Root path to start the tree from. Use empty string for repo root.",
            },
            "depth": {
                "type": "number",
                "description": "Maximum directory depth to display.",
            },
        },
        "required": ["path", "depth"],
        "additionalProperties": False,
    },
)

search = ToolDescription.new(
    "search",
    "Searches repository file contents for a query string and returns matching file paths.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to search in. Use empty string for repo root.",
            },
            "query": {
                "type": "string",
                "description": "Search query or pattern to look for.",
            },
            "max_results": {
                "type": "number",
                "description": "Maximum number of results to return.",
            },
        },
        "required": ["path", "query"],
        "additionalProperties": False,
    },
)

open_file = ToolDescription.new(
    "open_file",
    "Opens a file and returns its contents between the specified line range.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to open. Full absolute path or relative to the current working directory.",
            },
            "line_start": {
                "type": "number",
                "description": "First line number to display (1-indexed).",
            },
            "line_end": {
                "type": "number",
                "description": "Last line number to display (1-indexed).",
            },
            "line_numbers": {
                "type": "boolean",
                "description": "If true, prefix each output line with its line number in the file.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)

list_dir = ToolDescription.new(
    "list_dir",
    "Lists entries in a directory with optional depth control.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list. Use empty string for repo root.",
            },
            "depth": {
                "type": "number",
                "description": "Maximum directory depth to traverse.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)

apply_patch = ToolDescription.new(
    "apply_patch",
    "Patch a file",
    parameters={
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "Formatted patch code",
                "default": "*** Begin Patch\n*** End Patch\n",
            }
        },
    },
)


class PrintTree:
    """repo_browser.print_tree — {path: str, depth: int}

    Runs `find` to produce a sorted file tree. Output is plain text,
    matching the model's dominant expectation (72% plain text in sampling).
    """

    def __init__(self, env: Environment):
        self.env = env
        self.name = "print_tree"
        self.definition = ""
        self.args_names = ["path", "depth"]

    def use(self, tool_args: str) -> dict:
        try:
            args = json.loads(tool_args)
        except json.JSONDecodeError:
            raise ToolCallArgParsingError("print_tree", tool_args)

        path = _resolve_path(args.get("path", ""), self.env.config.cwd)
        depth = int(args.get("depth", 3))
        cmd = f"tree -L {depth} {json.dumps(path)}"
        try:
            return self.env.execute(cmd)
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(output=output, cmd=str(e.cmd), timeout=e.timeout)


class Search:
    """repo_browser.search — {path: str, query: str, max_results?: int}

    Runs ripgrep (grep-style file:line:content output). 91% of sampled
    responses expected JSON, but rg output is understood in both modes.
    """

    def __init__(self, env: Environment):
        self.env = env
        self.name = "search"
        self.definition = ""
        self.args_names = ["path", "query"]

    def use(self, tool_args: str) -> dict:
        try:
            args = json.loads(tool_args)
        except json.JSONDecodeError:
            raise ToolCallArgParsingError("search", tool_args)

        if "query" not in args:
            raise UnknownToolCallArg("search", list(args.keys()), self.args_names)

        path = _resolve_path(args.get("path", ""), self.env.config.cwd)
        max_results = int(args.get("max_results", 20))
        cmd = f"rg --no-heading -n {json.dumps(args['query'])} {json.dumps(path)} | head -n {max_results}"
        try:
            return self.env.execute(cmd)
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(output=output, cmd=str(e.cmd), timeout=e.timeout)


class OpenFile:
    """repo_browser.open_file — {path: str, line_start?: int, line_end?: int}

    Reads a file, optionally sliced to a line range. Accepts "file_path"
    and "start_line"/"end_line" as aliases (observed in ~25% / 8% of calls).
    Output is raw file content (plain text), matching the 85% JSON expectation
    via cat/sed — the model reads `content` from whatever text is returned.
    """

    def __init__(self, env: Environment):
        self.env = env
        self.name = "open_file"
        self.definition = ""
        self.args_names = ["path", "file_path"]

    def use(self, tool_args: str) -> dict:
        try:
            args = json.loads(tool_args)
        except json.JSONDecodeError:
            raise ToolCallArgParsingError("open_file", tool_args)

        # Accept path or file_path alias
        if "path" in args:
            file_path = _resolve_path(args["path"].strip(), self.env.config.cwd)
        elif "file_path" in args:
            file_path = _resolve_path(args["file_path"].strip(), self.env.config.cwd)
        else:
            raise UnknownToolCallArg("open_file", list(args.keys()), self.args_names)

        # Accept line_start/line_end or start_line/end_line aliases
        line_start = args.get("line_start") or args.get("start_line")
        line_end = args.get("line_end") or args.get("end_line")
        line_numbers = args.get("line_numbers", False)

        start = int(line_start) if line_start else 1
        end = int(line_end) if line_end else "$"
        cmd = f"sed -n '{start},{end}p' {json.dumps(file_path)}"
        if line_numbers:
            cmd += f" | nl -ba -v {start}"
        result = self.env.execute(cmd)
        if result["returncode"] != 0:
            result["output"] = result["output"].removeprefix("sed:")
        try:
            return result
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(output=output, cmd=str(e.cmd), timeout=e.timeout)


class RepoBrowserListDir:
    """repo_browser.list_dir — {path: str, depth?: int}

    Like PrintTree but depth is optional (default 1 = flat listing).
    Named RepoBrowserListDir to avoid collision with tools.py:ListDir.
    """

    def __init__(self, env: Environment):
        self.env = env
        self.name = "list_dir"
        self.definition = ""
        self.args_names = ["path"]

    def use(self, tool_args: str) -> dict:
        try:
            args = json.loads(tool_args)
        except json.JSONDecodeError:
            raise ToolCallArgParsingError("list_dir", tool_args)

        path = _resolve_path(args.get("path", ""), self.env.config.cwd)
        depth = int(args.get("depth", 1))
        cmd = f"find {json.dumps(path)} -maxdepth {depth} | sort"
        try:
            return self.env.execute(cmd)
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(output=output, cmd=str(e.cmd), timeout=e.timeout)


class ApplyPatch:
    def __init__(self, env: Environment):
        self.env = env
        self.name = "apply_patch"
        self.definition = ""
        self.args_names = ["patch"]

    def use(self, tool_args: str) -> dict:
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            raise ToolCallArgParsingError("apply_patch", tool_args)

        if "patch" in tool_args:
            apply_patch_command = f"apply_patch <<'PATCH'\n{tool_args['patch']}\nPATCH"
        else:
            raise UnknownToolCallArg("apply_patch", list(tool_args.keys()), self.args_names)
        try:
            return self.env.execute(apply_patch_command)
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(output=output, cmd=str(e.cmd), timeout=e.timeout)


# Namespace config for bundling repo_browser tools
REPO_BROWSER_NS = ToolNamespaceConfig(
    name="repo_browser",
    description="Repository browsing tools for exploring code structure.",
    tools=[print_tree, search, open_file, list_dir, apply_patch],
)
