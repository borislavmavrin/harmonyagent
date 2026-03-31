import json

import pytest

from harmonyagent.agents.exceptions.agent_exceptions import ExecutionTimeoutError, ToolCallArgParsingError
from harmonyagent.agents.exceptions.nonterminating_exceptions import UnknownToolCallArg, UnknownToolCalled
from harmonyagent.agents.tools.container_exec import ContainerExec
from harmonyagent.agents.tools.repo_browser import ApplyPatch
from harmonyagent.agents.tools.utils import pick_tool
from harmonyagent.environments.local import LocalEnvironment, LocalEnvironmentConfig

# --- pick_tool ---


class FakeTool:
    def __init__(self, name):
        self.name = name
        self.definition = ""
        self.args_names = []

    def use(self, tool_args):
        return {"output": "ok", "returncode": 0}


def test_pick_tool_exact_match():
    registry = {"container.exec": FakeTool("container.exec")}
    tool = pick_tool("container.exec", registry)
    assert tool.name == "container.exec"


def test_pick_tool_prefix_match():
    registry = {"container.exec": FakeTool("container.exec")}
    tool = pick_tool("container.exec<|channel|>commentary", registry)
    assert tool.name == "container.exec"


def test_pick_tool_unknown_raises():
    registry = {"container.exec": FakeTool("container.exec")}
    with pytest.raises(UnknownToolCalled, match="tool called: unknown, available: \\['container.exec'\\]"):
        pick_tool("unknown", registry)


def test_pick_tool_empty_registry_raises():
    with pytest.raises(UnknownToolCalled, match="tool called: container.exec, available: \\[\\]"):
        pick_tool("container.exec", {})


# --- BashTool ---


@pytest.fixture
def local_env(tmp_path):
    return LocalEnvironment(config=LocalEnvironmentConfig(cwd=str(tmp_path), timeout=10))


@pytest.fixture
def bash_tool(local_env):
    return ContainerExec(local_env)


class TestExecTool:
    def test_init(self, bash_tool):
        assert bash_tool.name == "container.exec"
        assert bash_tool.args_names == ["command", "cmd", "commands"]

    def test_use_command_key(self, bash_tool):
        result = bash_tool.use(json.dumps({"command": ["bash", "-lc", "echo hello"]}))
        assert result["output"].strip() == "hello"
        assert result["returncode"] == 0

    def test_use_cmd_key(self, bash_tool):
        result = bash_tool.use(json.dumps({"cmd": ["echo", "cmd_variant"]}))
        assert "cmd_variant" in result["output"]

    def test_use_commands_key(self, bash_tool):
        result = bash_tool.use(json.dumps({"commands": ["echo", "commands_variant"]}))
        assert "commands_variant" in result["output"]

    def test_use_strips_bash_prefix(self, bash_tool):
        result = bash_tool.use(json.dumps({"command": ["bash", "-lc", "echo stripped"]}))
        assert "stripped" in result["output"]

    def test_use_no_bash_prefix(self, bash_tool):
        result = bash_tool.use(json.dumps({"command": ["ls"]}))
        assert result["returncode"] == 0

    def test_use_unknown_arg_raises(self, bash_tool):
        with pytest.raises(
            UnknownToolCallArg,
            match="unknown tool args for shell got tool_args: \\['foo'\\], expected: \\['command', 'cmd', 'commands'\\]",
        ):
            bash_tool.use(json.dumps({"foo": ["bar"]}))

    def test_use_invalid_json_raises(self, bash_tool):
        with pytest.raises(ToolCallArgParsingError, match="failed to parse args for shell, args: 'not valid json{{{'"):
            bash_tool.use("not valid json{{{")

    def test_use_timeout_raises(self, tmp_path):
        env = LocalEnvironment(config=LocalEnvironmentConfig(cwd=str(tmp_path), timeout=1))
        tool = ContainerExec(env)
        with pytest.raises(ExecutionTimeoutError) as exc_info:
            tool.use(json.dumps({"command": ["bash", "-lc", "sleep 10"]}))
        assert isinstance(exc_info.value.output, str)

    def test_use_timeout_propagates_cmd_and_timeout(self, tmp_path):
        env = LocalEnvironment(config=LocalEnvironmentConfig(cwd=str(tmp_path), timeout=1))
        tool = ContainerExec(env)
        with pytest.raises(ExecutionTimeoutError) as exc_info:
            tool.use(json.dumps({"command": ["bash", "-lc", "sleep 10"]}))
        assert exc_info.value.timeout == 1
        assert isinstance(exc_info.value.cmd, str)
        assert len(exc_info.value.cmd) > 0


# --- ApplyPatch ---


@pytest.fixture
def apply_patch_tool(local_env):
    return ApplyPatch(local_env)


class TestApplyPatch:
    def test_init(self, apply_patch_tool):
        assert apply_patch_tool.name == "apply_patch"
        assert apply_patch_tool.args_names == ["patch"]

    def test_use_patch_key(self, apply_patch_tool):
        result = apply_patch_tool.use(json.dumps({"patch": "*** Begin Patch\n*** End Patch"}))
        # apply_patch command likely errors (not installed), but it runs and returns
        assert "output" in result
        assert "returncode" in result

    def test_use_unknown_arg_raises(self, apply_patch_tool):
        with pytest.raises(
            UnknownToolCallArg,
            match="unknown tool args for apply_patch got tool_args: \\['command'\\], expected: \\['patch'\\]",
        ):
            apply_patch_tool.use(json.dumps({"command": ["echo"]}))

    def test_use_invalid_json_raises(self, apply_patch_tool):
        with pytest.raises(
            ToolCallArgParsingError, match="failed to parse args for apply_patch, args: 'not valid json{{{'"
        ):
            apply_patch_tool.use("not valid json{{{")

    def test_use_timeout_raises(self, tmp_path, monkeypatch):
        import subprocess

        env = LocalEnvironment(config=LocalEnvironmentConfig(cwd=str(tmp_path), timeout=1))
        tool = ApplyPatch(env)

        original_run = subprocess.run

        def slow_run(*args, **kwargs):
            kwargs["timeout"] = 0.01
            return original_run("sleep 10", **kwargs)

        monkeypatch.setattr(subprocess, "run", slow_run)
        with pytest.raises(ExecutionTimeoutError) as exc_info:
            tool.use(json.dumps({"patch": "test"}))
        assert isinstance(exc_info.value.output, str)

    def test_use_timeout_propagates_cmd_and_timeout(self, tmp_path, monkeypatch):
        import subprocess

        env = LocalEnvironment(config=LocalEnvironmentConfig(cwd=str(tmp_path), timeout=1))
        tool = ApplyPatch(env)

        original_run = subprocess.run

        def slow_run(*args, **kwargs):
            kwargs["timeout"] = 0.01
            return original_run("sleep 10", **kwargs)

        monkeypatch.setattr(subprocess, "run", slow_run)
        with pytest.raises(ExecutionTimeoutError) as exc_info:
            tool.use(json.dumps({"patch": "test"}))
        assert exc_info.value.timeout == 0.01
        assert isinstance(exc_info.value.cmd, str)
        assert len(exc_info.value.cmd) > 0
