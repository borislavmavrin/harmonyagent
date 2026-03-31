import logging

import pytest
from openai_harmony import (
    HarmonyEncodingName,
    Message,
    RenderConversationConfig,
    Role,
    load_harmony_encoding,
)

from harmonyagent.agents.agent_config import AgentConfig
from harmonyagent.agents.exceptions.nonterminating_exceptions import (
    LongGeneration,
    NoToolCallNoFinalMessage,
    ToolCallAndFinalMessage,
    UnknownToolCalled,
)
from harmonyagent.agents.exceptions.terminating_exceptions import (
    LimitsExceeded,
    MaxNewTokensExceeded,
    UnexpectedFinishReason,
)
from harmonyagent.agents.harmony_agent import HarmonyAgent
from harmonyagent.agents.harmony_core.harmony_parsers import create_prompt_text
from harmonyagent.agents.harmony_core.sys_dev_message import get_developer_message, get_system_message
from harmonyagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from harmonyagent.models.test_models import TapeModel, TapeModelConfig


@pytest.fixture(autouse=True, scope="session")
def _replace_rich_handler():
    """Replace Rich logging handler with a plain StreamHandler to avoid Rich TypeError in tests."""
    agent_logger = logging.getLogger("harmonyagent")
    # Remove all existing handlers (including RichHandler)
    rich_handlers = [h for h in agent_logger.handlers if type(h).__name__ == "RichHandler"]
    for h in rich_handlers:
        agent_logger.removeHandler(h)
    # Add a plain handler
    plain = logging.StreamHandler()
    plain.setFormatter(logging.Formatter("%(name)s: %(levelname)s: %(message)s"))
    agent_logger.addHandler(plain)
    yield
    # Restore
    agent_logger.removeHandler(plain)
    for h in rich_handlers:
        agent_logger.addHandler(h)


# --- Harmony response text constants ---

SHELL_ECHO_HELLO = (
    "<|channel|>analysis<|message|>Let me check.<|end|>"
    "<|start|>assistant<|channel|>analysis to=container.exec code<|message|>"
    '{"command":["bash","-lc","echo hello"]}'
)

SHELL_CMD_KEY = (
    "<|channel|>analysis<|message|>Thinking.<|end|>"
    "<|start|>assistant<|channel|>analysis to=container.exec code<|message|>"
    '{"cmd":["echo","cmd_variant"]}'
)

SHELL_COMMANDS_KEY = (
    "<|channel|>analysis<|message|>Thinking.<|end|>"
    "<|start|>assistant<|channel|>analysis to=container.exec code<|message|>"
    '{"commands":["echo","commands_variant"]}'
)

SHELL_UNKNOWN_ARG = (
    "<|channel|>analysis<|message|>Thinking.<|end|>"
    "<|start|>assistant<|channel|>analysis to=container.exec code<|message|>"
    '{"foo":["echo","bar"]}'
)

SHELL_NO_BASH_PREFIX = (
    "<|channel|>analysis<|message|>Thinking.<|end|>"
    "<|start|>assistant<|channel|>analysis to=container.exec code<|message|>"
    '{"command":["ls","-la"]}'
)

APPLY_PATCH = (
    "<|channel|>analysis<|message|>Patching.<|end|>"
    "<|start|>assistant<|channel|>analysis to=functions.repo_browser.apply_patch code<|message|>"
    '{"patch":"*** Begin Patch\\n*** End Patch"}'
)

FINAL_MESSAGE = (
    "<|channel|>analysis<|message|>All done.<|end|><|start|>assistant<|channel|>final<|message|>Task completed."
)

TOOL_AND_FINAL = (
    '<|channel|>analysis to=container.exec code<|message|>{"command":["echo","hi"]}<|end|>'
    "<|start|>assistant<|channel|>final<|message|>Done"
)

ANALYSIS_ONLY = "<|channel|>analysis<|message|>Just thinking."

PYTHON_PRINT_42 = (
    "<|channel|>analysis<|message|>Let me compute.<|end|>"
    "<|start|>assistant<|channel|>analysis to=functions.python code<|message|>"
    "print(42)"
)


@pytest.fixture
def enc():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


@pytest.fixture
def local_env(tmp_path):
    return LocalEnvironment(config=LocalEnvironmentConfig(cwd=str(tmp_path), timeout=10))


@pytest.fixture
def git_env(tmp_path):
    import subprocess

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin",
        },
    )
    return LocalEnvironment(config=LocalEnvironmentConfig(cwd=str(tmp_path), timeout=10))


# --- Helpers ---


def compute_prompt_text(agent, enc):
    convo_config = RenderConversationConfig()
    convo_config.auto_drop_analysis = False
    prompt_text, _ = create_prompt_text(agent._messages, enc, convo_config, agent.config.max_context_window)
    return prompt_text


def make_tape_entry(prompt_text, response_text, finish_reason="stop"):
    return {
        "prompt_text": prompt_text,
        "response": {"choices": [{"text": response_text, "finish_reason": finish_reason}]},
    }


def setup_agent_for_query(env, config, tape, task="Fix the bug", task_id="test__test-123"):
    model = TapeModel(config=TapeModelConfig(tape=tape))
    agent = HarmonyAgent(model=model, env=env, task_id=task_id, config=config)
    agent._messages = []
    agent._messages.append(
        get_system_message(
            system_instructions=config.system_instructions if config.system_instructions else None,
            reasoning_effort=config.reasoning_effort,
            include_repo_browser_tools=True if [tool for tool in config.tools if "repo_browser." in tool] else False,
            include_container_tools=True if [tool for tool in config.tools if "container." in tool] else False,
        )
    )
    agent._messages.append(
        get_developer_message(config.developer_instructions if config.developer_instructions else None)
    )
    agent._messages.append(
        Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task=task))
    )
    return agent


# --- __init__ tests ---


class TestInit:
    def test_init_default_config(self, local_env):
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=local_env, task_id="test__test-1")
        assert isinstance(agent.config, AgentConfig)
        assert agent._messages == []
        assert agent.model is model
        assert agent.env is local_env
        assert agent.instance_id == "test__test-1"
        assert agent.agent_step == 0
        assert "container.exec" in agent.tool_registry
        assert "repo_browser.print_tree" in agent.tool_registry
        assert "repo_browser.search" in agent.tool_registry
        assert "repo_browser.open_file" in agent.tool_registry
        assert "repo_browser.list_dir" in agent.tool_registry
        assert "repo_browser.apply_patch" in agent.tool_registry

    def test_init_custom_config(self, local_env):
        config = AgentConfig(step_limit=50, cost_limit=1.0)
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=local_env, task_id="x", config=config)
        assert agent.config.step_limit == 50
        assert agent.config.cost_limit == 1.0

    def test_init_no_tools(self, local_env):
        config = AgentConfig(tools=[])
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=local_env, task_id="test__test-1", config=config)
        assert agent.tool_registry == {}

    def test_init_shell_only(self, local_env):
        config = AgentConfig(tools=["container.exec"])
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=local_env, task_id="test__test-1", config=config)
        assert "container.exec" in agent.tool_registry
        assert "apply_patch" not in agent.tool_registry
        assert len(agent.tool_registry) == 1

    def test_init_apply_patch_only(self, local_env):
        config = AgentConfig(tools=["repo_browser.apply_patch"])
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=local_env, task_id="test__test-1", config=config)
        assert "repo_browser.apply_patch" in agent.tool_registry
        assert "container.exec" not in agent.tool_registry
        assert len(agent.tool_registry) == 1

    def test_init_all_tools(self, local_env):
        config = AgentConfig(
            tools=[
                "container.exec",
                "repo_browser.print_tree",
                "repo_browser.search",
                "repo_browser.open_file",
                "repo_browser.list_dir",
                "repo_browser.apply_patch",
            ]
        )
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=local_env, task_id="test__test-1", config=config)
        assert "container.exec" in agent.tool_registry
        assert "repo_browser.print_tree" in agent.tool_registry
        assert "repo_browser.search" in agent.tool_registry
        assert "repo_browser.open_file" in agent.tool_registry
        assert "repo_browser.list_dir" in agent.tool_registry
        assert "repo_browser.apply_patch" in agent.tool_registry
        assert len(agent.tool_registry) == 6

    def test_init_unknown_tool_raises(self, local_env):
        config = AgentConfig(tools=["nonexistent_tool"])
        model = TapeModel(config=TapeModelConfig(tape=[]))
        with pytest.raises(KeyError):
            HarmonyAgent(model=model, env=local_env, task_id="test__test-1", config=config)


# --- render_template tests ---


class TestRenderTemplate:
    def test_render_template(self, local_env):
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=local_env, task_id="test__test-1")
        result = agent.render_template("Task: {{task}} Limit: {{step_limit}}", task="Fix bug #42")
        assert "Fix bug #42" in result
        assert str(agent.config.step_limit) in result


# --- messages property tests ---


class TestMessagesProperty:
    def test_messages_property(self, local_env):
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=local_env, task_id="test__test-1")
        agent._messages = [
            Message.from_role_and_content(Role.SYSTEM, "sys"),
            Message.from_role_and_content(Role.USER, "hi"),
        ]
        result = agent.messages
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(m, dict) for m in result)


# --- query() limits tests ---


class TestQueryLimits:
    def test_query_step_limit_exceeded(self, local_env, enc):
        config = AgentConfig(step_limit=1)
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        agent.model.n_calls = 1
        with pytest.raises(LimitsExceeded, match="limit step_limit exceeded: 1/1"):
            agent.query()

    def test_query_cost_limit_exceeded(self, local_env, enc):
        config = AgentConfig(cost_limit=0.5)
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        agent.model.cost = 0.5
        with pytest.raises(LimitsExceeded, match="limit cost_limit exceeded: 0.5/0.5"):
            agent.query()


# --- query() delay tests ---


class TestQueryDelay:
    def test_query_with_delay(self, local_env, enc):
        config = AgentConfig(delay=0.001)
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        assert len(received) > 0


# --- query() shell command variant tests ---


class TestQueryShellVariants:
    def test_query_shell_command_key(self, local_env, enc):
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        assert "hello" in obs[0].content[0].text

    def test_query_shell_cmd_key(self, local_env, enc):
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_CMD_KEY)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        assert "cmd_variant" in obs[0].content[0].text

    def test_query_shell_commands_key(self, local_env, enc):
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_COMMANDS_KEY)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        assert "commands_variant" in obs[0].content[0].text

    def test_query_shell_no_bash_prefix(self, local_env, enc):
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_NO_BASH_PREFIX)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        # "ls -la" has no "bash -lc " prefix, removeprefix is a no-op
        assert obs[0] is not None

    def test_query_shell_unknown_arg(self, local_env, enc):
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        # Two entries: first triggers UnknownToolCallArg, second succeeds (retry)
        tape = [
            make_tape_entry(prompt_text, SHELL_UNKNOWN_ARG),
            make_tape_entry(prompt_text, SHELL_ECHO_HELLO),
        ]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        assert "hello" in obs[0].content[0].text


# --- query() apply_patch tests ---


class TestQueryApplyPatch:
    def test_query_apply_patch(self, local_env, enc):
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, APPLY_PATCH)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        # apply_patch is formatted as: apply_patch <<'PATCH'\n...\nPATCH
        # It will run and produce some output (likely an error since apply_patch tool may not exist)
        assert obs[0] is not None


# --- query() final message tests ---


class TestQueryFinal:
    def test_query_final_submitted(self, git_env, enc):
        config = AgentConfig()
        agent = setup_agent_for_query(git_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, FINAL_MESSAGE)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        assert is_final is True

    def test_query_final_submitted_no_observation(self, git_env, enc):
        """Final messages return empty observation list (patch generation is in runner)."""
        config = AgentConfig()
        agent = setup_agent_for_query(git_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, FINAL_MESSAGE)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        assert is_final is True
        assert obs == []


# --- query() error retry tests ---


class TestQueryRetries:
    def test_query_tool_call_and_final_message(self, local_env, enc):
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        # First entry has both tool call and final → ToolCallAndFinalMessage (retry)
        # Second entry is valid
        tape = [
            make_tape_entry(prompt_text, TOOL_AND_FINAL),
            make_tape_entry(prompt_text, SHELL_ECHO_HELLO),
        ]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        assert "hello" in obs[0].content[0].text

    def test_query_no_tool_no_final(self, local_env, enc):
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        # First entry has only analysis (no tool, no final) → NoToolCallNoFinalMessage (retry)
        # Second entry is valid
        tape = [
            make_tape_entry(prompt_text, ANALYSIS_ONLY),
            make_tape_entry(prompt_text, SHELL_ECHO_HELLO),
        ]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        assert "hello" in obs[0].content[0].text

    def test_query_tool_call_and_final_message_message(self, local_env, enc, monkeypatch):
        """ToolCallAndFinalMessage includes tool recipient and final message text."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        from harmonyagent.agents.harmony_agent import HarmonyAgent

        fast_retry = retry(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        monkeypatch.setattr(HarmonyAgent, "query", fast_retry(HarmonyAgent.query.__wrapped__))

        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, TOOL_AND_FINAL)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with pytest.raises(
            ToolCallAndFinalMessage, match="received both tool call and final message, tool call recipient: "
        ):
            agent.query()

    def test_query_no_tool_no_final_message(self, local_env, enc, monkeypatch):
        """NoToolCallNoFinalMessage includes message count and channels."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        from harmonyagent.agents.harmony_agent import HarmonyAgent

        fast_retry = retry(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        monkeypatch.setattr(HarmonyAgent, "query", fast_retry(HarmonyAgent.query.__wrapped__))

        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, ANALYSIS_ONLY)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with pytest.raises(NoToolCallNoFinalMessage, match="no tool call and no final message are received, got: "):
            agent.query()

    def test_query_unknown_tool_raises(self, local_env, enc, monkeypatch):
        """Model calls a tool not in the registry → UnknownToolCalled at query level."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        from harmonyagent.agents.harmony_agent import HarmonyAgent

        # Disable retries so UnknownToolCalled propagates directly
        fast_retry = retry(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        monkeypatch.setattr(HarmonyAgent, "query", fast_retry(HarmonyAgent.query.__wrapped__))

        config = AgentConfig(tools=["repo_browser.apply_patch"])
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with pytest.raises(UnknownToolCalled, match="tool called: container.exec"):
            agent.query()

    def test_before_sleep_logs_with_prefix(self, local_env, enc, caplog):
        """Retry before_sleep callback logs _log_prefix (instance_id, step, calls)."""
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it", task_id="test__prefix-42")
        prompt_text = compute_prompt_text(agent, enc)
        # First response triggers NoToolCallNoFinalMessage (retry), second succeeds
        tape = [
            make_tape_entry(prompt_text, ANALYSIS_ONLY),
            make_tape_entry(prompt_text, SHELL_ECHO_HELLO),
        ]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.WARNING, logger="harmonyagent"):
            agent.query()
        assert any("test__prefix-42" in r.message and "retrying query" in r.message for r in caplog.records)


# --- query() execution timeout tests ---


class TestQueryExecutionTimeout:
    def test_query_execution_timeout(self, enc, tmp_path):
        env = LocalEnvironment(config=LocalEnvironmentConfig(cwd=str(tmp_path), timeout=1))
        config = AgentConfig()
        # Use a shell command that will timeout: sleep 10
        response_text = (
            "<|channel|>analysis<|message|>Running.<|end|>"
            "<|start|>assistant<|channel|>analysis to=container.exec code<|message|>"
            '{"command":["bash","-lc","sleep 10"]}'
        )
        agent = setup_agent_for_query(env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, response_text)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        received, obs, is_final = agent.query()
        # Observation should contain the timeout error string
        assert "timed out" in obs[0].content[0].text.lower() or "timeout" in obs[0].content[0].text.lower()


# --- query() high token warning tests ---


class TestQueryMaxTokensExceeded:
    def test_query_max_tokens_exceeded(self, local_env, enc):
        """When generated tokens >= max_new_tokens, raises MaxTokensExceeded."""
        config = AgentConfig(max_new_tokens=1)
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with pytest.raises(MaxNewTokensExceeded, match="max new tokens exceeded: 30/1"):
            agent.query()


# --- query() finish reason tests ---


class TestQueryFinishReason:
    def test_query_finish_reason_length_raises_long_generation(self, local_env, enc, monkeypatch):
        """finish_reason == 'length' raises LongGeneration (NonTerminatingException)."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        from harmonyagent.agents.harmony_agent import HarmonyAgent

        fast_retry = retry(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        original_query = HarmonyAgent.query.__wrapped__
        monkeypatch.setattr(HarmonyAgent, "query", fast_retry(original_query))

        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO, finish_reason="length")]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with pytest.raises(LongGeneration, match=r"finish_reason: length"):
            agent.query()

    def test_query_unexpected_finish_reason(self, local_env, enc):
        """finish_reason not in ('stop', 'length') raises UnexpectedFinishReason."""
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO, finish_reason="content_filter")]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with pytest.raises(UnexpectedFinishReason, match="content_filter"):
            agent.query()

    def test_run_unexpected_finish_reason(self, local_env, enc):
        """UnexpectedFinishReason is a TerminatingException, so run() returns it."""
        config = AgentConfig()
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=local_env, task_id="test__test-1", config=config)
        agent._messages = []
        agent._messages.append(
            get_system_message(
                system_instructions=config.system_instructions if config.system_instructions else None,
                reasoning_effort=config.reasoning_effort,
                include_repo_browser_tools=True
                if [tool for tool in config.tools if "repo_browser." in tool]
                else False,
                include_container_tools=True if [tool for tool in config.tools if "container." in tool] else False,
            )
        )
        agent._messages.append(
            get_developer_message(config.developer_instructions if config.developer_instructions else None)
        )
        agent._messages.append(
            Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task="Fix it"))
        )
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO, finish_reason="content_filter")]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        agent._messages = []

        exit_status, result = agent.run("Fix it")
        assert exit_status == "UnexpectedFinishReason"
        assert "content_filter" in result


# --- run() integration tests ---


class TestRun:
    def test_run_final_only(self, git_env, enc):
        config = AgentConfig()
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=git_env, task_id="test__test-1", config=config)
        # We need to pre-compute prompt_text — but run() builds messages itself.
        # So we simulate run()'s message setup to compute prompt_text, then build the tape.
        agent._messages = []
        agent._messages.append(
            get_system_message(
                system_instructions=config.system_instructions if config.system_instructions else None,
                reasoning_effort=config.reasoning_effort,
                include_repo_browser_tools=True
                if [tool for tool in config.tools if "repo_browser." in tool]
                else False,
                include_container_tools=True if [tool for tool in config.tools if "container." in tool] else False,
            )
        )
        agent._messages.append(
            get_developer_message(config.developer_instructions if config.developer_instructions else None)
        )
        agent._messages.append(
            Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task="Fix it"))
        )
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, FINAL_MESSAGE)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        # Reset messages so run() builds them fresh
        agent._messages = []

        exit_status, result = agent.run("Fix it")
        assert exit_status == "Submitted"

    def test_run_shell_then_final(self, git_env, enc):
        config = AgentConfig()
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=git_env, task_id="test__test-1", config=config)

        # Step 1: compute prompt_text for the initial messages
        agent._messages = []
        agent._messages.append(
            get_system_message(
                system_instructions=config.system_instructions if config.system_instructions else None,
                reasoning_effort=config.reasoning_effort,
                include_repo_browser_tools=True
                if [tool for tool in config.tools if "repo_browser." in tool]
                else False,
                include_container_tools=True if [tool for tool in config.tools if "container." in tool] else False,
            )
        )
        agent._messages.append(
            get_developer_message(config.developer_instructions if config.developer_instructions else None)
        )
        agent._messages.append(
            Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task="Fix it"))
        )
        prompt_text_1 = compute_prompt_text(agent, enc)

        # Step 2: simulate what happens after step 1 (shell echo hello) to get prompt_text_2
        tape_step1 = [make_tape_entry(prompt_text_1, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape_step1))
        received, obs, is_final = agent.query()
        agent._messages.extend(received)
        agent._messages.extend(obs)
        prompt_text_2 = compute_prompt_text(agent, enc)

        # Build the full 2-step tape
        full_tape = [
            make_tape_entry(prompt_text_1, SHELL_ECHO_HELLO),
            make_tape_entry(prompt_text_2, FINAL_MESSAGE),
        ]
        agent.model = TapeModel(config=TapeModelConfig(tape=full_tape))
        agent._messages = []

        exit_status, result = agent.run("Fix it")
        assert exit_status == "Submitted"
        assert agent.model.n_calls == 2

    def test_run_limits_exceeded(self, local_env, enc):
        config = AgentConfig(step_limit=1)
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=local_env, task_id="test__test-1", config=config)

        # Pre-compute prompt_text
        agent._messages = []
        agent._messages.append(
            get_system_message(
                system_instructions=config.system_instructions if config.system_instructions else None,
                reasoning_effort=config.reasoning_effort,
                include_repo_browser_tools=True
                if [tool for tool in config.tools if "repo_browser." in tool]
                else False,
                include_container_tools=True if [tool for tool in config.tools if "container." in tool] else False,
            )
        )
        agent._messages.append(
            get_developer_message(config.developer_instructions if config.developer_instructions else None)
        )
        agent._messages.append(
            Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task="Fix it"))
        )
        prompt_text = compute_prompt_text(agent, enc)

        # First call succeeds (shell), second call hits step limit
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        agent._messages = []

        exit_status, result = agent.run("Fix it")
        assert exit_status == "LimitsExceeded"

    def test_run_nonterminating_exception_after_retry_exhaustion(self, git_env, enc, monkeypatch):
        """After retries are exhausted, retry_error_callback raises RetrialsExceeded
        (a TerminatingException), which run() catches and returns."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        from harmonyagent.agents.harmony_agent import HarmonyAgent, _raise_terminating_exception

        # Patch the retry decorator to attempt only once with no wait
        fast_retry = retry(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            retry=retry_if_exception_type(Exception),
            retry_error_callback=_raise_terminating_exception,
        )
        original_query = HarmonyAgent.query.__wrapped__
        monkeypatch.setattr(HarmonyAgent, "query", fast_retry(original_query))

        config = AgentConfig(step_limit=2)
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=git_env, task_id="test__test-1", config=config)

        # Pre-compute prompt_text
        agent._messages = []
        agent._messages.append(
            get_system_message(
                system_instructions=config.system_instructions if config.system_instructions else None,
                reasoning_effort=config.reasoning_effort,
                include_repo_browser_tools=True
                if [tool for tool in config.tools if "repo_browser." in tool]
                else False,
                include_container_tools=True if [tool for tool in config.tools if "container." in tool] else False,
            )
        )
        agent._messages.append(
            get_developer_message(config.developer_instructions if config.developer_instructions else None)
        )
        agent._messages.append(
            Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task="Fix it"))
        )
        prompt_text_1 = compute_prompt_text(agent, enc)

        tape = [
            make_tape_entry(prompt_text_1, ANALYSIS_ONLY),  # fails → RetrialsExceeded
        ]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        agent._messages = []

        exit_status, result = agent.run("Fix it")
        assert exit_status == "RetrialsExceeded"
        assert "Failed after 1 attempts" in result


# --- run() tool config propagation tests ---


class TestRunToolConfig:
    @staticmethod
    def _setup_run(agent, config, enc, response_text, task="Fix it"):
        """Simulate run()'s message setup to compute prompt_text, then build tape."""
        agent._messages = []
        agent._messages.append(
            get_system_message(
                system_instructions=config.system_instructions if config.system_instructions else None,
                reasoning_effort=config.reasoning_effort,
                include_repo_browser_tools=True
                if [tool for tool in config.tools if "repo_browser." in tool]
                else False,
                include_container_tools=True if [tool for tool in config.tools if "container." in tool] else False,
            )
        )
        agent._messages.append(
            get_developer_message(config.developer_instructions if config.developer_instructions else None)
        )
        agent._messages.append(
            Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task=task))
        )
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, response_text)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        agent._messages = []

    def test_run_shell_only_submits(self, git_env, enc):
        """Agent with only shell tool can run and submit."""
        config = AgentConfig(tools=["container.exec"])
        agent = HarmonyAgent(
            model=TapeModel(config=TapeModelConfig(tape=[])), env=git_env, task_id="test__test-1", config=config
        )
        self._setup_run(agent, config, enc, FINAL_MESSAGE)

        exit_status, result = agent.run("Fix it")
        assert exit_status == "Submitted"

    def test_run_apply_patch_only_submits(self, git_env, enc):
        """Agent with only apply_patch tool can run and submit."""
        config = AgentConfig(tools=["repo_browser.apply_patch"])
        agent = HarmonyAgent(
            model=TapeModel(config=TapeModelConfig(tape=[])), env=git_env, task_id="test__test-1", config=config
        )
        self._setup_run(agent, config, enc, FINAL_MESSAGE)

        exit_status, result = agent.run("Fix it")
        assert exit_status == "Submitted"

    def test_run_no_tools_submits(self, git_env, enc):
        """Agent with no tools can still run and submit a final message."""
        config = AgentConfig(tools=[])
        agent = HarmonyAgent(
            model=TapeModel(config=TapeModelConfig(tape=[])), env=git_env, task_id="test__test-1", config=config
        )
        self._setup_run(agent, config, enc, FINAL_MESSAGE)

        exit_status, result = agent.run("Fix it")
        assert exit_status == "Submitted"

    def test_run_shell_call_without_shell_tool(self, local_env, enc, monkeypatch):
        """Model calls shell but only apply_patch is configured → UnknownToolCalled → RetrialsExceeded."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        from harmonyagent.agents.harmony_agent import HarmonyAgent, _raise_terminating_exception

        fast_retry = retry(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            retry=retry_if_exception_type(Exception),
            retry_error_callback=_raise_terminating_exception,
        )
        monkeypatch.setattr(HarmonyAgent, "query", fast_retry(HarmonyAgent.query.__wrapped__))

        config = AgentConfig(tools=["repo_browser.apply_patch"])
        agent = HarmonyAgent(
            model=TapeModel(config=TapeModelConfig(tape=[])), env=local_env, task_id="test__test-1", config=config
        )
        self._setup_run(agent, config, enc, SHELL_ECHO_HELLO)

        exit_status, result = agent.run("Fix it")
        assert exit_status == "RetrialsExceeded"
        assert "UnknownToolCalled" in result

    def test_run_apply_patch_call_without_apply_patch_tool(self, local_env, enc, monkeypatch):
        """Model calls apply_patch but only shell is configured → UnknownToolCalled → RetrialsExceeded."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        from harmonyagent.agents.harmony_agent import HarmonyAgent, _raise_terminating_exception

        fast_retry = retry(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            retry=retry_if_exception_type(Exception),
            retry_error_callback=_raise_terminating_exception,
        )
        monkeypatch.setattr(HarmonyAgent, "query", fast_retry(HarmonyAgent.query.__wrapped__))

        config = AgentConfig(tools=["container.exec"])
        agent = HarmonyAgent(
            model=TapeModel(config=TapeModelConfig(tape=[])), env=local_env, task_id="test__test-1", config=config
        )
        self._setup_run(agent, config, enc, APPLY_PATCH)

        exit_status, result = agent.run("Fix it")
        assert exit_status == "RetrialsExceeded"
        assert "UnknownToolCalled" in result

    def test_run_tool_call_with_no_tools(self, local_env, enc, monkeypatch):
        """Model calls shell but no tools are configured → UnknownToolCalled → RetrialsExceeded."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        from harmonyagent.agents.harmony_agent import HarmonyAgent, _raise_terminating_exception

        fast_retry = retry(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            retry=retry_if_exception_type(Exception),
            retry_error_callback=_raise_terminating_exception,
        )
        monkeypatch.setattr(HarmonyAgent, "query", fast_retry(HarmonyAgent.query.__wrapped__))

        config = AgentConfig(tools=[])
        agent = HarmonyAgent(
            model=TapeModel(config=TapeModelConfig(tape=[])), env=local_env, task_id="test__test-1", config=config
        )
        self._setup_run(agent, config, enc, SHELL_ECHO_HELLO)

        exit_status, result = agent.run("Fix it")
        assert exit_status == "RetrialsExceeded"
        assert "UnknownToolCalled" in result
        """Model calls python but only shell is configured → UnknownToolCalled → RetrialsExceeded."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        from harmonyagent.agents.harmony_agent import HarmonyAgent, _raise_terminating_exception

        fast_retry = retry(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            retry=retry_if_exception_type(Exception),
            retry_error_callback=_raise_terminating_exception,
        )
        monkeypatch.setattr(HarmonyAgent, "query", fast_retry(HarmonyAgent.query.__wrapped__))

        config = AgentConfig(tools=["container.exec"])
        agent = HarmonyAgent(
            model=TapeModel(config=TapeModelConfig(tape=[])), env=local_env, task_id="test__test-1", config=config
        )
        self._setup_run(agent, config, enc, PYTHON_PRINT_42)

        exit_status, result = agent.run("Fix it")
        assert exit_status == "RetrialsExceeded"
        assert "UnknownToolCalled" in result


# --- system_instructions tests ---


class TestSystemInstructions:
    @staticmethod
    def _setup_run_with_system_instructions(agent, config, enc, response_text, task="Fix it"):
        """Mirror run()'s message setup including system_instructions, then build tape."""
        agent._messages = []
        agent._messages.append(
            get_system_message(
                system_instructions=config.system_instructions if config.system_instructions else None,
                reasoning_effort=config.reasoning_effort,
                include_repo_browser_tools=True
                if [tool for tool in config.tools if "repo_browser." in tool]
                else False,
                include_container_tools=True if [tool for tool in config.tools if "container." in tool] else False,
            )
        )
        agent._messages.append(
            get_developer_message(config.developer_instructions if config.developer_instructions else None)
        )
        agent._messages.append(
            Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task=task))
        )
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, response_text)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        agent._messages = []

    def test_default_system_instructions_in_prompt(self, git_env, enc):
        """Default system_instructions is included in the rendered prompt."""
        config = AgentConfig()
        agent = HarmonyAgent(
            model=TapeModel(config=TapeModelConfig(tape=[])), env=git_env, task_id="test__test-1", config=config
        )
        self._setup_run_with_system_instructions(agent, config, enc, FINAL_MESSAGE)

        exit_status, _ = agent.run("Fix it")
        assert exit_status == "Submitted"

    def test_custom_system_instructions_in_prompt(self, git_env, enc):
        """Custom system_instructions changes the prompt and agent still submits."""
        custom_instructions = "You are a math tutor who helps students."
        config = AgentConfig(system_instructions=custom_instructions)
        agent = HarmonyAgent(
            model=TapeModel(config=TapeModelConfig(tape=[])), env=git_env, task_id="test__test-1", config=config
        )
        self._setup_run_with_system_instructions(agent, config, enc, FINAL_MESSAGE)

        exit_status, _ = agent.run("Fix it")
        assert exit_status == "Submitted"

    def test_custom_system_instructions_appears_in_prompt_text(self, git_env, enc):
        """Custom system_instructions text appears in the rendered prompt."""
        custom_instructions = "You are a specialized code reviewer."
        config = AgentConfig(system_instructions=custom_instructions)
        agent = HarmonyAgent(
            model=TapeModel(config=TapeModelConfig(tape=[])), env=git_env, task_id="test__test-1", config=config
        )
        agent._messages = []
        agent._messages.append(
            get_system_message(
                system_instructions=config.system_instructions if config.system_instructions else None,
                reasoning_effort=config.reasoning_effort,
                include_repo_browser_tools=True
                if [tool for tool in config.tools if "repo_browser." in tool]
                else False,
                include_container_tools=True if [tool for tool in config.tools if "container." in tool] else False,
            )
        )
        agent._messages.append(
            get_developer_message(config.developer_instructions if config.developer_instructions else None)
        )
        agent._messages.append(
            Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task="Fix it"))
        )
        prompt_text = compute_prompt_text(agent, enc)
        assert custom_instructions in prompt_text

    def test_empty_system_instructions_omits_model_identity(self, git_env, enc):
        """Empty system_instructions produces a different prompt than with instructions."""
        custom_instructions = "You are a specialized code reviewer."
        config_with = AgentConfig(system_instructions=custom_instructions)
        config_without = AgentConfig(system_instructions="")

        def _build_prompt(config):
            agent = HarmonyAgent(
                model=TapeModel(config=TapeModelConfig(tape=[])), env=git_env, task_id="test__test-1", config=config
            )
            agent._messages = []

            agent._messages.append(
                get_system_message(
                    system_instructions=config.system_instructions if config.system_instructions else None,
                    reasoning_effort=config.reasoning_effort,
                    include_repo_browser_tools=True
                    if [tool for tool in config.tools if "repo_browser." in tool]
                    else False,
                    include_container_tools=True if [tool for tool in config.tools if "container." in tool] else False,
                )
            )
            agent._messages.append(
                get_developer_message(config.developer_instructions if config.developer_instructions else None)
            )
            agent._messages.append(
                Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task="Fix it"))
            )
            return compute_prompt_text(agent, enc)

        prompt_with = _build_prompt(config_with)
        prompt_without = _build_prompt(config_without)
        assert custom_instructions in prompt_with
        assert custom_instructions not in prompt_without
        assert prompt_with != prompt_without

    def test_run_custom_system_instructions_shell_then_final(self, git_env, enc):
        """Multi-step run with custom system_instructions: shell call then submit."""
        custom_instructions = "You are an expert debugger."
        config = AgentConfig(system_instructions=custom_instructions)
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=git_env, task_id="test__test-1", config=config)

        # Step 1: compute prompt_text for initial messages
        agent._messages = []
        agent._messages.append(
            get_system_message(
                system_instructions=config.system_instructions if config.system_instructions else None,
                reasoning_effort=config.reasoning_effort,
                include_repo_browser_tools=True
                if [tool for tool in config.tools if "repo_browser." in tool]
                else False,
                include_container_tools=True if [tool for tool in config.tools if "container." in tool] else False,
            )
        )
        agent._messages.append(
            get_developer_message(config.developer_instructions if config.developer_instructions else None)
        )
        agent._messages.append(
            Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task="Fix it"))
        )
        prompt_text_1 = compute_prompt_text(agent, enc)

        # Step 2: simulate step 1 to get prompt_text_2
        tape_step1 = [make_tape_entry(prompt_text_1, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape_step1))
        received, obs, is_final = agent.query()
        agent._messages.extend(received)
        agent._messages.extend(obs)
        prompt_text_2 = compute_prompt_text(agent, enc)

        # Build full 2-step tape
        full_tape = [
            make_tape_entry(prompt_text_1, SHELL_ECHO_HELLO),
            make_tape_entry(prompt_text_2, FINAL_MESSAGE),
        ]
        agent.model = TapeModel(config=TapeModelConfig(tape=full_tape))
        agent._messages = []

        exit_status, _ = agent.run("Fix it")
        assert exit_status == "Submitted"
        assert agent.model.n_calls == 2
