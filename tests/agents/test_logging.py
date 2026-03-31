import logging

import pytest
from openai_harmony import (
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    RenderConversationConfig,
    Role,
    SystemContent,
    load_harmony_encoding,
)

from harmonyagent.agents.agent_config import AgentConfig
from harmonyagent.agents.harmony_agent import HarmonyAgent
from harmonyagent.agents.harmony_core.harmony_parsers import create_prompt_text
from harmonyagent.agents.harmony_core.sys_dev_message import get_developer_message, get_system_message
from harmonyagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from harmonyagent.models.test_models import TapeModel, TapeModelConfig

# --- Harmony response text constants ---

SHELL_ECHO_HELLO = (
    "<|channel|>analysis<|message|>Let me check.<|end|>"
    "<|start|>assistant<|channel|>analysis to=functions.container.exec code<|message|>"
    '{"command":["bash","-lc","echo hello"]}'
)

FINAL_MESSAGE = (
    "<|channel|>analysis<|message|>All done.<|end|><|start|>assistant<|channel|>final<|message|>Task completed."
)


# --- Fixtures ---


@pytest.fixture(autouse=True, scope="session")
def _replace_rich_handler():
    """Replace Rich logging handler with a plain StreamHandler to avoid Rich TypeError in tests."""
    agent_logger = logging.getLogger("harmonyagent")
    rich_handlers = [h for h in agent_logger.handlers if type(h).__name__ == "RichHandler"]
    for h in rich_handlers:
        agent_logger.removeHandler(h)
    plain = logging.StreamHandler()
    plain.setFormatter(logging.Formatter("%(name)s: %(levelname)s: %(message)s"))
    agent_logger.addHandler(plain)
    yield
    agent_logger.removeHandler(plain)
    for h in rich_handlers:
        agent_logger.addHandler(h)


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
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(ReasoningEffort.MEDIUM))
    )
    agent._messages.append(get_developer_message(config.developer_instructions))
    agent._messages.append(
        Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task=task))
    )
    return agent


# --- query() log presence tests ---


class TestQueryLogPresence:
    def test_query_logs_reasoning(self, local_env, enc, caplog):
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        assert any("reasoning" in r.message for r in caplog.records)

    def test_query_logs_response_text(self, local_env, enc, caplog):
        """response_text is logged on every query() call."""
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        response_logs = [r.message for r in caplog.records if "response_text:" in r.message]
        assert len(response_logs) == 1

    def test_query_logs_stats(self, local_env, enc, caplog):
        """Token counts and finish_reason are logged on every query() call."""
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        stats_logs = [r.message for r in caplog.records if "context tokens" in r.message]
        assert len(stats_logs) == 1
        assert "generated tokens" in stats_logs[0]
        assert "finish_reason: stop" in stats_logs[0]

    def test_query_logs_contain_step_and_calls(self, local_env, enc, caplog):
        """All query() log lines include step= and calls= fields."""
        config = AgentConfig()
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        agent_logs = [r.message for r in caplog.records if "test__test-123" in r.message]
        assert len(agent_logs) > 0
        for msg in agent_logs:
            assert "step=" in msg
            assert "calls=" in msg


# --- run() log presence tests ---


class TestRunLogPresence:
    def test_run_logs_terminating_exception(self, git_env, enc, caplog):
        """TerminatingException is logged when run() exits."""
        config = AgentConfig()
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=git_env, task_id="test__test-1", config=config)
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
        agent._messages = []

        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.run("Fix it")
        term_logs = [r.message for r in caplog.records if "TerminatingException:" in r.message]
        assert len(term_logs) == 1
        assert "Submitted" in term_logs[0]


# --- log_verbosity tests ---


class TestLogVerbosity:
    def test_response_text_log_truncated(self, local_env, enc, caplog):
        """Raw response_text is truncated to log_verbosity chars in log output."""
        config = AgentConfig(log_verbosity=5)
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        response_logs = [r.message for r in caplog.records if "response_text:" in r.message]
        assert len(response_logs) == 1
        # Full response_text starts with "<|channel|>analysis..." but only first 5 chars are repr'd
        assert SHELL_ECHO_HELLO not in response_logs[0]

    def test_reasoning_log_truncated(self, local_env, enc, caplog):
        """Reasoning text is truncated to log_verbosity chars in log output."""
        config = AgentConfig(log_verbosity=5)
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        reasoning_logs = [r.message for r in caplog.records if "reasoning:" in r.message]
        assert len(reasoning_logs) == 1
        # "Let me check." → truncated to "Let m" (5 chars)
        assert "'Let m'" in reasoning_logs[0]
        assert "Let me check." not in reasoning_logs[0]

    def test_reasoning_log_not_truncated_when_verbosity_large(self, local_env, enc, caplog):
        """With large log_verbosity, full reasoning text appears in log."""
        config = AgentConfig(log_verbosity=1000)
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        reasoning_logs = [r.message for r in caplog.records if "reasoning:" in r.message]
        assert len(reasoning_logs) == 1
        assert "Let me check." in reasoning_logs[0]

    def test_final_message_log_truncated(self, git_env, enc, caplog):
        """Final message text is truncated to log_verbosity chars in log output."""
        config = AgentConfig(log_verbosity=5)
        agent = setup_agent_for_query(git_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, FINAL_MESSAGE)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        final_logs = [r.message for r in caplog.records if "final_message:" in r.message]
        assert len(final_logs) == 1
        # "Task completed." → truncated to "Task " (5 chars)
        assert "'Task '" in final_logs[0]
        assert "Task completed." not in final_logs[0]

    def test_tool_output_log_truncated(self, local_env, enc, caplog):
        """Tool result output is truncated to log_verbosity chars in log output."""
        config = AgentConfig(log_verbosity=3)
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        output_logs = [r.message for r in caplog.records if "return code:" in r.message]
        assert len(output_logs) == 1
        # "hello\n" → truncated to "hel" (3 chars)
        assert "'hel'" in output_logs[0]
        assert "'hello" not in output_logs[0]

    def test_tool_name_log_not_truncated(self, local_env, enc, caplog):
        """Tool name is logged in full regardless of log_verbosity."""
        config = AgentConfig(log_verbosity=3)
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        tool_logs = [r.message for r in caplog.records if "tool called:" in r.message]
        assert len(tool_logs) == 1
        assert "'container.exec" in tool_logs[0]

    def test_tool_args_log_truncated(self, local_env, enc, caplog):
        """Tool args are truncated to log_verbosity chars in log output."""
        config = AgentConfig(log_verbosity=5)
        agent = setup_agent_for_query(local_env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, SHELL_ECHO_HELLO)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        args_logs = [r.message for r in caplog.records if "received arguments:" in r.message]
        assert len(args_logs) == 1
        # tool_args is JSON like '{"command":["bash","-lc","echo hello"]}'
        # With log_verbosity=5, only first 5 chars are kept, then repr'd
        full_args = '{"command":["bash","-lc","echo hello"]}'
        assert repr(full_args[:5]) in args_logs[0]
        assert full_args not in args_logs[0]

    def test_task_log_truncated_in_run(self, git_env, enc, caplog):
        """Task text is truncated to log_verbosity chars in run() log output."""
        long_task = "A" * 200
        config = AgentConfig(log_verbosity=100)
        model = TapeModel(config=TapeModelConfig(tape=[]))
        agent = HarmonyAgent(model=model, env=git_env, task_id="test__test-1", config=config)
        # Pre-compute prompt_text by simulating run()'s message setup
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
            Message.from_role_and_content(Role.USER, agent.render_template(config.instance_template, task=long_task))
        )
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, FINAL_MESSAGE)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        agent._messages = []

        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.run(long_task)
        task_logs = [r.message for r in caplog.records if "task:" in r.message]
        assert len(task_logs) == 1
        # With log_verbosity=10, only first 10 chars of the task should appear
        assert "A" * 26 in task_logs[0]
        assert "A" * 200 not in task_logs[0]

    def test_timeout_observation_log_truncated(self, enc, tmp_path, caplog):
        """Timeout observation is truncated to log_verbosity chars in log output."""
        env = LocalEnvironment(config=LocalEnvironmentConfig(cwd=str(tmp_path), timeout=1))
        config = AgentConfig(log_verbosity=5)
        response_text = (
            "<|channel|>analysis<|message|>Running.<|end|>"
            "<|start|>assistant<|channel|>analysis to=functions.container.exec code<|message|>"
            '{"command":["bash","-lc","sleep 10"]}'
        )
        agent = setup_agent_for_query(env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, response_text)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        timeout_logs = [r.message for r in caplog.records if "timed out" in r.message]
        assert len(timeout_logs) == 1
        # The observation string starts with "The last command timed out..."
        # With log_verbosity=5, only first 5 chars should be repr'd
        assert "'The l'" in timeout_logs[0]

    def test_timeout_log_contains_cmd_and_timeout(self, enc, tmp_path, caplog):
        """Timeout log message includes cmd= and timeout duration."""
        env = LocalEnvironment(config=LocalEnvironmentConfig(cwd=str(tmp_path), timeout=1))
        config = AgentConfig(log_verbosity=200)
        response_text = (
            "<|channel|>analysis<|message|>Running.<|end|>"
            "<|start|>assistant<|channel|>analysis to=functions.container.exec code<|message|>"
            '{"command":["bash","-lc","sleep 10"]}'
        )
        agent = setup_agent_for_query(env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, response_text)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            agent.query()
        timeout_logs = [r.message for r in caplog.records if "timed out" in r.message]
        assert len(timeout_logs) == 1
        assert "cmd=" in timeout_logs[0]
        assert "after 1s" in timeout_logs[0]

    def test_timeout_observation_includes_timeout_seconds(self, enc, tmp_path, caplog):
        """Rendered timeout observation includes the timeout duration in seconds."""
        env = LocalEnvironment(config=LocalEnvironmentConfig(cwd=str(tmp_path), timeout=1))
        config = AgentConfig(log_verbosity=500)
        response_text = (
            "<|channel|>analysis<|message|>Running.<|end|>"
            "<|start|>assistant<|channel|>analysis to=functions.container.exec code<|message|>"
            '{"command":["bash","-lc","sleep 10"]}'
        )
        agent = setup_agent_for_query(env, config, tape=[], task="Fix it")
        prompt_text = compute_prompt_text(agent, enc)
        tape = [make_tape_entry(prompt_text, response_text)]
        agent.model = TapeModel(config=TapeModelConfig(tape=tape))
        with caplog.at_level(logging.INFO, logger="harmonyagent"):
            _, observation_messages, _ = agent.query()
        # The observation message content should contain "after 1 seconds"
        obs_text = observation_messages[0].content[0].text
        assert "after 1 seconds" in obs_text
