from pathlib import Path

from openai_harmony import DeveloperContent, Message, Role, SystemContent

from harmonyagent.agents.harmony_core.sys_dev_message import (
    REASONING_EFFORT_MAP,
    get_developer_message,
    get_system_message,
)

APPLY_PATCH_INS = Path("src/harmonyagent/agents/tools/apply_patch.md").read_text()


# --- get_system_message ---


def test_get_system_message_returns_system_role():
    msg = get_system_message()
    assert msg.author.role == Role.SYSTEM


def test_get_system_message_default_medium_effort():
    msg = get_system_message()
    content = msg.content[0]
    assert isinstance(content, SystemContent)
    assert content.reasoning_effort == REASONING_EFFORT_MAP["medium"]


def test_get_system_message_high_effort():
    msg = get_system_message(reasoning_effort="high")
    assert msg.content[0].reasoning_effort == REASONING_EFFORT_MAP["high"]


def test_get_system_message_low_effort():
    msg = get_system_message(reasoning_effort="low")
    assert msg.content[0].reasoning_effort == REASONING_EFFORT_MAP["low"]


def test_get_system_message_with_instructions():
    msg = get_system_message(system_instructions="You are a coding assistant.")
    assert msg.content[0].model_identity == "You are a coding assistant."


def test_get_system_message_without_instructions():
    msg = get_system_message()
    # model_identity should be the default, not our custom one
    assert msg.content[0].model_identity != "custom"


# --- get_developer_message ---


def test_get_developer_message_returns_developer_role():
    msg = get_developer_message()
    assert msg.author.role == Role.DEVELOPER


def test_get_developer_message_is_message():
    msg = get_developer_message()
    assert isinstance(msg, Message)


def test_get_developer_message_no_args_has_no_tools():
    msg = get_developer_message()
    content = msg.content[0]
    assert isinstance(content, DeveloperContent)
    assert not content.tools


def test_get_developer_message_no_args_has_no_instructions():
    msg = get_developer_message()
    content = msg.content[0]
    assert not content.instructions


def test_get_developer_message_with_instructions():
    msg = get_developer_message(developer_instructions="Do X")
    content = msg.content[0]
    assert "Do X" in content.instructions
