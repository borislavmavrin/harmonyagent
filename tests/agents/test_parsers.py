import json

import pytest
from openai_harmony import HarmonyEncodingName, Message, RenderConversationConfig, Role, load_harmony_encoding

from harmonyagent.agents.agent_config import AgentConfig
from harmonyagent.agents.exceptions.nonterminating_exceptions import (
    HarmonyErrorWithDetails,
    HarmonyMessageMissingChannel,
    LongGeneration,
    MultipleFinalMessages,
    MultipleReasoningMessages,
    MultipleToolCalls,
    ToolNameParsingError,
)
from harmonyagent.agents.exceptions.terminating_exceptions import (
    MaxContextWindowOverflow,
    MaxNewTokensExceeded,
    UnexpectedFinishReason,
)
from harmonyagent.agents.harmony_core.harmony_parsers import (
    create_prompt_text,
    get_final_message,
    get_reasoning_message,
    get_tool_args,
    get_tool_call,
    get_tool_name,
    parse_response_text,
    validate_generation_length,
    validate_messages,
)


@pytest.fixture
def encoder():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


@pytest.fixture
def convo_config():
    config = RenderConversationConfig()
    config.auto_drop_analysis = False
    return config


# --- create_prompt_text ---


def test_create_prompt_text_success(encoder, convo_config):
    msgs = [_system_msg("hello")]
    text, tokens = create_prompt_text(msgs, encoder, convo_config, AgentConfig.max_context_window)
    assert isinstance(text, str)
    assert "hello" in text
    assert len(tokens) > 0
    assert len(tokens) < AgentConfig.max_context_window


def test_create_prompt_text_returns_decodable_tokens(encoder, convo_config):
    msgs = [_system_msg("test prompt")]
    text, tokens = create_prompt_text(msgs, encoder, convo_config, AgentConfig.max_context_window)
    assert encoder.decode(tokens) == text


def test_create_prompt_text_multiple_messages(encoder, convo_config):
    msgs = [
        _system_msg("system prompt"),
        _assistant_msg("reasoning", channel="analysis"),
    ]
    text, tokens = create_prompt_text(msgs, encoder, convo_config, AgentConfig.max_context_window)
    assert "system prompt" in text
    assert len(tokens) > 0


def test_create_prompt_text_overflow_message(encoder, convo_config):
    """MaxContextWindowOverflow message includes token count and limit."""
    msgs = [_system_msg("hello")]
    with pytest.raises(MaxContextWindowOverflow, match="max context window of model is reached: 7/1") as exc_info:
        create_prompt_text(msgs, encoder, convo_config, max_context_window=1)
    assert "/1" in str(exc_info.value)


# --- parse_response_text ---


def test_parse_response_text_success(encoder):
    response_text = "<|message|><|channel|>analysis\nI am thinking<|eom|>"
    result = parse_response_text(encoder, response_text)
    assert len(result) == 1


def test_parse_response_text_harmony_error(encoder):
    with pytest.raises(HarmonyErrorWithDetails):
        parse_response_text(encoder, "<|INVALID|><<<not valid harmony>>>")


def test_parse_response_text_empty(encoder):
    with pytest.raises(HarmonyErrorWithDetails):
        parse_response_text(encoder, "")


def _assistant_msg(text, channel=None, recipient=None):
    m = Message.from_role_and_content(Role.ASSISTANT, text)
    if channel:
        m = m.with_channel(channel)
    if recipient:
        m = m.with_recipient(recipient)
    return m


def _system_msg(text):
    return Message.from_role_and_content(Role.SYSTEM, text)


# --- validate_messages ---


def test_validate_messages_system_without_channel():
    """System messages don't need a channel."""
    validate_messages([_system_msg("system prompt")])


def test_validate_messages_with_channel():
    validate_messages([_assistant_msg("reasoning", channel="analysis")])


def test_validate_messages_missing_channel():
    with pytest.raises(HarmonyMessageMissingChannel, match=r"role:.*content:.*no channel"):
        validate_messages([_assistant_msg("no channel")])


def test_validate_messages_mixed_valid():
    msgs = [
        _system_msg("sys"),
        _assistant_msg("reasoning", channel="analysis"),
    ]
    validate_messages(msgs)


def test_validate_messages_mixed_with_missing_channel():
    msgs = [
        _system_msg("sys"),
        _assistant_msg("no channel"),
    ]
    with pytest.raises(HarmonyMessageMissingChannel, match=r"role:.*content:.*no channel"):
        validate_messages(msgs)


# --- get_final_message ---


def test_get_final_message_found():
    msg = _assistant_msg("done", channel="final")
    assert get_final_message([msg]) is msg


def test_get_final_message_none():
    assert get_final_message([_assistant_msg("reasoning", channel="analysis")]) is None


def test_get_final_message_empty():
    assert get_final_message([]) is None


def test_get_final_message_multiple():
    msgs = [
        _assistant_msg("done1", channel="final"),
        _assistant_msg("done2", channel="final"),
    ]
    with pytest.raises(MultipleFinalMessages, match="expected 1 final message, got 2: "):
        get_final_message(msgs)


# --- get_reasoning_message ---


def test_get_reasoning_message_found():
    msg = _assistant_msg("thinking...", channel="analysis")
    result = get_reasoning_message([msg])
    assert result is msg


def test_get_reasoning_message_none():
    msg = _assistant_msg("final answer", channel="final")
    assert get_reasoning_message([msg]) is None


def test_get_reasoning_message_excludes_tool_calls():
    """A message with channel=analysis but with a recipient is a tool call, not reasoning."""
    msgs = [
        _assistant_msg("thinking", channel="analysis"),
        _assistant_msg('{"command":["ls"]}', channel="analysis", recipient="functions.shell"),
    ]
    result = get_reasoning_message(msgs)
    assert result.content[0].text == "thinking"


def test_get_reasoning_message_empty():
    assert get_reasoning_message([]) is None


def test_get_reasoning_message_multiple():
    msgs = [
        _assistant_msg("thought 1", channel="analysis"),
        _assistant_msg("thought 2", channel="analysis"),
    ]
    with pytest.raises(MultipleReasoningMessages, match="expected 1 reasoning message, got 2: "):
        get_reasoning_message(msgs)


# --- get_tool_call ---


def test_get_tool_call_found():
    tool_msg = _assistant_msg(
        json.dumps({"command": ["ls"]}),
        channel="analysis",
        recipient="functions.shell",
    )
    reasoning = _assistant_msg("reasoning", channel="analysis")
    result = get_tool_call([reasoning, tool_msg])
    assert result is tool_msg


def test_get_tool_call_none():
    assert get_tool_call([_assistant_msg("reasoning", channel="analysis")]) is None


def test_get_tool_call_empty():
    assert get_tool_call([]) is None


def test_get_tool_call_multiple():
    msgs = [
        _assistant_msg('{"command":["ls"]}', channel="analysis", recipient="container.exec"),
        _assistant_msg('{"command":["pwd"]}', channel="analysis", recipient="container.exec"),
    ]
    with pytest.raises(MultipleToolCalls, match="expected 1 reasoning message, got 2: "):
        get_tool_call(msgs)


# --- get_tool_name ---


def test_get_tool_name_shell():
    msg = _assistant_msg('{"command":["ls"]}', channel="analysis", recipient="functions.container.exec")
    assert get_tool_name(msg) == "container.exec"


def test_get_tool_name_shell_with_suffix():
    """Returns full recipient including suffix — pick_tool handles prefix matching."""
    msg = _assistant_msg(
        '{"command":["ls"]}', channel="analysis", recipient="functions.container.exec<|channel|>commentary"
    )
    assert get_tool_name(msg) == "container.exec<|channel|>commentary"


def test_get_tool_name_apply_patch():
    msg = _assistant_msg('{"patch":"..."}', channel="analysis", recipient="functions.apply_patch")
    assert get_tool_name(msg) == "apply_patch"


def test_get_tool_name_returns_any_recipient():
    """get_tool_name no longer validates tool names — pick_tool does that."""
    msg = _assistant_msg("{}", channel="analysis", recipient="functions.unknown_tool")
    assert get_tool_name(msg) == "unknown_tool"


def test_get_tool_name_empty_recipient():
    msg = _assistant_msg("{}", channel="analysis", recipient="")
    with pytest.raises(ToolNameParsingError, match=r"recipient:"):
        get_tool_name(msg)


# --- get_tool_args ---


def test_get_tool_args_shell_command():
    msg = _assistant_msg(
        json.dumps({"command": ["bash", "-lc", "ls -R"]}),
        recipient="functions.shell",
    )
    args = get_tool_args(msg)
    assert args == json.dumps({"command": ["bash", "-lc", "ls -R"]})


def test_get_tool_args_apply_patch():
    patch_content = "*** Begin Patch\n--- a/file.py\n+++ b/file.py\n"
    msg = _assistant_msg(
        json.dumps({"patch": patch_content}),
        recipient="functions.apply_patch",
    )
    args = get_tool_args(msg)
    assert json.loads(args)["patch"] == patch_content


def test_get_tool_args_returns_raw_text():
    msg = _assistant_msg("not valid json{{{", recipient="functions.shell")
    args = get_tool_args(msg)
    assert args == "not valid json{{{"


# --- validate_generation_length ---


def _make_response(finish_reason="stop", text="hello"):
    return {"choices": [{"finish_reason": finish_reason, "text": text}]}


def test_validate_generation_length_stop():
    validate_generation_length(_make_response("stop"), [1, 2, 3], AgentConfig.max_new_tokens)


def test_validate_generation_length_length_raises():
    with pytest.raises(LongGeneration, match=r"finish_reason: length"):
        validate_generation_length(_make_response("length"), [1, 2, 3], AgentConfig.max_new_tokens)


def test_validate_generation_length_unexpected_reason_raises():
    with pytest.raises(UnexpectedFinishReason):
        validate_generation_length(_make_response("content_filter"), [1, 2, 3], AgentConfig.max_new_tokens)


def test_validate_generation_length_high_token_count_raises():
    """When tokens >= MAX_NEW_TOKENS but finish_reason is stop, raises MaxTokensExceeded."""
    tokens = list(range(AgentConfig.max_new_tokens))
    with pytest.raises(MaxNewTokensExceeded, match="max new tokens exceeded: 4096/4096"):
        validate_generation_length(_make_response("stop"), tokens, AgentConfig.max_new_tokens)
