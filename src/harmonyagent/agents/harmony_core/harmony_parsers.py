from openai_harmony import (
    Conversation,
    HarmonyEncoding,
    HarmonyError,
    Message,
    RenderConversationConfig,
    Role,
)

from harmonyagent.agents.exceptions.agent_exceptions import *

ROLE_MAP = {
    "system": Role.SYSTEM,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL,
}


def create_prompt_text(
    messages: list[Message], encoder: HarmonyEncoding, convo_config: RenderConversationConfig, max_context_window: int
) -> tuple[str, str]:
    convo = Conversation.from_messages(messages)
    tokens = encoder.render_conversation_for_completion(convo, Role.ASSISTANT, convo_config)
    if len(tokens) >= max_context_window:
        raise MaxContextWindowOverflow(len(tokens), max_context_window)
    return encoder.decode(tokens), tokens


def parse_response_text(encoder: HarmonyEncoding, response_text) -> list[Message]:
    response_tokens = encoder.encode(response_text, allowed_special="all")
    try:
        messages = encoder.parse_messages_from_completion_tokens(response_tokens, role=Role.ASSISTANT, strict=True)
    except HarmonyError as e:
        raise HarmonyErrorWithDetails(str(e), response_text)
    return messages


def validate_messages(messages: list[Message]) -> None:
    for message in messages:
        channel = getattr(message, "channel", None)
        if message.author.role != ROLE_MAP["system"] and not channel:
            raise HarmonyMessageMissingChannel(message)


def get_reasoning_message(messages: list[Message]) -> Message | None:
    reasoning_messages = [m for m in messages if m.channel == "analysis" and not m.recipient]
    if len(reasoning_messages) > 1:
        raise MultipleReasoningMessages(reasoning_messages)
    return reasoning_messages[0] if reasoning_messages else None


def get_tool_call(messages: list[Message]) -> Message | None:
    tool_calls = [m for m in messages if m.recipient]
    if len(tool_calls) > 1:
        raise MultipleToolCalls(tool_calls)
    return tool_calls[0] if tool_calls else None


def get_final_message(messages: list[Message]) -> Message | None:
    final_messages = [m for m in messages if m.channel == "final"]
    if len(final_messages) > 1:
        raise MultipleFinalMessages(final_messages)
    return final_messages[0] if final_messages else None


def get_tool_name(tool_call: Message) -> str:
    if not isinstance(tool_call.recipient, str) or tool_call.recipient == "":
        raise ToolNameParsingError(tool_call.recipient)
    return tool_call.recipient.removeprefix("functions.")


def get_tool_args(tool_call: Message) -> str:
    return tool_call.content[0].text


def validate_generation_length(response: dict, response_tokens: list[int], max_new_tokens: int):
    if response["choices"][0]["finish_reason"] != "stop":
        if response["choices"][0]["finish_reason"] == "length":
            raise LongGeneration(response)
        else:
            raise UnexpectedFinishReason(response["choices"][0]["finish_reason"])
    if len(response_tokens) >= max_new_tokens:
        raise MaxNewTokensExceeded(len(response_tokens), max_new_tokens)
