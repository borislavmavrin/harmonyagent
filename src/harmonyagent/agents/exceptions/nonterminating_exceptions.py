from openai_harmony import Message


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""

    def __init__(self, output: str = "", cmd: str = "", timeout: int = 0):
        super().__init__(output)
        self.output = output
        self.cmd = cmd
        self.timeout = timeout


class MultipleReasoningMessages(NonTerminatingException):
    """Raised when multiple reasoning messages."""

    def __init__(self, reasoning_messages: list[Message]):
        super().__init__(
            f"expected 1 reasoning message, got {len(reasoning_messages)}: {[repr(m.content[0].text[:200]) for m in reasoning_messages]}"
        )


class ToolNameParsingError(NonTerminatingException):
    """Raised when tool name failed to parse."""

    def __init__(self, tool_call_recipient: str):
        super().__init__(f"failed to parse tool call recipient, recipient: {repr(tool_call_recipient)}")


class UnknownToolCalled(NonTerminatingException):
    """Unknown tool called."""

    def __init__(self, tool_name_called: str, tool_name_available: list[str]):
        super().__init__(f"tool called: {tool_name_called}, available: {tool_name_available}")


class MultipleToolCalls(NonTerminatingException):
    """Multiple tool calls."""

    def __init__(self, tool_calls: list[Message]):
        super().__init__(
            f"expected 1 reasoning message, got {len(tool_calls)}: {[repr(m.content[0].text[:200]) for m in tool_calls]}"
        )


class ToolCallArgParsingError(NonTerminatingException):
    """Failed to parse tool call arguments."""

    def __init__(self, tool_name: str, tool_args: str):
        super().__init__(f"failed to parse args for {tool_name}, args: {repr(tool_args[:200])}")


class UnknownToolCallArg(NonTerminatingException):
    """Tool call has unknown argument."""

    def __init__(self, tool_name: str, tool_args: list[str], expected_tool_args: list[str]):
        super().__init__(
            f"unknown tool args for {tool_name} got tool_args: {tool_args}, expected: {expected_tool_args}"
        )


class HarmonyErrorWithDetails(NonTerminatingException):
    """Harmony error with dump."""

    def __init__(self, harmony_error: str, response_text: str):
        super().__init__(
            f"failed to parse harmony response_text, harmony error: {harmony_error}, response_text: {response_text}"
        )


class HarmonyMessageMissingChannel(NonTerminatingException):
    """Raised when a non-system harmony message is missing a channel."""

    def __init__(self, harmony_message: Message):
        super().__init__(
            f"harmony message missing channel role: {harmony_message.author.role}, content: {repr(harmony_message.content[0].text[:200])}"
        )


class MultipleFinalMessages(NonTerminatingException):
    """Raised when multiple final messages are found."""

    def __init__(self, final_messages: list[Message]):
        super().__init__(
            f"expected 1 final message, got {len(final_messages)}: {[repr(m.content[0].text[:200]) for m in final_messages]}"
        )


class ToolCallAndFinalMessage(NonTerminatingException):
    """Raised when both tool call and final message are received."""

    def __init__(self, tool_call_message: Message, final_message: Message):
        super().__init__(
            f"received both tool call and final message, tool call recipient: {tool_call_message.recipient}, final message content: {repr(final_message.content[0].text[:200])}"
        )


class NoToolCallNoFinalMessage(NonTerminatingException):
    """Raised when no tool call and no final message are received."""

    def __init__(self, received_messages: list[Message]):
        super().__init__(
            f"no tool call and no final message are received, got: {len(received_messages)} messages, channels: {[getattr(m, 'channel', None) for m in received_messages]}, content: {[repr(m.content[0].text[:200]) for m in received_messages]}"
        )


class LongGeneration(NonTerminatingException):
    """Raised when generation exceeded max_tokens."""

    def __init__(self, response: dict):
        super().__init__(
            f"generation was truncated, finish_reason: {response['choices'][0]['finish_reason']}, usage: {response.get('usage')}"
        )
