import json

from openai_harmony import Message


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""

    def __init__(self, limit_type: str, current_counter: int | float, limit: int | float):
        super().__init__(f"limit {limit_type} exceeded: {current_counter}/{limit}")


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""

    def __init__(self, received_messages: list[Message]):
        super().__init__(json.dumps([m.to_dict() for m in received_messages]))


class MaxContextWindowOverflow(TerminatingException):
    """Raised when max context window of the model is reached."""

    def __init__(self, current_token_usage: int, max_context_window: int):
        super().__init__(f"max context window of model is reached: {current_token_usage}/{max_context_window}")


class UnexpectedFinishReason(TerminatingException):
    """Raised finish reason is not stop."""

    def __init__(self, finish_reason: str):
        super().__init__(f"unexpected finish reason: {finish_reason}")


class MaxNewTokensExceeded(TerminatingException):
    """Raised when max_tokens is exceeded."""

    def __init__(self, num_new_tokens: int, max_new_tokens: int):
        super().__init__(f"max new tokens exceeded: {num_new_tokens}/{max_new_tokens}")


class RetrialsExceeded(TerminatingException):
    """Raised when retrials on NonTerminatingException 's are exceeded."""

    def __init__(self, retrial_attempts: int, last_exception_class: str, last_exception: str):
        super().__init__(
            f"Failed after {retrial_attempts} attempts, last exception: {last_exception_class} {last_exception}"
        )
