from openai_harmony import (
    DeveloperContent,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
)

from harmonyagent.agents.tools.tool_registry import *

REASONING_EFFORT_MAP = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


def get_system_message(
    system_instructions: str | None = None,
    reasoning_effort: str = "medium",
    include_repo_browser_tools: bool = False,
    include_container_tools: bool = False,
) -> Message:

    system_message_content = SystemContent.new().with_reasoning_effort(REASONING_EFFORT_MAP[reasoning_effort])
    if system_instructions:
        system_message_content = system_message_content.with_model_identity(system_instructions)
    if include_repo_browser_tools:
        system_message_content = system_message_content.with_tools(REPO_BROWSER_NS)
    if include_container_tools:
        system_message_content = system_message_content.with_tools(CONTAINER_NS)
    return Message.from_role_and_content(Role.SYSTEM, system_message_content)


def get_developer_message(
    developer_instructions: str | None = None,
    include_apply_patch_dev_ins: bool = False,
    additional_tools: list[str] | None = None,
) -> Message:
    aggregated_developer_instructions = ""
    tools = []

    if developer_instructions:
        aggregated_developer_instructions += developer_instructions + "\n"
    if include_apply_patch_dev_ins:
        aggregated_developer_instructions += APPLY_PATCH_DEVELOPER_INSTRUCTIONS
    developer_message_content = DeveloperContent.new()
    if aggregated_developer_instructions:
        developer_message_content = developer_message_content.with_instructions(aggregated_developer_instructions)
    if tools:
        developer_message_content = developer_message_content.with_function_tools(tools)
    return Message.from_role_and_content(Role.DEVELOPER, developer_message_content)
