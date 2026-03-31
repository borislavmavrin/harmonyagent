import traceback
from typing import Any

from harmonyagent.aime25.harmony_adapter import HarmonyAdapter
from harmonyagent.aime25.model import MessageList, SamplerBase, SamplerResponse


class HarmonySampler(SamplerBase):
    """Sampler from a Harmony compatible API."""

    def __init__(
        self,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        "Takes in a list of openai messages, returns response, appends reasoning trace."
        try:
            adapter = HarmonyAdapter()
            response = adapter.run(task=message_list[0]["content"])
            final_answer = ""
            for message in response:
                if message["channel"] == "final":
                    final_answer = message["content"][0]["text"]
                if message["channel"] in {"analysis", "commentary"}:
                    message_list.append(self._pack_message("assistant", message["content"][0]["text"]))

            return SamplerResponse(
                response_text=final_answer,
                response_metadata={"usage": None},
                actual_queried_message_list=message_list,
            )
        except Exception:
            traceback.print_exc()

            return SamplerResponse(
                response_text="",
                response_metadata={"usage": None},
                actual_queried_message_list=[],
            )
