import time
from pathlib import Path

from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    HarmonyError,
    Message,
    RenderConversationConfig,
    Role,
    SystemContent,
    load_harmony_encoding,
)

from harmonyagent.agents.exceptions.nonterminating_exceptions import (
    HarmonyErrorWithDetails as HarmonyParsingError,
)
from harmonyagent.agents.exceptions.nonterminating_exceptions import (
    HarmonyMessageMissingChannel,
)
from harmonyagent.agents.harmony_core.sys_dev_message import REASONING_EFFORT_MAP
from harmonyagent.models.vllm_raw import VllmRawModel, VllmRawModelConfig


class VllmRawLongContextTest:
    def __init__(
        self,
        reasoning_effort: str,
        drop_reasoning: bool,
        temperature: float,
        top_p: int,
    ):
        self.model = VllmRawModel(VllmRawModelConfig(temperature=temperature, top_p=top_p))
        self.encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.reasoning_effort = reasoning_effort
        self.harmony_config = RenderConversationConfig
        self.harmony_config.auto_drop_analysis = drop_reasoning
        self.temperature = temperature
        self.top_p = top_p

        self.stop_token_ids = self.encoder.stop_tokens_for_assistant_actions()  # self.encoder.stop_tokens()
        self.harmony_messages = [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_reasoning_effort(REASONING_EFFORT_MAP[reasoning_effort]),
            ),
        ]

    def add_task(
        self,
        task: str,
    ) -> tuple[str, list[int]]:
        self.harmony_messages.append(
            Message.from_role_and_content(
                Role.USER,
                task,
            )
        )
        convo = Conversation.from_messages(self.harmony_messages)
        tokens = self.encoder.render_conversation_for_completion(convo, Role.ASSISTANT, self.harmony_config)
        print(f"num_tokens: {len(tokens)}")
        text = self.encoder.decode(tokens)
        return text, tokens

    def text_to_harmony_messages(self, completion_text: str) -> tuple[list[dict], list[int]]:
        try:
            # leave for debugging
            completion_tokens = self.encoder.encode(completion_text, allowed_special="all")
            harmony_messages = self.encoder.parse_messages_from_completion_tokens(
                completion_tokens, Role.ASSISTANT, strict=False
            )
        except HarmonyError as e:
            raise HarmonyParsingError(str(e), completion_text)

        is_final = False
        for harmony_message in harmony_messages:
            channel = getattr(harmony_message, "channel", None)
            if not channel:
                raise HarmonyMessageMissingChannel(harmony_message)
            if channel == "final":
                is_final = True

        if not is_final:
            print("final message is missing.")
            log_path = Path("logs/harmony/") / f"{int(time.time())}.txt"
            log_path.write_text(completion_text)

        return harmony_messages, completion_tokens

    def __call__(
        self,
        task: str,
    ):
        prompt_text, prompt_tokens = self.add_task(task)
        start_time = time.time()
        max_context_size = 128 * 1_024
        response = self.model.query(
            prompt_text=prompt_text,
            stop_token_ids=self.stop_token_ids,
            max_tokens=max_context_size - len(prompt_tokens),
        )
        completion_text = response["choices"][0]["text"]
        new_harmony_messages, completion_tokens = self.text_to_harmony_messages(completion_text)
        print(f"{len(completion_tokens) / (time.time() - start_time): .2f} tokens/sec")
        self.harmony_messages.extend(new_harmony_messages)


if __name__ == "__main__":
    import json
    from pathlib import Path

    from datasets import load_dataset

    dataset_dir = Path("data")
    if not dataset_dir.is_dir():
        dataset_dir.mkdir(exist_ok=True, parents=True)
        squad_dataset = load_dataset("mbpp", "full", split="train")
        squad_dataset.to_json(dataset_dir / "mbpp_full_train.jsonl")

    vllm_raw_lon_context_test = VllmRawLongContextTest(
        reasoning_effort="low", drop_reasoning=True, temperature=1.0, top_p=1.0
    )
    mbpp_dataset = [json.loads(row) for row in open(Path(dataset_dir / "mbpp_full_train.jsonl")).readlines()]
    for row in mbpp_dataset:
        vllm_raw_lon_context_test(row["text"])
    # vllm init param: skip_tokenizer_init
