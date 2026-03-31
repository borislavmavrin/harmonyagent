from dataclasses import asdict, dataclass
from typing import Any

from harmonyagent.models.global_stats import GLOBAL_MODEL_STATS


class PromptTextMismatch(Exception):
    """Raised when prompt text does not match the tape."""


@dataclass
class TapeModelConfig:
    tape: list[dict]
    model_name: str = "tape"
    tape_path: str = ""
    temperature: float = 1.0
    top_p: float = 1.0
    read_timeout: int = 60
    num_retrials: int = 10
    cost_per_call: float = 0.0


class TapeModel:
    def __init__(self, config: TapeModelConfig):
        """
        Initialize with a list of outputs to return in sequence.
        """
        self.config = config
        self.current_index = -1
        self.cost = 0.0
        self.n_calls = 0

    def query(self, prompt_text: str, stop_token_ids: list[str], max_tokens: int) -> dict:
        self.current_index += 1
        step = self.config.tape[self.current_index]
        if prompt_text != step["prompt_text"]:
            raise PromptTextMismatch
        self.n_calls += 1
        self.cost += self.config.cost_per_call
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return step["response"]

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
