import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from tenacity import before_sleep_log, retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential_jitter

from harmonyagent.models.global_stats import GLOBAL_MODEL_STATS
from harmonyagent.utils.log import logger
from harmonyagent.utils.tape import record_step


@dataclass
class VllmRawModelConfig:
    model_name: str = "gpt-oss"
    tape_path: str = ""
    temperature: float = 1.0
    top_p: float = 1.0
    read_timeout: int = 60
    num_retrials: int = 10
    cost_per_call: float = 0.0


class VllmRawModel:
    def __init__(
        self,
        config: VllmRawModelConfig,
    ):
        self.config = config
        self.cost = 0.0
        self.n_calls = 0
        logger.info(f"temperature: {self.config.temperature} top_p: {self.config.top_p}")

    def _query(self, prompt_text: str, stop_token_ids: list[int], max_tokens: int) -> dict:
        @retry(
            stop=stop_after_attempt(self.config.num_retrials),
            wait=wait_exponential_jitter(initial=10, max=180),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            retry=retry_if_not_exception_type((KeyboardInterrupt,)),
        )
        def _do_query():
            url = "http://localhost:8000/v1/completions"
            payload = {
                "prompt": prompt_text,
                "skip_special_tokens": False,
                "stop_token_ids": stop_token_ids,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "min_p": 0.0,
                "top_k": 0,
            }
            response = requests.post(url, json=payload, stream=False, timeout=self.config.read_timeout)
            response.raise_for_status()
            return response.json()

        return _do_query()

    def query(self, prompt_text: str, stop_token_ids: list[str], max_tokens: int) -> str:
        response = self._query(prompt_text, stop_token_ids, max_tokens)
        if self.config.tape_path:
            record_step(Path(self.config.tape_path), self.n_calls, prompt_text, response)
        self.n_calls += 1
        cost = 0
        assert cost >= 0.0, f"Cost is negative: {cost}"
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        return response

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
