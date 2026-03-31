"""Helper to re-record the e2e tape for django__django-14122 using the real vLLM model."""

import shutil
from pathlib import Path

import yaml

from harmonyagent.agents.harmony_agent import AgentConfig, HarmonyAgent
from harmonyagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from harmonyagent.models.vllm_raw import VllmRawModel, VllmRawModelConfig

INSTANCE_ID = "django__django-14122"
TASK = (
    "Meta.ordering fields must not be included in GROUP BY clause\n"
    "Description\n"
    "\t\n"
    "This continues (closed) [1] ticket.\n"
    "I beleave it was not properly fixed in commit [0ddb4ebf].\n"
    "While commit [0ddb4ebf] removes ORDER BY when Meta.ordering is used "
    "it still does populates GROUP BY with Meta.ordering fields thus leads to wrong aggregation.\n"
    "PR with test case was added at [2].\n"
    "[1] https://code.djangoproject.com/ticket/14357\n"
    "[2] \u200b\u200bhttps://github.com/django/django/pull/14122\n"
)
CONFIG_PATH = Path("src/harmonyagent/config/swebench_harmony.yaml")
TAPE_DIR = Path("tests/e2e/data/django__django-14122")


def record_tape():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    agent_config = AgentConfig(**config["agent"])

    if TAPE_DIR.exists():
        shutil.rmtree(TAPE_DIR)
    TAPE_DIR.mkdir(parents=True)

    model_config = VllmRawModelConfig(**config["model"])
    model_config.tape_path = str(TAPE_DIR)
    model_config.read_timeout = 90
    model = VllmRawModel(config=model_config)

    env_config = config.get("environment", {})
    env_config.pop("environment_class", None)
    env = DockerEnvironment(
        config=DockerEnvironmentConfig(
            image=f"docker.io/swebench/sweb.eval.x86_64.django_1776_{INSTANCE_ID.split('__')[1]}:latest",
            **env_config,
        )
    )
    env.copy_to_container("tools/bin/.", "/bin/")

    agent = HarmonyAgent(model=model, env=env, task_id=INSTANCE_ID, config=agent_config)
    exit_status, result = agent.run(TASK)

    print(f"exit_status: {exit_status}")
    print(f"n_calls: {model.n_calls}")

    print(f"Tape recorded to {TAPE_DIR} ({model.n_calls} steps, prompt_text files removed)")


if __name__ == "__main__":
    record_tape()
