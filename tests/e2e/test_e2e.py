from pathlib import Path

import yaml

from harmonyagent.agents.harmony_agent import AgentConfig, HarmonyAgent
from harmonyagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from harmonyagent.models.test_models import TapeModel, TapeModelConfig
from harmonyagent.utils.tape import read_tape


def test_successful_completion():
    """Test agent completes successfully when COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT is encountered."""

    # from datasets import load_dataset
    # dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
    instance_id = "django__django-14122"
    config_path = Path("src/harmonyagent/config/swebench_harmony.yaml")
    config = yaml.safe_load(config_path.read_text())
    agent_config = AgentConfig(**config["agent"])

    task = "Meta.ordering fields must not be included in GROUP BY clause\nDescription\n\t\nThis continues (closed) [1] ticket.\nI beleave it was not properly fixed in commit [0ddb4ebf].\nWhile commit [0ddb4ebf] removes ORDER BY when Meta.ordering is used it still does populates GROUP BY with Meta.ordering fields thus leads to wrong aggregation.\nPR with test case was added at [2].\n[1] https://code.djangoproject.com/ticket/14357\n[2] \u200b\u200bhttps://github.com/django/django/pull/14122\n"
    tape = read_tape(Path("tests/e2e/data/django__django-14122"))
    model = TapeModel(config=TapeModelConfig(tape=tape))
    env_config = config.get("environment", {})
    env_config.pop("environment_class", None)
    env = DockerEnvironment(
        config=DockerEnvironmentConfig(
            image=f"docker.io/swebench/sweb.eval.x86_64.django_1776_{instance_id.split('__')[1]}:latest",
            **env_config,
        )
    )
    env.copy_to_container("tools/bin/.", "/bin/")
    agent = HarmonyAgent(
        model=model,
        env=env,
        task_id=instance_id,
        config=agent_config,
    )

    exit_status, result = agent.run(task)
    assert exit_status == "Submitted"
    # Diff generation is now in the runner (swebench_harmony.py), not in the agent
    assert agent.model.n_calls == len(tape)
