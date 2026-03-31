#!/usr/bin/env python3

"""Run mini-SWE-agent on AIME 2025 math problems in batch mode."""

import concurrent.futures
import dataclasses
import json
import random
import re
import threading
import time
import traceback
from pathlib import Path

import typer
import yaml
from rich.live import Live

from harmonyagent.agents.harmony_agent import AgentConfig, HarmonyAgent
from harmonyagent.aime25.aime_eval import extract_boxed_text, load_aime25_dataset
from harmonyagent.config.utils import config_dir
from harmonyagent.domain_model import Environment
from harmonyagent.environments.utils import get_environment_class, get_environment_class_config_class
from harmonyagent.models.utils import get_model_class, get_model_class_config
from harmonyagent.run.batch_progress import RunBatchProgressManager
from harmonyagent.run.save import save_traj
from harmonyagent.utils.log import add_file_handler, logger

PYTHON_PACKAGES = "sympy scikit-learn shapely matplotlib networkx fraction mpmath"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(HarmonyAgent):
    """Simple wrapper around HarmonyAgent that provides progress updates."""

    def __init__(self, model, env, progress_manager: RunBatchProgressManager, instance_id: str = "", config=None):
        super().__init__(model, env, instance_id, config=config)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def create_shared_environment(config: dict) -> Environment:
    """Create a single shared Docker environment for all instances."""
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    env_class_name = env_config.pop("environment_class")
    EnvClass = get_environment_class(env_class_name)
    ConfigClass = get_environment_class_config_class(env_class_name)
    env = EnvClass(config=ConfigClass(**env_config))
    result = env.execute(f"pip install {PYTHON_PACKAGES}", timeout=120)
    if result["returncode"] != 0:
        raise RuntimeError(f"Failed to install Python packages: {result}")
    logger.info(f"pip install return code: {result['returncode']}")
    return env


def update_preds_file(
    output_path: Path,
    instance_id: str,
    model_name: str,
    extracted_answer,
    correct_answer,
    score: float,
    exit_status: str,
):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "extracted_answer": extracted_answer,
            "correct_answer": correct_answer,
            "score": score,
            "exit_status": exit_status,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def extract_answer_from_messages(result_str: str) -> int | None:
    """Extract the agent's answer from the Submitted result JSON.

    Parses the JSON message list and searches all text content for \\boxed{}.
    """
    try:
        messages = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        content = msg if isinstance(msg, str) else msg.get("content", "")
        # content can be a string or a list of content blocks like [{"type": "text", "text": "..."}]
        texts = []
        if isinstance(content, str):
            texts = [content]
        elif isinstance(content, list):
            texts = [block.get("text", "") for block in content if isinstance(block, dict)]
        for text in texts:
            extracted = extract_boxed_text(text)
            if extracted:
                try:
                    return int(extracted)
                except (ValueError, TypeError):
                    pass
    return None


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    env: Environment,
) -> float:
    """Process a single AIME instance."""
    instance_id = instance["instance_id"]
    correct_answer = instance["answer"]
    trajs_dir = output_dir / "trajs" / instance_id
    trajs_dir.mkdir(parents=True, exist_ok=True)
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (trajs_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    model_config = config.get("model", {}).copy()
    model_class_name = model_config.pop("model_class", "")
    ModelClass = get_model_class(model_class_name)
    ConfigClass = get_model_class_config(model_class_name)
    valid_fields = {f.name for f in dataclasses.fields(ConfigClass)}
    cfg_kwargs = {k: v for k, v in model_config.items() if k in valid_fields}
    model = ModelClass(config=ConfigClass(**cfg_kwargs))

    task = instance["question"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Starting")

    agent = None
    extra_info = None
    exit_status = ""
    score = 0.0
    extracted_answer = None

    try:
        env.execute(f"mkdir -p /testbed/{instance_id}")
        agent_config = AgentConfig(**config.get("agent", {}))
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            config=agent_config,
        )
        exit_status, result_str = agent.run(task)
        extracted_answer = extract_answer_from_messages(result_str)
        score = 1.0 if extracted_answer == correct_answer else 0.0
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status = type(e).__name__
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            trajs_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=json.dumps({"extracted_answer": extracted_answer, "correct_answer": correct_answer, "score": score}),
            extra_info=extra_info,
            print_fct=logger.info,
        )
        update_preds_file(
            output_dir / "preds.json",
            instance_id,
            model.config.model_name,
            extracted_answer,
            correct_answer,
            score,
            exit_status,
        )
        result_label = "Correct" if score == 1.0 else "Wrong"
        logger.info(f"{instance_id}: {result_label} (got={extracted_answer}, expected={correct_answer})")
        progress_manager.on_instance_end(instance_id, result_label)

    return score


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command()
def main(
    n_repeats: int = typer.Option(1, "--n-repeats", help="Number of times to repeat each instance", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_path: Path = typer.Option(config_dir / "aime25_harmony.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
):
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "harmonyagent.log")

    logger.info("Loading AIME 2025 dataset...")
    instances = load_aime25_dataset()
    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)

    if n_repeats > 1:
        repeated = []
        for r in range(n_repeats):
            for inst in instances:
                repeated.append({**inst, "instance_id": f"{inst['instance_id']}_r{r}"})
        instances = repeated
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    config_path = Path(config_path)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    env = None
    try:
        env = create_shared_environment(config)

        def process_futures(futures: dict[concurrent.futures.Future, str]):
            scores = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    score = future.result()
                    scores.append(score)
                except concurrent.futures.CancelledError:
                    pass
                except Exception as e:
                    instance_id = futures[future]
                    logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                    progress_manager.on_uncaught_exception(instance_id, e)
            return scores

        with Live(progress_manager.render_group, refresh_per_second=4):
            scores = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_instance, instance, output_path, config, progress_manager, env): instance[
                        "instance_id"
                    ]
                    for instance in instances
                }
                try:
                    scores = process_futures(futures)
                except KeyboardInterrupt:
                    logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                    for future in futures:
                        if not future.running() and not future.done():
                            future.cancel()
                    scores = process_futures(futures)

        n_correct = sum(1 for s in scores if s == 1.0)
        total = len(scores)
        pct = (n_correct / total * 100) if total > 0 else 0.0
        print(f"{n_correct}/{total} correct ({pct:.1f}%)")
    finally:
        if env:
            env.cleanup()


if __name__ == "__main__":
    app()
