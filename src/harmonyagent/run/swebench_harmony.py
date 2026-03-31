#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

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
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from rich.live import Live

from harmonyagent.agents.harmony_agent import AgentConfig, HarmonyAgent
from harmonyagent.config.utils import config_dir
from harmonyagent.domain_model import Environment
from harmonyagent.environments.utils import get_environment_class, get_environment_class_config_class
from harmonyagent.models.utils import get_model_class, get_model_class_config
from harmonyagent.run.batch_progress import RunBatchProgressManager
from harmonyagent.run.save import save_traj
from harmonyagent.swebench.evaluate_docker import evaluate_instance_in_docker
from harmonyagent.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(HarmonyAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

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


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + image_name
    env_class_name = env_config.pop("environment_class")
    EnvClass = get_environment_class(env_class_name)
    ConfigClass = get_environment_class_config_class(env_class_name)
    env = EnvClass(config=ConfigClass(**env_config))
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
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


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> str:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    trajs_dir = output_dir / "trajs" / instance_id
    trajs_dir.mkdir(parents=True, exist_ok=True)
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (trajs_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model_config = config.get("model", {}).copy()
    model_class_name = model_config.pop("model_class", "")
    ModelClass = get_model_class(model_class_name)
    ConfigClass = get_model_class_config(model_class_name)
    valid_fields = {f.name for f in dataclasses.fields(ConfigClass)}
    cfg_kwargs = {k: v for k, v in model_config.items() if k in valid_fields}
    model = ModelClass(config=ConfigClass(**cfg_kwargs))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent = None
    env = None
    extra_info = None
    exit_status = ""
    resolution_status = ""

    try:
        env = get_sb_environment(config, instance)
        env.copy_to_container("tools/bin/.", "/bin/")
        env.execute("apt update && apt install -y ripgrep tree")
        agent_config = AgentConfig(**config.get("agent", {}))
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            config=agent_config,
        )
        exit_status, last_messages = agent.run(task)
        patch_diff_result = env.execute(
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached", timeout=60 * 15
        )
        if patch_diff_result["returncode"] != 0:
            RuntimeError("evaluation failed")
        patch_diff = patch_diff_result["output"]
        logger.info(f"{instance_id} patch diff: {repr(patch_diff)}")
        logs_dir = output_dir / "logs" / instance_id
        report = evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status=exit_status,
            patch_diff=patch_diff,
            env=env,
            logs_dir=logs_dir,
        )
        resolution_status = "Resolved" if report[instance_id]["resolved"] else "Not Resolved"
        logger.info(f"{instance_id} resolution_status: {resolution_status}")
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, patch_diff = type(e).__name__, None
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            trajs_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=patch_diff,
            extra_info=extra_info,
            print_fct=logger.info,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, patch_diff)

        if env:
            env.cleanup()
        if resolution_status == "Resolved":
            progress_manager.on_instance_end(instance_id, resolution_status)
        else:
            progress_manager.on_instance_end(instance_id, exit_status)

    return resolution_status


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
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
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    instance_names_from_file: str = typer.Option("", "--instance-names-from-file", help="Load instance names from file.", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'harmonyagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_path: Path = typer.Option( config_dir / "swebench_harmony.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> list[str]:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "harmonyagent.log")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))
    
    if instance_names_from_file:
        instance_ids_from_file = [row.strip() for row in open(instance_names_from_file).readlines()]
        instances_map = {instance["instance_id"]: instance for instance in instances}
        instances = [instances_map[instance_id] for instance_id in instance_ids_from_file]
    
    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    config_path = Path(config_path)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        resolution_statuses = []
        for future in concurrent.futures.as_completed(futures):
            try:
                resolution_status = future.result()
                resolution_statuses.append(resolution_status)
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)
        return resolution_statuses

    with Live(progress_manager.render_group, refresh_per_second=4):
        resolution_statuses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            try:
                resolution_statuses = process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                resolution_statuses = process_futures(futures)
    return resolution_statuses
