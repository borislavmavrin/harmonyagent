"""Integration tests for evaluate_instance_in_docker using real Docker containers.

These tests apply the gold patch to a real SWE-bench container and verify
that the evaluation pipeline correctly reports the instance as resolved.

Requires: Docker available and SWE-bench images pulled locally.
"""

import subprocess

import pytest

from harmonyagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from harmonyagent.run.swebench_harmony import get_swebench_docker_image_name
from harmonyagent.swebench.evaluate_docker import evaluate_instance_in_docker

from .conftest import BASH_ONLY_20


def is_docker_available():
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_docker_available(), reason="Docker not available"),
]


@pytest.mark.slow
@pytest.mark.parametrize("instance_id", BASH_ONLY_20)
def test_gold_patch_resolves(swebench_instances, tmp_path, instance_id):
    """Apply the gold patch and verify the instance evaluates as resolved."""
    instance = swebench_instances[instance_id]
    image_name = get_swebench_docker_image_name(instance)
    config = DockerEnvironmentConfig(image=image_name, timeout=300, cpus=12)
    env = DockerEnvironment(config=config)

    try:
        # Apply gold patch
        gold_patch = instance["patch"]
        apply_result = env.execute(f"git apply - <<'PATCH_EOF'\n{gold_patch}\nPATCH_EOF")
        assert apply_result["returncode"] == 0, f"Failed to apply gold patch:\n{apply_result['output']}"

        # Get the diff as the agent would produce it
        diff_result = env.execute("git diff")
        assert diff_result["returncode"] == 0
        patch_diff = diff_result["output"]

        logs_dir = tmp_path / "logs" / instance_id
        report = evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Submitted",
            patch_diff=patch_diff,
            env=env,
            logs_dir=logs_dir,
        )

        assert report[instance_id]["resolved"] is True, (
            f"{instance_id} not resolved with gold patch. Report: {report[instance_id]}"
        )
        assert report[instance_id]["patch_exists"] is True
        assert report[instance_id]["patch_successfully_applied"] is True
        assert (logs_dir / "report.json").exists()
        assert (logs_dir / "test_output.txt").exists()
        assert (logs_dir / "eval.sh").exists()
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("instance_id", BASH_ONLY_20)
def test_no_patch_not_resolved(swebench_instances, tmp_path, instance_id):
    """Without applying a patch, the instance should not resolve."""
    instance = swebench_instances[instance_id]
    image_name = get_swebench_docker_image_name(instance)
    config = DockerEnvironmentConfig(image=image_name, timeout=300)
    env = DockerEnvironment(config=config)

    try:
        # Get diff without applying any patch (should be empty)
        diff_result = env.execute("git diff")
        assert diff_result["returncode"] == 0
        patch_diff = diff_result["output"]

        logs_dir = tmp_path / "logs" / instance_id
        report = evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Submitted",
            patch_diff=patch_diff,
            env=env,
            logs_dir=logs_dir,
        )

        assert report[instance_id]["resolved"] is False
    finally:
        env.cleanup()
