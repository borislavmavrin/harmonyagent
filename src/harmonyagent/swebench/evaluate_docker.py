"""Evaluate a SWE-bench instance inside a Docker container."""

import json
from pathlib import Path

from harmonyagent.domain_model import Environment
from harmonyagent.swebench.eval_script import make_eval_script_list_py
from harmonyagent.swebench.grading import get_eval_report


def evaluate_instance_in_docker(
    instance: dict,
    instance_id: str,
    exit_status: str,
    patch_diff: str,
    env: Environment,
    logs_dir: Path,
) -> dict:
    """Run SWE-bench evaluation for a single instance and return the report.

    Returns the report dict keyed by instance_id.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    if exit_status == "Submitted":
        eval_file = logs_dir / "eval.sh"
        eval_script = make_eval_script_list_py(instance)
        eval_file.write_text(eval_script)
        env.copy_to_container(eval_file, "/eval.sh")
        eval_result = env.execute("/bin/bash /eval.sh", timeout=60 * 15)
        test_output = eval_result["output"]
        test_output_path = logs_dir / "test_output.txt"
        with open(test_output_path, "w") as f:
            f.write(test_output)
        report = get_eval_report(
            instance=instance,
            prediction={
                "instance_id": instance_id,
                "model_name_or_path": "model_name_to_be_fixed",
                "model_patch": patch_diff,
            },
            test_log_path=test_output_path,
            include_tests_status=True,
        )
    else:
        report = {
            instance_id: {
                "patch_is_None": True,
                "patch_exists": False,
                "patch_successfully_applied": False,
                "resolved": False,
                "tests_status": {},
            }
        }

    json.dump(report[instance_id], open(logs_dir / "report.json", "w"), indent=4)
    return report
