"""Tests for evaluate_instance_in_docker using real SWE-bench Verified data."""

from unittest.mock import MagicMock

import pytest

from harmonyagent.swebench.evaluate_docker import evaluate_instance_in_docker

from .conftest import BASH_ONLY_20


def _make_mock_env():
    env = MagicMock()
    env.execute.return_value = {"output": "", "returncode": 0}
    env.copy_to_container = MagicMock()
    return env


class TestNonSubmittedStatus:
    """When exit_status != 'Submitted', no docker interaction should happen."""

    def test_not_submitted_returns_not_resolved(self, swebench_instances, tmp_path):
        instance_id = "django__django-14122"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "logs" / instance_id

        report = evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Failed",
            patch_diff="",
            env=env,
            logs_dir=logs_dir,
        )

        assert instance_id in report
        assert report[instance_id]["resolved"] is False
        assert report[instance_id]["patch_is_None"] is True
        assert report[instance_id]["patch_exists"] is False
        assert report[instance_id]["patch_successfully_applied"] is False
        assert report[instance_id]["tests_status"] == {}

    def test_not_submitted_does_not_call_env(self, swebench_instances, tmp_path):
        instance_id = "django__django-14122"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "logs" / instance_id

        evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="MaxStepsExceeded",
            patch_diff="some diff",
            env=env,
            logs_dir=logs_dir,
        )

        env.execute.assert_not_called()
        env.copy_to_container.assert_not_called()

    def test_not_submitted_writes_report_json(self, swebench_instances, tmp_path):
        instance_id = "sympy__sympy-18199"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "logs" / instance_id

        evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Error",
            patch_diff="",
            env=env,
            logs_dir=logs_dir,
        )

        report_file = logs_dir / "report.json"
        assert report_file.exists()

    def test_not_submitted_creates_logs_dir(self, swebench_instances, tmp_path):
        instance_id = "pytest-dev__pytest-7432"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "deep" / "nested" / "logs"

        evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Timeout",
            patch_diff="",
            env=env,
            logs_dir=logs_dir,
        )

        assert logs_dir.exists()

    @pytest.mark.parametrize("instance_id", BASH_ONLY_20)
    def test_not_submitted_all_instances(self, swebench_instances, tmp_path, instance_id):
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "logs" / instance_id

        report = evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Failed",
            patch_diff="",
            env=env,
            logs_dir=logs_dir,
        )

        assert report[instance_id]["resolved"] is False


class TestSubmittedStatus:
    """When exit_status == 'Submitted', the function should interact with the env."""

    def test_submitted_calls_copy_and_execute(self, swebench_instances, tmp_path):
        instance_id = "django__django-14122"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "logs" / instance_id

        evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Submitted",
            patch_diff="diff --git a/foo b/foo",
            env=env,
            logs_dir=logs_dir,
        )

        env.copy_to_container.assert_called_once()
        env.execute.assert_called_once_with("/bin/bash /eval.sh", timeout=60 * 15)

    def test_submitted_writes_eval_script(self, swebench_instances, tmp_path):
        instance_id = "django__django-14122"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "logs" / instance_id

        evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Submitted",
            patch_diff="diff --git a/foo b/foo",
            env=env,
            logs_dir=logs_dir,
        )

        eval_file = logs_dir / "eval.sh"
        assert eval_file.exists()
        content = eval_file.read_text()
        assert content.startswith("#!/bin/bash")
        assert "conda activate" in content

    def test_submitted_writes_test_output(self, swebench_instances, tmp_path):
        instance_id = "django__django-13512"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        env.execute.return_value = {"output": "test output from container"}
        logs_dir = tmp_path / "logs" / instance_id

        evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Submitted",
            patch_diff="diff --git a/foo b/foo",
            env=env,
            logs_dir=logs_dir,
        )

        test_output_path = logs_dir / "test_output.txt"
        assert test_output_path.exists()
        assert test_output_path.read_text() == "test output from container"

    def test_submitted_empty_output_not_resolved(self, swebench_instances, tmp_path):
        """With empty test output, the patch is not considered resolved."""
        instance_id = "django__django-14122"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        env.execute.return_value = {"output": ""}
        logs_dir = tmp_path / "logs" / instance_id

        report = evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Submitted",
            patch_diff="diff --git a/foo b/foo",
            env=env,
            logs_dir=logs_dir,
        )

        assert report[instance_id]["resolved"] is False

    def test_submitted_writes_report_json(self, swebench_instances, tmp_path):
        instance_id = "django__django-14349"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "logs" / instance_id

        evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Submitted",
            patch_diff="diff --git a/foo b/foo",
            env=env,
            logs_dir=logs_dir,
        )

        report_file = logs_dir / "report.json"
        assert report_file.exists()

    @pytest.mark.parametrize("instance_id", BASH_ONLY_20)
    def test_submitted_generates_eval_script_for_all(self, swebench_instances, tmp_path, instance_id):
        """Verify eval script generation works for all bash-only-20 instances."""
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "logs" / instance_id

        evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Submitted",
            patch_diff="diff --git a/foo b/foo",
            env=env,
            logs_dir=logs_dir,
        )

        eval_file = logs_dir / "eval.sh"
        assert eval_file.exists()
        content = eval_file.read_text()
        assert "#!/bin/bash" in content
        assert instance["base_commit"] in content


class TestReportStructure:
    """Verify the report dict structure for various scenarios."""

    def test_report_keyed_by_instance_id(self, swebench_instances, tmp_path):
        instance_id = "django__django-14725"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "logs" / instance_id

        report = evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Failed",
            patch_diff="",
            env=env,
            logs_dir=logs_dir,
        )

        assert list(report.keys()) == [instance_id]

    def test_submitted_report_has_expected_keys(self, swebench_instances, tmp_path):
        instance_id = "django__django-14771"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "logs" / instance_id

        report = evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Submitted",
            patch_diff="diff --git a/foo b/foo",
            env=env,
            logs_dir=logs_dir,
        )

        entry = report[instance_id]
        assert "patch_is_None" in entry
        assert "patch_exists" in entry
        assert "patch_successfully_applied" in entry
        assert "resolved" in entry

    def test_not_submitted_report_has_tests_status(self, swebench_instances, tmp_path):
        instance_id = "sympy__sympy-19040"
        instance = swebench_instances[instance_id]
        env = _make_mock_env()
        logs_dir = tmp_path / "logs" / instance_id

        report = evaluate_instance_in_docker(
            instance=instance,
            instance_id=instance_id,
            exit_status="Failed",
            patch_diff="",
            env=env,
            logs_dir=logs_dir,
        )

        assert "tests_status" in report[instance_id]
