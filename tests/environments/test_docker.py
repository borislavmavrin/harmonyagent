import os
import subprocess
from unittest.mock import patch

import pytest

from harmonyagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig


def is_docker_available():
    """Check if Docker is available and running."""
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Test parameters for both Docker and Podman
environment_params = [
    pytest.param(
        "docker",
        marks=pytest.mark.skipif(not is_docker_available(), reason="Docker not available"),
        id="docker",
    )
]


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_config_defaults(executable):
    """Test that DockerEnvironmentConfig has correct default values."""
    config = DockerEnvironmentConfig(image="python:3.11", executable=executable)

    assert config.image == "python:3.11"
    assert config.cwd == "/testbed"
    assert config.env == {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
    }
    assert config.forward_env == []
    assert config.timeout == 300
    assert config.executable == executable


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_basic_execution(executable):
    """Test basic command execution in Docker container."""
    env = DockerEnvironment(config=DockerEnvironmentConfig(image="python:3.11", executable=executable))

    try:
        result = env.execute("echo 'hello world'")
        assert result["returncode"] == 0
        assert "hello world" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_set_env_variables(executable):
    """Test setting environment variables in the container."""
    env = DockerEnvironment(
        config=DockerEnvironmentConfig(
            image="python:3.11", executable=executable, env={"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"}
        )
    )

    try:
        # Test single environment variable
        result = env.execute("echo $TEST_VAR")
        assert result["returncode"] == 0
        assert "test_value" in result["output"]

        # Test multiple environment variables
        result = env.execute("echo $TEST_VAR $ANOTHER_VAR")
        assert result["returncode"] == 0
        assert "test_value another_value" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_forward_env_variables(executable):
    """Test forwarding environment variables from host to container."""
    with patch.dict(os.environ, {"HOST_VAR": "host_value", "ANOTHER_HOST_VAR": "another_host_value"}):
        env = DockerEnvironment(
            config=DockerEnvironmentConfig(
                image="python:3.11", executable=executable, forward_env=["HOST_VAR", "ANOTHER_HOST_VAR"]
            )
        )

        try:
            # Test single forwarded environment variable
            result = env.execute("echo $HOST_VAR")
            assert result["returncode"] == 0
            assert "host_value" in result["output"]

            # Test multiple forwarded environment variables
            result = env.execute("echo $HOST_VAR $ANOTHER_HOST_VAR")
            assert result["returncode"] == 0
            assert "host_value another_host_value" in result["output"]
        finally:
            env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_forward_nonexistent_env_variables(executable):
    """Test forwarding non-existent environment variables (should be empty)."""
    env = DockerEnvironment(
        config=DockerEnvironmentConfig(image="python:3.11", executable=executable, forward_env=["NONEXISTENT_VAR"])
    )

    try:
        result = env.execute('echo "[$NONEXISTENT_VAR]"')
        assert result["returncode"] == 0
        assert "[]" in result["output"]  # Empty variable should result in empty string
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_combined_env_and_forward(executable):
    """Test both setting and forwarding environment variables together."""
    with patch.dict(os.environ, {"HOST_VAR": "from_host"}):
        env = DockerEnvironment(
            config=DockerEnvironmentConfig(
                image="python:3.11", executable=executable, env={"SET_VAR": "from_config"}, forward_env=["HOST_VAR"]
            )
        )

        try:
            result = env.execute("echo $SET_VAR $HOST_VAR")
            assert result["returncode"] == 0
            assert "from_config from_host" in result["output"]
        finally:
            env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_env_override_forward(executable):
    """Test that explicitly set env variables take precedence over forwarded ones."""
    with patch.dict(os.environ, {"CONFLICT_VAR": "from_host"}):
        env = DockerEnvironment(
            config=DockerEnvironmentConfig(
                image="python:3.11",
                executable=executable,
                env={"CONFLICT_VAR": "from_config"},
                forward_env=["CONFLICT_VAR"],
            )
        )

        try:
            result = env.execute("echo $CONFLICT_VAR")
            assert result["returncode"] == 0
            # The explicitly set env should take precedence (comes first in docker exec command)
            assert "from_config" in result["output"]
        finally:
            env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_custom_cwd(executable):
    """Test executing commands in a custom working directory."""
    env = DockerEnvironment(config=DockerEnvironmentConfig(image="python:3.11", executable=executable, cwd="/tmp"))

    try:
        result = env.execute("pwd")
        assert result["returncode"] == 0
        assert "/tmp" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_cwd_parameter_override(executable):
    """Test that the cwd parameter in execute() overrides the config cwd."""
    env = DockerEnvironment(config=DockerEnvironmentConfig(image="python:3.11", executable=executable, cwd="/"))

    try:
        result = env.execute("pwd", cwd="/tmp")
        assert result["returncode"] == 0
        assert "/tmp" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_command_failure(executable):
    """Test that command failures are properly captured."""
    env = DockerEnvironment(config=DockerEnvironmentConfig(image="python:3.11", executable=executable))

    try:
        result = env.execute("bash -c 'exit 42'")
        assert result["returncode"] == 42
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_shell_builtin(executable):
    """Test that shell builtins work when wrapped with bash -lc."""
    env = DockerEnvironment(config=DockerEnvironmentConfig(image="python:3.11", executable=executable))

    try:
        result = env.execute("bash -lc 'exit 42'")
        assert result["returncode"] == 42
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_pipe(executable):
    """Test that piped commands work when wrapped with bash -lc."""
    env = DockerEnvironment(config=DockerEnvironmentConfig(image="python:3.11", executable=executable))

    try:
        result = env.execute("bash -lc 'echo hello world | grep world'")
        assert result["returncode"] == 0
        assert "hello world" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_compound_commands(executable):
    """Test that compound commands (&&, ||, ;) work when wrapped with bash -lc."""
    env = DockerEnvironment(config=DockerEnvironmentConfig(image="python:3.11", executable=executable))

    try:
        # Test && (both should execute)
        result = env.execute("bash -lc 'echo first && echo second'")
        assert result["returncode"] == 0
        assert "first" in result["output"]
        assert "second" in result["output"]

        # Test || (only first should execute since it succeeds)
        result = env.execute("bash -lc 'echo ok || echo fallback'")
        assert result["returncode"] == 0
        assert "ok" in result["output"]
        assert "fallback" not in result["output"]

        # Test ; (both should execute regardless)
        result = env.execute("bash -lc 'echo aaa; echo bbb'")
        assert result["returncode"] == 0
        assert "aaa" in result["output"]
        assert "bbb" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_subshell(executable):
    """Test that subshell commands work when wrapped with bash -lc."""
    env = DockerEnvironment(config=DockerEnvironmentConfig(image="python:3.11", executable=executable))

    try:
        result = env.execute("bash -lc 'echo count=$(echo 42)'")
        assert result["returncode"] == 0
        assert "count=42" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_custom_container_timeout(executable):
    """Test that custom container_timeout is respected."""
    import time

    env = DockerEnvironment(
        config=DockerEnvironmentConfig(image="python:3.11", executable=executable, container_timeout="3s")
    )

    try:
        result = env.execute("echo 'container is running'")
        assert result["returncode"] == 0
        assert "container is running" in result["output"]
        time.sleep(5)
        with pytest.raises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            # This command should fail because the container has stopped
            subprocess.run(
                [executable, "exec", env.container_id, "echo", "still running"],
                check=True,
                capture_output=True,
                timeout=2,
            )
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_timeout_kills_container_process(executable):
    """Test that timeout kills the subprocess inside the container, not just docker exec on host.

    Regression: previously subprocess.run timeout only killed the docker exec host process,
    leaving the container subprocess running as a zombie at ~100% CPU indefinitely.

    Real-world case (2026-02-26): the model submitted this command inside a SWE-bench container:

        python - << 'PY'
        from sympy import symbols, I
        from sympy.polys.polytools import Poly
        x, y = symbols('x y')
        poly = Poly(y-1, x, y, domain=Poly('1', x, y).domain.algebraic_field(I))
        print(poly.factor_list())
        PY

    docker exec timed out on the host after 300s, but the python process inside the container
    kept running for 27+ minutes at 99.8% CPU, eventually starving the vLLM server and causing
    all 4 worker threads to stall waiting for model responses.
    """
    import time

    env = DockerEnvironment(
        config=DockerEnvironmentConfig(
            image="docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-19346:latest",
            executable=executable,
        )
    )

    # Exact command from the logs that caused the zombie — uses heredoc like the real agent call
    sympy_command = """\
python3 - << 'PY'
from sympy import symbols, I
from sympy.polys.polytools import Poly
x, y = symbols('x y')
poly = Poly(y-1, x, y, domain=Poly('1', x, y).domain.algebraic_field(I))
print(poly.factor_list())
PY"""

    try:
        # Run the actual command from the logs with a short timeout
        with pytest.raises(subprocess.TimeoutExpired):
            env.execute(sympy_command, timeout=5)

        time.sleep(0.5)  # let signal propagate

        # Verify the python3 process is actually dead inside the container, not orphaned.
        # pgrep exits 1 when no matching processes are found.
        result = subprocess.run(
            [executable, "exec", env.container_id, "pgrep", "-af", "python3"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode != 0, (
            f"Zombie python3 process still running inside container after timeout:\n{result.stdout}"
        )
    finally:
        env.cleanup()


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_oom_exception(executable):
    """Test OOM execution in Docker container."""
    # Note: This test assumes the container or environment enforces a memory limit,
    # otherwise it might freeze the host or take a long time to be killed.
    env = DockerEnvironment(config=DockerEnvironmentConfig(image="python:3.11", executable=executable))

    try:
        # Attempt to consume unlimited memory
        result = env.execute("python3 -c \"[' ' * 1024 * 1024 for _ in iter(int, 1)]\"")
        # Exit code 137 usually indicates SIGKILL (9) + 128, often due to OOM killer
        assert result["returncode"] == 137
    finally:
        env.cleanup()
