import logging
import os
import shlex
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DockerEnvironmentConfig:
    image: str
    cwd: str = "/testbed"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(
        default_factory=lambda: {
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
        }
    )
    """Environment variables to set in the container."""
    environment_class: str = "docker"
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 300
    """Timeout for executing commands in the container."""
    executable: str = "docker"
    """Path to the docker/container executable."""
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    memory: str = "2g"
    cpus: str = "4"


class DockerEnvironment:
    def __init__(self, config: DockerEnvironmentConfig, logger: logging.Logger | None = None):
        """This class executes bash commands in a Docker container using direct docker commands."""
        self.logger = logger or logging.getLogger("harmonyagent.environment")
        self.container_id: str | None = None
        self.config = config
        self._start_container()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"harmonyagent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--memory",
            f"{self.config.memory}",
            "--memory-swappiness",
            "0",
            "--cpus",
            f"{self.config.cpus}",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        cwd = cwd or self.config.cwd
        effective_timeout = timeout or self.config.timeout

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        # Use timeout inside the container so the subprocess is actually killed on timeout,
        # not just the docker exec process on the host (which would leave zombie processes).
        # -k 5: send SIGKILL 5s after SIGTERM if the process hasn't exited yet.
        # Trap SIGTERM to kill the entire process group (`kill 0`), ensuring child
        # processes (e.g. python) are properly terminated instead of becoming zombies.
        cmd.extend(
            [
                self.container_id,
                "timeout",
                "-k",
                "5",
                str(effective_timeout),
                "bash",
                "-lc",
                f"trap 'kill 0' TERM; {command}",
            ]
        )

        result = subprocess.run(
            cmd,
            text=True,
            timeout=effective_timeout
            + 10,  # safety net if timeout command is unavailable - [ ] TODO: check when creating and image
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode == 124:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=effective_timeout, output=(result.stdout or "").encode())
        return {"output": result.stdout, "returncode": result.returncode}

    def copy_to_container(self, source_path: str, container_path: str):
        cmd = [
            self.config.executable,
            "cp",
            source_path,
            f"{self.container_id}:{container_path}",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        self.logger.info(f"{result.stdout}, {result.returncode}")
        self.logger.info(f"Copied {source_path} to {self.container_id}:{container_path}.")

    def cleanup(self):
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1"
            subprocess.run(cmd, shell=True)
            self.logger.info(f"{self.container_id} is stopped and removed.")

