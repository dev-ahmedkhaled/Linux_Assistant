# sandbox/docker_sandbox.py
# Runs commands safely inside a Docker container
# Agent never touches the real system directly

import docker
from dataclasses import dataclass
from typing import Optional
from config.settings import config

DESTRUCTIVE_PATTERNS = [
    "rm -rf /", "rm -rf ~", "mkfs", "dd if=",
    ":(){:|:&};:", "chmod -R 777 /", "> /dev/sda",
    "shutdown", "reboot", "halt", "poweroff",
]

@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    blocked: bool = False
    block_reason: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.blocked

    @property
    def has_error(self) -> bool:
        return not self.success

    def __str__(self):
        if self.blocked:
            return f"[BLOCKED] {self.block_reason}"
        if self.timed_out:
            return "[TIMED OUT]"
        out = []
        if self.stdout.strip():
            out.append(self.stdout.strip())
        if self.stderr.strip():
            out.append(f"[stderr] {self.stderr.strip()}")
        return "\n".join(out) or "(no output)"


class DockerSandbox:
    """
    Executes shell commands inside an isolated Docker container.
    Mounts the user's home directory at /workspace so the agent
    can actually see and work with real files.
    """

    def __init__(self):
        self.cfg = config.sandbox
        self._client = None
        self._session_container = None

    @property
    def client(self):
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _is_destructive(self, command: str) -> Optional[str]:
        cmd_lower = command.lower()
        for pattern in DESTRUCTIVE_PATTERNS:
            if pattern in cmd_lower:
                return f"Command matches destructive pattern: '{pattern}'"
        return None

    def run(self, command: str, workdir: str = "/workspace") -> CommandResult:
        """Run a single command in an isolated container"""
        block_reason = self._is_destructive(command)
        if block_reason and config.agent.safe_mode:
            return CommandResult(
                stdout="", stderr="", exit_code=1,
                blocked=True, block_reason=block_reason
            )

        try:
            container = self.client.containers.run(
                image=self.cfg.image,
                command=["bash", "-c", command],
                working_dir=workdir,
                mem_limit=self.cfg.memory_limit,
                network_disabled=self.cfg.network_disabled,
                volumes={
                    self.cfg.mount_path: {"bind": "/workspace", "mode": "rw"}
                },
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
                timeout=self.cfg.timeout,
            )
            stdout = container.decode("utf-8", errors="replace") if isinstance(container, bytes) else ""
            return CommandResult(stdout=stdout, stderr="", exit_code=0)

        except docker.errors.ContainerError as e:
            return CommandResult(
                stdout=e.stderr.decode("utf-8", errors="replace") if e.stderr else "",
                stderr=str(e),
                exit_code=e.exit_status,
            )
        except Exception as e:
            if "timeout" in str(e).lower():
                return CommandResult(stdout="", stderr=str(e), exit_code=1, timed_out=True)
            return CommandResult(stdout="", stderr=str(e), exit_code=1)

    def run_session(self, command: str, workdir: str = "/workspace") -> CommandResult:
        """
        Run command in a persistent session container.
        Mounts cfg.mount_path from host to /workspace inside container.
        """
        block_reason = self._is_destructive(command)
        if block_reason and config.agent.safe_mode:
            return CommandResult(
                stdout="", stderr="", exit_code=1,
                blocked=True, block_reason=block_reason
            )

        if self._session_container is None:
            self._session_container = self.client.containers.run(
                image=self.cfg.image,
                command="sleep infinity",
                mem_limit=self.cfg.memory_limit,
                network_disabled=self.cfg.network_disabled,
                volumes={
                    self.cfg.mount_path: {"bind": "/workspace", "mode": "rw"}
                },
                working_dir="/workspace",
                detach=True,
            )

        try:
            result = self._session_container.exec_run(
                cmd=["bash", "-c", command],
                workdir=workdir,
                demux=True,
            )
            stdout = (result.output[0] or b"").decode("utf-8", errors="replace")
            stderr = (result.output[1] or b"").decode("utf-8", errors="replace")
            return CommandResult(stdout=stdout, stderr=stderr, exit_code=result.exit_code)
        except Exception as e:
            return CommandResult(stdout="", stderr=str(e), exit_code=1)

    def close_session(self):
        if self._session_container:
            try:
                self._session_container.stop()
                self._session_container.remove()
            except Exception:
                pass
            self._session_container = None

    def is_available(self) -> bool:
        try:
            self.client.ping()
            return True
        except Exception:
            return False


# Module-level singleton
sandbox = DockerSandbox()