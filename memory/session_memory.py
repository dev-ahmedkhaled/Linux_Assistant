# memory/session_memory.py
# Tracks what happened during the current session
# Gives the agent context about previous commands, cwd, errors

import json
import os
import psutil
import subprocess
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from config.settings import config


@dataclass
class CommandEntry:
    command: str
    stdout: str
    stderr: str
    exit_code: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    explanation: str = ""


@dataclass
class SystemInfo:
    os: str = ""
    distro: str = ""
    kernel: str = ""
    shell: str = ""
    user: str = ""
    hostname: str = ""
    cwd: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0.0
    gpu: str = ""

    @classmethod
    def collect(cls) -> "SystemInfo":
        """Gather system info at session start"""
        import platform

        def run(cmd): 
            try:
                return subprocess.getoutput(cmd).strip()
            except Exception:
                return ""

        distro = run("cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '\"'")
        gpu = run("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A'")

        return cls(
            os=platform.system(),
            distro=distro or platform.platform(),
            kernel=platform.release(),
            shell=os.environ.get("SHELL", "bash"),
            user=run("whoami"),
            hostname=run("hostname"),
            cwd=os.getcwd(),
            cpu_cores=psutil.cpu_count(),
            ram_gb=round(psutil.virtual_memory().total / 1e9, 1),
            gpu=gpu,
        )

    def to_prompt_string(self) -> str:
        return (
            f"OS: {self.distro} | Kernel: {self.kernel} | "
            f"Shell: {self.shell} | User: {self.user}@{self.hostname} | "
            f"CPU: {self.cpu_cores} cores | RAM: {self.ram_gb}GB | GPU: {self.gpu}"
        )


class SessionMemory:
    """
    Tracks commands, errors, and context for the current session.
    Also persists history across sessions to a JSON file.
    """

    def __init__(self):
        self.system_info: SystemInfo = SystemInfo.collect()
        self.cwd: str = "/workspace"
        self.command_history: list[CommandEntry] = []
        self.error_history: list[CommandEntry] = []
        self.session_start: str = datetime.now().isoformat()
        self._load_history()

    def add_command(self, command: str, stdout: str, stderr: str,
                    exit_code: int, explanation: str = "") -> CommandEntry:
        entry = CommandEntry(
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            explanation=explanation,
        )
        self.command_history.append(entry)
        if exit_code != 0:
            self.error_history.append(entry)
        return entry

    def update_cwd(self, new_cwd: str):
        self.cwd = new_cwd
        self.system_info.cwd = new_cwd

    def get_recent_commands(self, n: int = 5) -> list[CommandEntry]:
        return self.command_history[-n:]

    def get_context_summary(self) -> str:
        """Returns a summary string to inject into the agent's prompt"""
        lines = [
            f"System: {self.system_info.to_prompt_string()}",
            f"Current directory: {self.cwd}",
        ]
        recent = self.get_recent_commands(3)
        if recent:
            lines.append("Recent commands:")
            for entry in recent:
                status = "✓" if entry.exit_code == 0 else "✗"
                lines.append(f"  {status} {entry.command}")
        return "\n".join(lines)

    def _load_history(self):
        """Load persistent history from file"""
        try:
            if os.path.exists(config.history_file):
                with open(config.history_file) as f:
                    data = json.load(f)
                    # just load last 50 entries as context
                    for entry in data[-50:]:
                        self.command_history.append(CommandEntry(**entry))
        except Exception:
            pass

    def save_history(self):
        """Persist session history to file"""
        try:
            existing = []
            if os.path.exists(config.history_file):
                with open(config.history_file) as f:
                    existing = json.load(f)
            existing.extend([asdict(e) for e in self.command_history])
            with open(config.history_file, "w") as f:
                json.dump(existing[-500:], f, indent=2)  # keep last 500
        except Exception:
            pass

    def clear(self):
        self.command_history = []
        self.error_history = []


# Module-level singleton
session = SessionMemory()