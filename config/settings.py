# config/settings.py
from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class ModelConfig:
    model_name: str = "qwen3.5:4b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 1024

@dataclass
class SandboxConfig:
    image: str = "python:3.11-slim"
    timeout: int = 30
    memory_limit: str = "256m"
    network_disabled: bool = False
    max_fix_attempts: int = 3
    mount_path: str = os.path.expanduser("~")  # mounts /home/glitch → /workspace

@dataclass
class RAGConfig:
    chroma_path: str = "./chroma_db"
    collection_name: str = "linux_docs"
    embedding_model: str = "nomic-embed-text"
    top_k: int = 3

@dataclass
class AgentConfig:
    max_iterations: int = 10
    verbose: bool = True
    safe_mode: bool = True
    confirm_before_run: bool = True

@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    log_level: str = "INFO"
    history_file: str = os.path.expanduser("~/.linux_assistant_history.json")

config = AppConfig()