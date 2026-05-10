# Linux Assistant

> A fine-tuned LLM-powered Linux terminal assistant with agentic capabilities, built as part of an NLP course project at university.

---

## Overview

Linux Assistant is an intelligent terminal agent that combines a fine-tuned language model with a ReAct (Reasoning + Acting) agent architecture. It translates natural language into shell commands, answers Linux questions, explains complex commands, diagnoses errors, and executes multi-step tasks — all within a secure Docker sandbox.

The system is built on three pillars:

- **Fine-tuned model** — Qwen3.5-4B trained on NL2Bash + synthetic ReAct traces using Unsloth + QLoRA
- **Agentic reasoning** — LangChain 1.x ReAct loop with tool calling and multi-turn conversation memory
- **Safe execution** — Commands run inside an isolated Docker container with destructive pattern blocking

---

## Features

| Capability | Description |
|---|---|
| NL → Bash | Translates plain English to shell commands |
| Q&A | Answers general Linux questions with optional RAG over man pages |
| Command explanation | Breaks down flags, pipes, and subcommands |
| Error recovery | Diagnoses stderr, classifies error type, proposes and runs a fix |
| Multi-step execution | Plans and executes compound tasks step by step |
| Session memory | Tracks cwd, command history, and conversation context across turns |
| Safe mode | Blocks destructive patterns, warns before risky operations |

---

## Project Structure

```
linux_assistant/
├── main.py                  # Entry point — CLI or evaluation mode
├── requirements.txt
├── config/
│   └── settings.py          # Centralized configuration (model, sandbox, RAG)
├── agent/
│   └── linux_agent.py       # ReAct agent core with conversation history
├── tools/
│   └── linux_tools.py       # LangChain tools: execute_command, rag_search,
│                            #   get_system_info, explain_command, classify_error
├── sandbox/
│   └── docker_sandbox.py    # Isolated Docker execution with volume mounting
├── memory/
│   └── session_memory.py    # Session context, system info, command history
├── cli/
│   └── terminal.py          # Terminal UI with ANSI colors and special commands
└── evaluation/
    └── eval.py              # BLEU score, exact match, agent task success rate
```

---

## Installation

### Prerequisites

- Python 3.11+
- Docker
- Ollama (for local model inference)

```bash
# Install Docker (Arch Linux)
sudo pacman -S docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER   # avoid needing sudo for docker

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
pip install -r requirements.txt
```

### Model Setup

Before running, pull a base model or load your fine-tuned model via Ollama:

```bash
# Use a base model for testing
ollama pull qwen3.5:4b

# Or load your fine-tuned GGUF
ollama create linux-assistant -f Modelfile
```

---

## Usage

```bash
# Start the assistant with a specific model
python main.py --model qwen3.5:4b

# Disable confirmation prompts (runs commands immediately)
python main.py --model qwen3.5:4b --no-confirm

# Disable safe mode (allow all commands)
python main.py --model qwen3.5:4b --no-safe

# Run evaluation suite
python main.py --eval --test-file data/test.json --limit 100
```

### CLI Commands

| Command | Description |
|---|---|
| `:help` | Show available commands |
| `:history` | Show commands run this session |
| `:sysinfo` | Show system information |
| `:safe` | Toggle safe mode on/off |
| `:reset` | Clear conversation history |
| `:clear` | Clear the terminal screen |
| `:exit` | Exit the assistant |

### Example Queries

```
find all python files modified in the last 24 hours
how do I check which process is using port 8080?
explain: tar -czf archive.tar.gz ./folder
set up a python venv called myenv and install flask
i got permission denied on /etc/shadow, how do I fix this?
show disk usage of my home directory sorted by size
```

---

## Configuration

All settings are in `config/settings.py`. Key options:

```python
ModelConfig:
    model_name      # Ollama model name
    temperature     # Lower = more deterministic commands (default: 0.1)

SandboxConfig:
    image           # Docker image (default: python:3.11-slim)
    timeout         # Command timeout in seconds (default: 30)
    memory_limit    # Container memory cap (default: 256m)
    mount_path      # Host path mounted as /workspace (default: ~/home)
    network_disabled # Disable network inside container (default: False)

AgentConfig:
    safe_mode           # Block destructive commands (default: True)
    confirm_before_run  # Propose commands before running (default: True)
    max_iterations      # Max ReAct loop steps (default: 10)
```

---

## Evaluation

The evaluation module measures model quality across two dimensions:

- **NL2Bash** — Exact match and BLEU score against the NL2Bash test set
- **Agent tasks** — Keyword-based success rate on a benchmark of 5 multi-step tasks

```bash
python main.py --eval --test-file data/nl2bash_test.json --limit 100
```

---

## Fine-tuning

The model is fine-tuned using [Unsloth](https://github.com/unslothai/unsloth) with LoRA on a mixed dataset:

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| NL2Bash | HuggingFace: `AnishJoshi/nl2bash-custom` | ~19k | Command translation |
| Bash Q&A | HuggingFace: `aelhalili/bash-commands-dataset` | ~500 | General Q&A |
| ReAct traces | Synthetically generated | ~400 | Agentic behavior |

Training target: **Qwen3.5-4B** with bf16 LoRA (~10GB VRAM), exported to GGUF for Ollama deployment.

---

## Architecture

```
User Input
    ↓
Intent Classifier
    ↓
ReAct Agent Loop (LangChain 1.x)
    ├── Thought  →  reason about the task
    ├── Action   →  call a tool
    │               ├── execute_command  →  Docker sandbox
    │               ├── rag_search       →  ChromaDB (man pages)
    │               ├── get_system_info  →  psutil + session memory
    │               ├── explain_command  →  parse flags/pipes
    │               └── classify_error   →  error type + fix suggestion
    └── Observation → read output, decide next step
    ↓
Final Answer → CLI
```

---

## Tech Stack

- **LangChain 1.x** — Agent framework
- **Ollama** — Local model inference
- **Unsloth** — Efficient fine-tuning
- **ChromaDB** — Vector store for RAG
- **Docker** — Sandboxed command execution
- **psutil** — System information
- **NLTK** — BLEU score evaluation

---

## License

MIT