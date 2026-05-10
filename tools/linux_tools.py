# tools/linux_tools.py
# All LangChain tools available to the agent
# Each tool is self-contained and independently testable

from langchain.tools import tool
from sandbox.docker_sandbox import sandbox
from memory.session_memory import session
from config.settings import config
import chromadb
from chromadb.utils import embedding_functions


# ─────────────────────────────────────────────
# Tool 1: Execute Command
# ─────────────────────────────────────────────

@tool
def execute_command(command: str) -> str:
    """
    Execute a bash command in a secure Docker sandbox.
    Use this to run Linux commands, check output, install packages, etc.
    Returns stdout, stderr, and exit code.
    Input: the exact bash command string to run.
    """
    result = sandbox.run_session(command, workdir=session.cwd)

    # Track in session memory
    session.add_command(
        command=command,
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
    )

    if result.blocked:
        return f"BLOCKED: {result.block_reason}\nThis command was blocked for safety reasons."

    if result.timed_out:
        return f"TIMED OUT: Command exceeded {config.sandbox.timeout}s limit."

    output_parts = []
    if result.stdout.strip():
        output_parts.append(f"STDOUT:\n{result.stdout.strip()}")
    if result.stderr.strip():
        output_parts.append(f"STDERR:\n{result.stderr.strip()}")
    output_parts.append(f"EXIT CODE: {result.exit_code}")

    return "\n".join(output_parts) if output_parts else "Command ran with no output. Exit code: 0"


# ─────────────────────────────────────────────
# Tool 2: RAG Search (Man pages / Arch wiki)
# ─────────────────────────────────────────────

def _get_rag_collection():
    """Lazy-load the ChromaDB collection"""
    client = chromadb.PersistentClient(path=config.rag.chroma_path)
    ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=config.rag.embedding_model,
    )
    return client.get_or_create_collection(
        name=config.rag.collection_name,
        embedding_function=ef,
    )

@tool
def rag_search(query: str) -> str:
    """
    Search Linux documentation — man pages, Arch wiki, command references.
    Use this when you need to look up how a command works, its flags,
    or when answering a general Linux question.
    Input: a natural language search query.
    """
    try:
        collection = _get_rag_collection()
        results = collection.query(
            query_texts=[query],
            n_results=config.rag.top_k,
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant documentation found."
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"[Doc {i}]\n{doc[:600]}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"RAG search unavailable: {e}"


# ─────────────────────────────────────────────
# Tool 3: Get System Info
# ─────────────────────────────────────────────

@tool
def get_system_info(detail: str = "all") -> str:
    """
    Get current system information — OS, kernel, CPU, RAM, GPU, current directory,
    recent command history. Use this to understand the user's environment before
    generating commands or when the user asks about their system.
    Input: 'all', 'os', 'hardware', 'history', or 'cwd'
    """
    info = session.system_info
    detail = detail.lower().strip()

    if detail == "os":
        return f"OS: {info.distro} | Kernel: {info.kernel} | Shell: {info.shell}"
    elif detail == "hardware":
        return f"CPU: {info.cpu_cores} cores | RAM: {info.ram_gb}GB | GPU: {info.gpu}"
    elif detail == "history":
        recent = session.get_recent_commands(5)
        if not recent:
            return "No commands run yet this session."
        lines = [f"{'✓' if e.exit_code == 0 else '✗'} {e.command}" for e in recent]
        return "Recent commands:\n" + "\n".join(lines)
    elif detail == "cwd":
        return f"Current directory: {session.cwd}"
    else:
        return session.get_context_summary()


# ─────────────────────────────────────────────
# Tool 4: Explain Command
# ─────────────────────────────────────────────

@tool
def explain_command(command: str) -> str:
    """
    Break down and explain what a bash command does — each flag, pipe, and subcommand.
    Use this when the user pastes a command they don't understand,
    or to explain a command before running it.
    Input: the bash command string to explain.
    """
    # This tool returns a structured prompt — the LLM fills in the explanation
    # In practice the agent itself does the explaining, this just formats it
    parts = command.split("|")
    if len(parts) > 1:
        pipeline = " | ".join(f"({p.strip()})" for p in parts)
        return f"Pipeline detected: {pipeline}\nExplain each stage and how they connect."
    return f"Explain the command: {command}\nBreak down each flag and argument."


# ─────────────────────────────────────────────
# Tool 5: Classify Error
# ─────────────────────────────────────────────

@tool
def classify_error(stderr: str) -> str:
    """
    Classify a Linux error message and suggest a fix category.
    Use this when a command fails and you need to decide how to fix it.
    Input: the stderr or error message string.
    """
    stderr_lower = stderr.lower()

    error_map = {
        "permission denied":        ("PERMISSION",    "Try with sudo or check file permissions with ls -la"),
        "command not found":        ("NOT_FOUND",     "Package not installed — use apt/pacman/pip to install it"),
        "no such file or directory":("MISSING_PATH",  "Path doesn't exist — check spelling or create the directory"),
        "connection refused":       ("NETWORK",       "Service not running or wrong port — check with systemctl status"),
        "address already in use":   ("PORT_CONFLICT", "Port is taken — find the process with lsof -i :<port>"),
        "killed":                   ("OOM",           "Process killed by OOM killer — reduce memory usage"),
        "syntax error":             ("SYNTAX",        "Shell syntax error — check quotes, brackets, semicolons"),
        "disk quota exceeded":      ("DISK_FULL",     "Disk full — check with df -h and clear space"),
        "broken pipe":              ("PIPE",          "Downstream process closed — check the full pipeline"),
        "timeout":                  ("TIMEOUT",       "Command timed out — check network or increase timeout"),
    }

    for pattern, (error_type, suggestion) in error_map.items():
        if pattern in stderr_lower:
            return f"ERROR TYPE: {error_type}\nSUGGESTION: {suggestion}\nRAW: {stderr[:200]}"

    return f"ERROR TYPE: UNKNOWN\nSUGGESTION: Search docs or check man page\nRAW: {stderr[:200]}"


# ─────────────────────────────────────────────
# Export all tools as a list
# ─────────────────────────────────────────────

ALL_TOOLS = [
    execute_command,
    rag_search,
    get_system_info,
    explain_command,
    classify_error,
]
