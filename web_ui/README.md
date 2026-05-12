# LinuxGPT — AI Linux Assistant

A terminal-aesthetic web UI for an AI-powered Linux assistant, built with **Python + Gradio + Claude Sonnet**.

---

## Features

| Feature | Details |
|---|---|
| 💬 Chat | Streaming chat with Claude Sonnet via Anthropic API |
| 🖥 System Info | Live CPU, RAM, Disk, OS info in the sidebar |
| ▶ Run Commands | Execute shell commands directly from the UI (with basic safety filter) |
| 📋 Paste Output | Paste terminal output and ask Claude to explain it |
| ⚡ Quick Prompts | One-click common Linux diagnostic prompts |
| 🎨 Dark Terminal UI | JetBrains Mono, green-on-black, polished Gradio custom CSS |

---

## Setup

### 1. Install dependencies

```bash
pip install gradio anthropic psutil
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run

```bash
python linux_assistant.py
```

Then open **http://localhost:7860** in your browser.

---

## Usage Tips

- **Chat box** — ask anything Linux-related (commands, scripting, debugging, sysadmin)
- **Run Command** — type a shell command, hit Execute; output appears below
- **Paste Output** — paste terminal output in the lower text area before sending a message; Claude will receive it as context automatically
- **Quick Prompts** — click any shortcut in the sidebar to prefill common questions
- **Refresh** — click ⟳ Refresh to update the live system stats

---

## Configuration

| Option | Location | Default |
|---|---|---|
| Model | `linux_assistant.py` → `model=` | `claude-sonnet-4-20250514` |
| Port | `demo.launch(server_port=...)` | `7860` |
| Public share link | `demo.launch(share=True)` | `False` |
| Blocked commands | `BLOCKED` list in `run_safe_command()` | rm -rf /, shutdown, etc. |

---

## Security Note

The **Run Command** panel executes real shell commands as your user. The built-in blocklist only catches obvious destructive patterns. Do **not** expose this app publicly (`share=False`, no reverse-proxy without auth) unless you have proper access controls.
