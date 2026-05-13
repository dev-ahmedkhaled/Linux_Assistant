"""
╔══════════════════════════════════════════════════════════╗
║  LinuxGPT — AI Linux Assistant                           ║
║  Backend : Ollama (local LLM)                            ║
║  UI      : Gradio  (clean modern dark)                   ║
║                                                          ║
║  Requirements:                                           ║
║    pip install gradio psutil requests                    ║
║    ollama pull llama3  (or any model you prefer)         ║
║                                                          ║
║  Run: python linux_assistant_ui.py                       ║
╚══════════════════════════════════════════════════════════╝
"""
import os
import time
import json
import shutil
import psutil
import platform
import requests
import subprocess
import gradio as gr
from datetime import datetime

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("LINUX_ASSISTANT_MODEL", "llama3")

SYSTEM_PROMPT = """You are LinuxGPT, an expert Linux assistant. You help with:
- Linux commands, shell scripting, and one-liners
- Debugging errors and interpreting terminal output
- System administration, networking, and performance tuning
- Package management, file operations, and permissions
- Explaining concepts clearly with practical examples

Always format command examples in code blocks. Be concise but thorough.
When you suggest commands, briefly explain what each flag does."""

BLOCKED_CMDS = {
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev/zero",
    ":(){ :|:& };:", "chmod -R 777 /", "> /dev/sda",
    "shutdown", "reboot", "halt", "poweroff",
}

QUICK_PROMPTS = [
    ("📂 Disk usage", "Show me disk usage sorted by size for the current directory"),
    ("🔍 Find large files", "How do I find the 10 largest files on my system?"),
    ("🔒 Check permissions", "Explain Linux file permissions and how to fix 'permission denied'"),
    ("🌐 Network debug", "How do I debug network connectivity issues in Linux?"),
    ("⚙️  Running services", "How do I list all running services and check their status?"),
    ("📋 View logs", "What are the best ways to view and filter system logs?"),
    ("🔄 Background jobs", "Explain Linux job control: bg, fg, nohup, and screen/tmux"),
    ("💾 Memory info", "How do I check memory usage and find memory-hungry processes?"),
]

# ─────────────────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────────────────

def list_ollama_models() -> list[str]:
    """Return available models from Ollama, fallback to default."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if r.ok:
            models = [m["name"] for m in r.json().get("models", [])]
            return models if models else [DEFAULT_MODEL]
    except Exception:
        pass
    return [DEFAULT_MODEL]


def check_ollama() -> tuple[bool, str]:
    """Check if Ollama is reachable."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if r.ok:
            models = r.json().get("models", [])
            return True, f"Ollama running · {len(models)} model(s) available"
    except requests.ConnectionError:
        return False, "Cannot reach Ollama — is it running? (ollama serve)"
    except Exception as e:
        return False, str(e)
    return False, "Ollama returned an unexpected response"


def stream_ollama(model: str, messages: list[dict]) -> str:
    """Stream a response from Ollama and yield chunks."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    try:
        with requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            full = ""
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    full += delta
                    yield full
    except requests.ConnectionError:
        yield "❌ **Ollama not reachable.** Make sure `ollama serve` is running."
    except Exception as e:
        yield f"❌ **Error:** {e}"


# ─────────────────────────────────────────────────────────
# System stats
# ─────────────────────────────────────────────────────────

def get_system_stats() -> str:
    """Return a markdown summary of live system stats."""
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    boot = datetime.fromtimestamp(psutil.boot_time())
    uptime_secs = int(time.time() - psutil.boot_time())
    h, r = divmod(uptime_secs, 3600)
    m, s = divmod(r, 60)
    uptime_str = f"{h}h {m}m {s}s"

    try:
        uname = platform.uname()
        os_info = f"{uname.system} {uname.release}"
    except Exception:
        os_info = platform.system()

    def bar(pct, width=20):
        filled = int(pct / 100 * width)
        return "█" * filled + "░" * (width - filled)

    ram_used_gb = ram.used / 1024**3
    ram_total_gb = ram.total / 1024**3
    disk_used_gb = disk.used / 1024**3
    disk_total_gb = disk.total / 1024**3

    lines = [
        f"**OS:** `{os_info}` &nbsp;|&nbsp; **Uptime:** `{uptime_str}`",
        "",
        f"**CPU** &nbsp;{cpu:.1f}%",
        f"`{bar(cpu)}`",
        "",
        f"**RAM** &nbsp;{ram_used_gb:.1f} / {ram_total_gb:.1f} GB &nbsp;({ram.percent:.1f}%)",
        f"`{bar(ram.percent)}`",
        "",
        f"**Disk** &nbsp;{disk_used_gb:.1f} / {disk_total_gb:.1f} GB &nbsp;({disk.percent:.1f}%)",
        f"`{bar(disk.percent)}`",
    ]

    # Top 5 processes by CPU
    try:
        procs = sorted(
            psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]),
            key=lambda p: p.info["cpu_percent"] or 0,
            reverse=True,
        )[:5]
        lines += ["", "**Top Processes**"]
        for p in procs:
            name = (p.info["name"] or "?")[:18]
            cpu_p = p.info["cpu_percent"] or 0
            mem_p = p.info["memory_percent"] or 0
            lines.append(f"`{p.info['pid']:>6}` `{name:<18}` CPU {cpu_p:>5.1f}%  MEM {mem_p:>4.1f}%")
    except Exception:
        pass

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Shell command runner
# ─────────────────────────────────────────────────────────

def run_command(cmd: str) -> str:
    """Run a shell command safely and return output."""
    if not cmd.strip():
        return ""
    # Safety check
    cmd_lower = cmd.strip().lower()
    for blocked in BLOCKED_CMDS:
        if blocked in cmd_lower:
            return f"⛔ **Blocked:** `{blocked}` is not allowed for safety reasons."

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        code = result.returncode

        parts = []
        if out:
            parts.append(f"```\n{out}\n```")
        if err:
            parts.append(f"**stderr:**\n```\n{err}\n```")
        if not out and not err:
            parts.append(f"*(no output — exit code {code})*")
        if code != 0:
            parts.append(f"*Exit code: {code}*")
        return "\n".join(parts)

    except subprocess.TimeoutExpired:
        return "⏱ **Timeout** — command took more than 30 seconds."
    except Exception as e:
        return f"❌ **Error:** {e}"


# ─────────────────────────────────────────────────────────
# Chat logic
# ─────────────────────────────────────────────────────────

def build_messages(history: list, user_msg: str, paste_ctx: str) -> list[dict]:
    """Assemble the Ollama messages list."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for role, content in history:
        messages.append({"role": "user" if role == "user" else "assistant", "content": content})

    # Append pasted context if present
    if paste_ctx.strip():
        user_msg = f"{user_msg}\n\n**Terminal output to analyse:**\n```\n{paste_ctx.strip()}\n```"

    messages.append({"role": "user", "content": user_msg})
    return messages


def chat(user_msg, history, model, paste_ctx):
    """Called by Gradio chat submit — streams the reply."""
    if not user_msg.strip():
        yield history, ""
        return

    messages = build_messages(history, user_msg, paste_ctx)
    history = history + [("user", user_msg)]
    partial = ""

    for chunk in stream_ollama(model, messages):
        partial = chunk
        display = history + [("assistant", partial)]
        yield display, ""

    # Final yield — clear paste box after send
    yield history + [("assistant", partial)], ""


def clear_chat():
    return [], "", ""


def apply_quick_prompt(prompt_text):
    return prompt_text


def refresh_stats():
    return get_system_stats()


# ─────────────────────────────────────────────────────────
# CSS  — clean modern dark
# ─────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-base:    #0f1117;
    --bg-surface: #161b27;
    --bg-card:    #1c2333;
    --bg-input:   #1e2638;
    --border:     #2a3348;
    --border-hi:  #3d4f70;
    --accent:     #4f8ef7;
    --accent-dim: #1e3a6e;
    --green:      #3dd68c;
    --red:        #f76f6f;
    --amber:      #f5a623;
    --text-1:     #e8edf5;
    --text-2:     #8b99b5;
    --text-3:     #4a566e;
    --mono:       'DM Mono', monospace;
    --sans:       'DM Sans', sans-serif;
    --radius:     10px;
    --radius-sm:  6px;
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-base) !important;
    font-family: var(--sans) !important;
    color: var(--text-1) !important;
    font-size: 14px !important;
}

.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 0 16px !important; }

/* ── App header ── */
#app-header {
    padding: 20px 0 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
#app-header .logo {
    width: 40px; height: 40px;
    background: var(--accent-dim);
    border-radius: var(--radius-sm);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-weight: 500; font-size: 18px; color: var(--accent);
}
#app-header h1 {
    font-family: var(--sans); font-size: 20px; font-weight: 600;
    color: var(--text-1); margin: 0; letter-spacing: -0.3px;
}
#app-header .sub {
    font-size: 12px; color: var(--text-2); margin: 2px 0 0;
}
#status-pill {
    margin-left: auto;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 14px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-2);
}
#status-pill.ok  { border-color: var(--green); color: var(--green); }
#status-pill.err { border-color: var(--red);   color: var(--red); }

/* ── Section labels ── */
.section-label {
    font-size: 10px; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--text-3);
    margin: 0 0 10px; padding: 0;
}

/* ── Panels / cards ── */
.panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
}
.panel-head {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 11px; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: var(--text-2);
    background: var(--bg-card);
}

/* ── Inputs ── */
input[type=text], textarea, .gr-input, .gr-textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-1) !important;
    font-family: var(--sans) !important;
    font-size: 14px !important;
    border-radius: var(--radius-sm) !important;
    caret-color: var(--accent) !important;
    transition: border-color .15s !important;
}
input[type=text]:focus, textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,.12) !important;
}

/* ── Buttons ── */
button { font-family: var(--sans) !important; }

button.primary, .gr-button-primary {
    background: var(--accent) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: var(--radius-sm) !important;
    padding: 9px 18px !important;
    transition: opacity .15s, transform .1s !important;
}
button.primary:hover { opacity: .88 !important; }
button.primary:active { transform: scale(.97) !important; }

button.secondary, .gr-button-secondary {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-2) !important;
    font-size: 12px !important;
    border-radius: var(--radius-sm) !important;
    padding: 7px 14px !important;
    transition: border-color .15s, color .15s !important;
}
button.secondary:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Chatbot ── */
.gr-chatbot, [class*="chatbot"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.message.user    { background: var(--accent-dim) !important; color: var(--text-1) !important; border-radius: 12px 12px 2px 12px !important; }
.message.bot     { background: var(--bg-card)    !important; color: var(--text-1) !important; border-radius: 12px 12px 12px 2px !important; border: 1px solid var(--border) !important; }
.message.user .md, .message.bot .md { font-family: var(--sans) !important; font-size: 14px !important; line-height: 1.65 !important; }
.message code { background: rgba(0,0,0,.4) !important; font-family: var(--mono) !important; font-size: 12px !important; padding: 1px 6px !important; border-radius: 4px !important; color: #93c5fd !important; }
.message pre  { background: #0a0e18 !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; padding: 12px !important; }
.message pre code { background: transparent !important; color: var(--green) !important; font-size: 12px !important; }

/* ── Markdown (stats panel) ── */
.gr-markdown { color: var(--text-1) !important; }
.gr-markdown code { background: var(--bg-card) !important; color: var(--green) !important; font-family: var(--mono) !important; font-size: 11.5px !important; padding: 2px 6px !important; border-radius: 4px !important; }
.gr-markdown strong { color: var(--text-1) !important; font-weight: 600 !important; }

/* ── Dropdown ── */
.gr-dropdown, select {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-1) !important;
    font-family: var(--sans) !important;
    font-size: 13px !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Tabs ── */
.tabs .tab-nav button {
    font-family: var(--sans) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    transition: color .15s, border-color .15s !important;
}
.tabs .tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
.tabs { border-bottom: 1px solid var(--border) !important; margin-bottom: 16px !important; }

/* ── Command output ── */
#cmd-output textarea {
    font-family: var(--mono) !important;
    font-size: 12.5px !important;
    color: var(--green) !important;
    background: #070b12 !important;
    border: 1px solid var(--border) !important;
    min-height: 140px !important;
    line-height: 1.6 !important;
}

/* ── Labels ── */
label > span, .label-wrap span {
    font-family: var(--sans) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: .8px !important;
    text-transform: uppercase !important;
    color: var(--text-3) !important;
}

/* ── Footer ── */
#footer {
    text-align: center; padding: 16px;
    font-size: 11px; color: var(--text-3);
    border-top: 1px solid var(--border); margin-top: 24px;
    font-family: var(--mono);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-hi); }
"""

# ─────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────

def build_ui():
    ok, status_msg = check_ollama()
    models = list_ollama_models()
    status_class = "ok" if ok else "err"
    status_icon  = "● " if ok else "✕ "

    with gr.Blocks(css=CSS, title="LinuxGPT", theme=gr.themes.Base()) as demo:

        # ── Header ──
        gr.HTML(f"""
        <div id="app-header">
          <div class="logo">&gt;_</div>
          <div>
            <h1>LinuxGPT</h1>
            <div class="sub">AI-powered Linux assistant · Ollama backend</div>
          </div>
          <div id="status-pill" class="{status_class}">{status_icon}{status_msg}</div>
        </div>
        """)

        with gr.Tabs():

            # ═══════════════════════════════════════
            # TAB 1 — CHAT
            # ═══════════════════════════════════════
            with gr.Tab("💬  Chat"):
                with gr.Row(equal_height=False):

                    # Left sidebar
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<p class="section-label">Model</p>')
                        model_dd = gr.Dropdown(
                            choices=models,
                            value=models[0] if models else DEFAULT_MODEL,
                            label="",
                            interactive=True,
                        )

                        gr.HTML('<p class="section-label" style="margin-top:20px">Quick Prompts</p>')
                        for icon_label, prompt_text in QUICK_PROMPTS:
                            btn = gr.Button(icon_label, variant="secondary", size="sm")
                            btn.click(
                                fn=apply_quick_prompt,
                                inputs=gr.State(prompt_text),
                                outputs=gr.Textbox(visible=False),  # temp, wired below
                            )

                        gr.HTML('<p class="section-label" style="margin-top:20px">Paste Terminal Output</p>')
                        paste_box = gr.Textbox(
                            placeholder="Paste error or command output here…\nClaude will use it as context.",
                            label="",
                            lines=6,
                            max_lines=12,
                        )
                        clear_paste_btn = gr.Button("Clear paste", variant="secondary", size="sm")
                        clear_paste_btn.click(fn=lambda: "", outputs=paste_box)

                    # Main chat area
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="",
                            height=520,
                            show_label=False,
                            show_copy_button=True,
                            bubble_full_width=False,
                        )
                        with gr.Row():
                            user_input = gr.Textbox(
                                placeholder="Ask anything Linux — commands, debugging, scripting…",
                                label="",
                                scale=5,
                                show_label=False,
                                container=False,
                            )
                            send_btn = gr.Button("Send ↑", variant="primary", scale=1)
                        with gr.Row():
                            clear_btn = gr.Button("Clear chat", variant="secondary", size="sm")

                # Wire quick-prompt buttons properly
                qp_state = gr.State("")
                for icon_label, prompt_text in QUICK_PROMPTS:
                    pass  # re-wired below with the actual user_input component

                # Chat submit
                submit_args = dict(
                    fn=chat,
                    inputs=[user_input, chatbot, model_dd, paste_box],
                    outputs=[chatbot, user_input],
                )
                send_btn.click(**submit_args)
                user_input.submit(**submit_args)
                clear_btn.click(fn=clear_chat, outputs=[chatbot, user_input, paste_box])

                # Quick prompts → user input
                for icon_label, prompt_text in QUICK_PROMPTS:
                    # Re-find button — rebuild reference by recreating inside loop
                    pass

            # ═══════════════════════════════════════
            # TAB 2 — RUN COMMANDS
            # ═══════════════════════════════════════
            with gr.Tab("⚙️  Run Commands"):
                gr.Markdown(
                    "> **Safety note:** Commands run as your current user. "
                    "Destructive patterns (`rm -rf /`, `mkfs`, etc.) are blocked.",
                    elem_classes="gr-markdown",
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<p class="section-label">Common Commands</p>')
                        common_cmds = [
                            ("📋 List files",          "ls -lah"),
                            ("💾 Disk usage (dir)",    "du -sh ./* | sort -hr | head -20"),
                            ("🔎 Disk free",           "df -h"),
                            ("⚙️  Running processes",  "ps aux --sort=-%cpu | head -20"),
                            ("🌐 Network interfaces",  "ip a"),
                            ("🔗 Open ports",          "ss -tulnp"),
                            ("📜 Recent errors (log)", "journalctl -p err -n 30 --no-pager"),
                            ("🔑 Who is logged in",    "w"),
                            ("📦 Last installs (apt)", "grep ' install ' /var/log/dpkg.log 2>/dev/null | tail -20 || echo 'Not a Debian/Ubuntu system'"),
                            ("🧩 Kernel version",      "uname -r"),
                        ]
                        for label, cmd_text in common_cmds:
                            b = gr.Button(label, variant="secondary", size="sm")
                            b.click(fn=lambda c=cmd_text: c, outputs=gr.Textbox(visible=False))

                    with gr.Column(scale=3):
                        cmd_input = gr.Textbox(
                            label="Command",
                            placeholder="e.g.   ls -lah   or   ps aux | grep python",
                            lines=2,
                        )
                        with gr.Row():
                            run_btn  = gr.Button("▶ Execute", variant="primary")
                            copy_btn = gr.Button("Explain with AI →", variant="secondary")

                        cmd_output = gr.Markdown(
                            label="Output",
                            elem_id="cmd-output",
                        )

                        run_btn.click(fn=run_command, inputs=cmd_input, outputs=cmd_output)
                        cmd_input.submit(fn=run_command, inputs=cmd_input, outputs=cmd_output)

                # Wire common command buttons to cmd_input
                for label, cmd_text in common_cmds:
                    pass  # handled below via a proper pattern

            # ═══════════════════════════════════════
            # TAB 3 — SYSTEM STATS
            # ═══════════════════════════════════════
            with gr.Tab("📊  System Stats"):
                with gr.Row():
                    refresh_btn = gr.Button("⟳ Refresh", variant="secondary", size="sm")

                stats_md = gr.Markdown(
                    value=get_system_stats(),
                    elem_classes="gr-markdown",
                )

                refresh_btn.click(fn=refresh_stats, outputs=stats_md)

                # Auto-refresh every 5s
                stats_timer = gr.Timer(value=5.0)
                stats_timer.tick(fn=refresh_stats, outputs=stats_md)

        # Footer
        gr.HTML("""
        <div id="footer">
          LinuxGPT &nbsp;·&nbsp; Ollama backend &nbsp;·&nbsp; Commands execute on this machine
        </div>
        """)

    return demo


# ─────────────────────────────────────────────────────────
# Rebuild with proper button wiring (Gradio needs
# component references to be in scope at .click() time)
# ─────────────────────────────────────────────────────────

def build_ui_v2():
    ok, status_msg = check_ollama()
    models = list_ollama_models()
    status_class = "ok" if ok else "err"
    status_icon  = "● " if ok else "✕ "

    with gr.Blocks(css=CSS, title="LinuxGPT", theme=gr.themes.Base()) as demo:

        # ── Header ──
        gr.HTML(f"""
        <div id="app-header">
          <div class="logo">&gt;_</div>
          <div>
            <h1>LinuxGPT</h1>
            <div class="sub">AI-powered Linux assistant · Ollama backend</div>
          </div>
          <div id="status-pill" class="{status_class}">{status_icon}{status_msg}</div>
        </div>
        """)

        with gr.Tabs():

            # ═══════════════════════════════════════
            # TAB 1 — CHAT
            # ═══════════════════════════════════════
            with gr.Tab("💬  Chat"):
                with gr.Row(equal_height=False):

                    # ── Sidebar ──
                    with gr.Column(scale=1, min_width=230):
                        gr.HTML('<p class="section-label">Model</p>')
                        model_dd = gr.Dropdown(
                            choices=models,
                            value=models[0] if models else DEFAULT_MODEL,
                            label="",
                            interactive=True,
                        )

                        gr.HTML('<p class="section-label" style="margin-top:20px">Quick Prompts</p>')
                        qp_buttons = []
                        for icon_label, _ in QUICK_PROMPTS:
                            b = gr.Button(icon_label, variant="secondary", size="sm")
                            qp_buttons.append(b)

                        gr.HTML('<p class="section-label" style="margin-top:20px">Paste Terminal Output</p>')
                        paste_box = gr.Textbox(
                            placeholder="Paste error or command output here…\nClaude will include it as context.",
                            label="",
                            lines=6,
                            max_lines=14,
                        )
                        clear_paste_btn = gr.Button("Clear paste", variant="secondary", size="sm")

                    # ── Chat column ──
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="",
                            height=520,
                            show_label=False,
                            show_copy_button=True,
                            bubble_full_width=False,
                        )
                        with gr.Row():
                            user_input = gr.Textbox(
                                placeholder="Ask anything Linux — commands, debugging, scripting…",
                                label="",
                                scale=5,
                                show_label=False,
                                container=False,
                            )
                            send_btn = gr.Button("Send ↑", variant="primary", scale=1)
                        with gr.Row():
                            clear_btn = gr.Button("Clear chat", variant="secondary", size="sm")

                # Wire quick prompts to user_input
                for btn, (_, prompt_text) in zip(qp_buttons, QUICK_PROMPTS):
                    btn.click(fn=lambda p=prompt_text: p, outputs=user_input)

                clear_paste_btn.click(fn=lambda: "", outputs=paste_box)

                submit_args = dict(
                    fn=chat,
                    inputs=[user_input, chatbot, model_dd, paste_box],
                    outputs=[chatbot, user_input],
                )
                send_btn.click(**submit_args)
                user_input.submit(**submit_args)
                clear_btn.click(fn=clear_chat, outputs=[chatbot, user_input, paste_box])

            # ═══════════════════════════════════════
            # TAB 2 — RUN COMMANDS
            # ═══════════════════════════════════════
            with gr.Tab("⚙️  Run Commands"):
                gr.HTML("""
                <div style="background:#1c1a14;border:1px solid #3d3720;border-radius:8px;
                            padding:10px 16px;margin-bottom:16px;font-size:13px;color:#f5a623;">
                  ⚠ Commands execute on this machine as your current user.
                  Destructive patterns are blocked.
                </div>
                """)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=200):
                        gr.HTML('<p class="section-label">Common Commands</p>')
                        common_cmds = [
                            ("📋 List files (detail)",  "ls -lah"),
                            ("💾 Dir sizes",            "du -sh ./* | sort -hr | head -20"),
                            ("🔎 Disk free space",      "df -h"),
                            ("⚙️  Top CPU processes",   "ps aux --sort=-%cpu | head -20"),
                            ("🌐 Network interfaces",   "ip addr show"),
                            ("🔗 Open ports",           "ss -tulnp"),
                            ("📜 System errors",        "journalctl -p err -n 30 --no-pager"),
                            ("👥 Logged in users",      "w"),
                            ("🧩 Kernel version",       "uname -r"),
                            ("📦 Systemd services",     "systemctl list-units --type=service --state=running"),
                        ]
                        cc_buttons = []
                        for label, _ in common_cmds:
                            b = gr.Button(label, variant="secondary", size="sm")
                            cc_buttons.append(b)

                    with gr.Column(scale=3):
                        cmd_input = gr.Textbox(
                            label="Command",
                            placeholder="e.g.   ls -lah   or   cat /etc/os-release",
                            lines=2,
                        )
                        with gr.Row():
                            run_btn     = gr.Button("▶ Execute", variant="primary")
                            explain_btn = gr.Button("🤖 Explain output with AI", variant="secondary")

                        cmd_output_md = gr.Markdown(
                            value="",
                            label="Output",
                            elem_id="cmd-output",
                        )

                # Wire common command buttons
                for btn, (_, cmd_text) in zip(cc_buttons, common_cmds):
                    btn.click(fn=lambda c=cmd_text: c, outputs=cmd_input)

                run_btn.click(fn=run_command, inputs=cmd_input, outputs=cmd_output_md)
                cmd_input.submit(fn=run_command, inputs=cmd_input, outputs=cmd_output_md)

            # ═══════════════════════════════════════
            # TAB 3 — SYSTEM STATS
            # ═══════════════════════════════════════
            with gr.Tab("📊  System Stats"):
                with gr.Row():
                    refresh_btn = gr.Button("⟳ Refresh", variant="secondary", size="sm")

                stats_md = gr.Markdown(
                    value=get_system_stats(),
                    elem_classes="gr-markdown",
                )

                refresh_btn.click(fn=refresh_stats, outputs=stats_md)
                stats_timer = gr.Timer(value=5.0)
                stats_timer.tick(fn=refresh_stats, outputs=stats_md)

        gr.HTML("""
        <div id="footer">
          LinuxGPT &nbsp;·&nbsp; Ollama backend &nbsp;·&nbsp;
          Commands execute on this machine &nbsp;·&nbsp;
          Runs at <code>http://localhost:7860</code>
        </div>
        """)

    return demo


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  LinuxGPT — AI Linux Assistant")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    ok, msg = check_ollama()
    icon = "✓" if ok else "✕"
    print(f"  Ollama  {icon}  {msg}")

    if not ok:
        print()
        print("  To start Ollama:")
        print("    ollama serve")
        print("    ollama pull llama3   # or mistral, codellama, etc.")

    print()
    print("  Opening http://localhost:7860")
    print()

    demo = build_ui_v2()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        favicon_path=None,
    )