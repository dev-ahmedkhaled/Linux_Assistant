# cli/terminal.py
# Terminal UI for the Linux assistant
# Clean, minimal interface with color output

import sys
import signal
import readline  # enables arrow keys + history in input()
from agent.linux_agent import LinuxAssistant
from config.settings import config

# ─────────────────────────────────────────────
# ANSI Colors
# ─────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[32m"
    CYAN    = "\033[36m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    WHITE   = "\033[97m"

def print_banner():
    banner = f"""
{C.CYAN}{C.BOLD}
  ██╗     ██╗███╗   ██╗██╗   ██╗██╗  ██╗
  ██║     ██║████╗  ██║██║   ██║╚██╗██╔╝
  ██║     ██║██╔██╗ ██║██║   ██║ ╚███╔╝ 
  ██║     ██║██║╚██╗██║██║   ██║ ██╔██╗ 
  ███████╗██║██║ ╚████║╚██████╔╝██╔╝ ██╗
  ╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝
{C.RESET}{C.DIM}  Linux Assistant — powered by {config.model.model_name}{C.RESET}
"""
    print(banner)

def print_help():
    print(f"""
{C.BOLD}Commands:{C.RESET}
  {C.CYAN}:help{C.RESET}       Show this help
  {C.CYAN}:history{C.RESET}    Show command history this session
  {C.CYAN}:sysinfo{C.RESET}    Show system information
  {C.CYAN}:clear{C.RESET}      Clear the screen
  {C.CYAN}:safe{C.RESET}       Toggle safe mode (currently {'ON' if config.agent.safe_mode else 'OFF'})
  {C.CYAN}:exit{C.RESET}       Exit the assistant
  {C.CYAN}:reset{C.RESET}      Clear conversation history

{C.BOLD}Example queries:{C.RESET}
  find all python files modified today
  how do I check which process is using port 8080?
  explain: tar -czf archive.tar.gz ./folder
  set up a python venv and install flask
""")

def print_user(text: str):
    print(f"\n{C.BOLD}{C.GREEN}you{C.RESET} {C.DIM}›{C.RESET} {text}")

def print_assistant(text: str):
    print(f"\n{C.BOLD}{C.CYAN}linux{C.RESET} {C.DIM}›{C.RESET} {text}\n")

def print_thinking():
    print(f"\n{C.DIM}  thinking...{C.RESET}")

def print_error(text: str):
    print(f"\n{C.RED}  ✗ {text}{C.RESET}")

def print_info(text: str):
    print(f"\n{C.DIM}  {text}{C.RESET}")

def confirm(prompt: str) -> bool:
    """Ask user for y/n confirmation"""
    try:
        ans = input(f"\n{C.YELLOW}  ⚠ {prompt} [y/N] {C.RESET}").strip().lower()
        return ans in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


# ─────────────────────────────────────────────
# Handle special CLI commands
# ─────────────────────────────────────────────

def handle_special(cmd: str, assistant: LinuxAssistant) -> bool:
    """Returns True if it was a special command"""
    cmd = cmd.strip().lower()

    if cmd == ":help":
        print_help()
        return True
    if cmd == ":history":
        recent = assistant.session.get_recent_commands(10)
        if not recent:
            print_info("No commands yet.")
        else:
            for e in recent:
                icon = f"{C.GREEN}✓{C.RESET}" if e.exit_code == 0 else f"{C.RED}✗{C.RESET}"
                print(f"  {icon} {e.command}")
        return True
    if cmd == ":sysinfo":
        print_assistant(assistant.session.get_context_summary())
        return True
    if cmd == ":clear":
        print("\033[2J\033[H", end="")
        print_banner()
        return True
    if cmd == ":safe":
        config.agent.safe_mode = not config.agent.safe_mode
        state = f"{C.GREEN}ON{C.RESET}" if config.agent.safe_mode else f"{C.RED}OFF{C.RESET}"
        print_info(f"Safe mode: {state}")
        return True
    if cmd == ":reset":
        assistant.clear_history()
        print_info("Conversation history cleared.")
        return True
    if cmd in (":exit", ":quit", ":q"):
        raise KeyboardInterrupt
    return False


# ─────────────────────────────────────────────
# Main REPL
# ─────────────────────────────────────────────

def run():
    print_banner()
    print_info("Type :help for commands, :exit to quit\n")

    assistant = LinuxAssistant()

    def on_exit(sig=None, frame=None):
        print(f"\n\n{C.DIM}  Saving session... bye!{C.RESET}\n")
        assistant.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    while True:
        try:
            user_input = input(f"{C.BOLD}{C.MAGENTA}❯{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            on_exit()

        if not user_input:
            continue

        if handle_special(user_input, assistant):
            continue

        print_thinking()

        try:
            result = assistant.run(user_input)
            print_assistant(result["answer"])
        except Exception as e:
            print_error(f"Agent error: {e}")
            if config.agent.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    run()