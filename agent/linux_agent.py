# agent/linux_agent.py
# Core agent — LangChain 1.x with conversation memory

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

from tools.linux_tools import ALL_TOOLS
from memory.session_memory import session
from config.settings import config
from sandbox.docker_sandbox import sandbox


SYSTEM_PROMPT = """You are a Linux assistant with access to tools.
You help users by answering Linux questions, generating and running shell commands, explaining commands, and recovering from errors.

Rules:
- Always explain commands before running them
- For destructive operations, warn the user explicitly
- If a command fails, use classify_error then fix and retry
- For Q&A questions with no command needed, answer directly
- Keep your final answer clear and user-friendly
- If the user says "yes", "run it", "do it", "go ahead" — execute the previously proposed commands immediately
- Remember the full conversation context when deciding what to do next

Current session context:
{system_context}"""


def classify_intent(user_input: str) -> str:
    inp = user_input.lower().strip()
    # Confirmation words — user is approving a previous proposal
    if inp in ("yes", "y", "run it", "do it", "go ahead", "sure", "ok", "yep", "yeah"):
        return "confirm"
    if any(w in inp for w in ["explain", "what does", "what is", "how does", "break down"]):
        return "explain"
    if any(w in inp for w in ["error", "failed", "permission denied", "not found", "fix"]):
        return "error"
    if any(w in inp for w in ["how do i", "how to", "what command", "which command"]):
        return "qa_or_command"
    if any(w in inp for w in ["run", "execute", "do", "create", "install", "setup", "configure"]):
        return "command"
    return "qa"


def build_agent():
    llm = ChatOllama(
        model=config.model.model_name,
        base_url=config.model.base_url,
        temperature=config.model.temperature,
    )

    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT.format(
            system_context=session.get_context_summary()
        ),
    )

    return agent


class LinuxAssistant:
    def __init__(self):
        self.agent = build_agent()
        self.session = session
        # conversation history — list of {role, content} dicts
        self.history: list[dict] = []

    def run(self, user_input: str) -> dict:
        intent = classify_intent(user_input)

        # Build the message list with full history
        messages = []
        for msg in self.history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessage(content=msg["content"]))

        # Add hint for confirmation intents so agent knows to execute
        if intent == "confirm":
            augmented_input = (
                user_input +
                "\n[The user is confirming the previously proposed commands — execute them now using the execute_command tool]"
            )
        elif config.agent.confirm_before_run and intent in ("command", "qa_or_command"):
            augmented_input = user_input + "\n[Propose the command with explanation before running it]"
        else:
            augmented_input = user_input

        messages.append(HumanMessage(content=augmented_input))

        result = self.agent.invoke({"messages": messages})

        # Extract answer from last AI message
        answer = ""
        result_messages = result.get("messages", [])
        for msg in reversed(result_messages):
            content = getattr(msg, "content", "")
            if content and isinstance(content, str):
                answer = content
                break

        # Save to history (store original input, not augmented)
        self.history.append({"role": "human", "content": user_input})
        self.history.append({"role": "ai", "content": answer})

        # Keep history to last 10 exchanges (20 messages) to avoid context bloat
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return {
            "answer": answer,
            "steps": result_messages,
            "intent": intent,
        }

    def clear_history(self):
        """Clear conversation history — useful between unrelated tasks"""
        self.history = []

    def cleanup(self):
        self.session.save_history()
        sandbox.close_session()