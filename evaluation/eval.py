# evaluation/eval.py
# Evaluate the assistant on NL2Bash and agent task benchmarks

import json
import re
from dataclasses import dataclass, field
from typing import Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

@dataclass
class EvalResult:
    total: int = 0
    exact_match: int = 0
    bleu_scores: list = field(default_factory=list)
    success_rate: int = 0
    errors: list = field(default_factory=list)

    @property
    def exact_match_pct(self) -> float:
        return (self.exact_match / self.total * 100) if self.total else 0

    @property
    def avg_bleu(self) -> float:
        return sum(self.bleu_scores) / len(self.bleu_scores) if self.bleu_scores else 0

    @property
    def success_pct(self) -> float:
        return (self.success_rate / self.total * 100) if self.total else 0

    def report(self) -> str:
        return (
            f"Total samples:   {self.total}\n"
            f"Exact match:     {self.exact_match_pct:.1f}% ({self.exact_match}/{self.total})\n"
            f"Avg BLEU:        {self.avg_bleu:.4f}\n"
            f"Success rate:    {self.success_pct:.1f}%\n"
        )


def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()
    smoother = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)


def normalize_command(cmd: str) -> str:
    cmd = cmd.strip()
    cmd = re.sub(r"\s+", " ", cmd)
    return cmd


def extract_command_from_answer(answer: str) -> str:
    code_match = re.search(r"```(?:bash|shell)?\n?(.*?)```", answer, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    action_match = re.search(r"<action>(.*?)</action>", answer, re.DOTALL)
    if action_match:
        return action_match.group(1).strip()
    for line in answer.split("\n"):
        line = line.strip()
        if line and not line.endswith("?") and not line[0].isupper():
            return line
    return answer.strip()


# ─────────────────────────────────────────────
# NL2Bash Evaluator
# ─────────────────────────────────────────────

class NL2BashEvaluator:
    def __init__(self, assistant):
        self.assistant = assistant

    def evaluate(self, test_file: str, limit: int = 100) -> EvalResult:
        result = EvalResult()

        with open(test_file) as f:
            samples = json.load(f) if test_file.endswith(".json") else [
                json.loads(l) for l in f if l.strip()
            ]

        samples = samples[:limit]
        result.total = len(samples)

        for i, sample in enumerate(samples):
            nl = sample.get("nl_command") or sample.get("prompt") or ""
            reference = sample.get("bash_code") or sample.get("response") or ""
            if not nl or not reference:
                continue
            try:
                output = self.assistant.run(nl)
                predicted = extract_command_from_answer(output["answer"])
                if normalize_command(predicted) == normalize_command(reference):
                    result.exact_match += 1
                bleu = compute_bleu(reference, predicted)
                result.bleu_scores.append(bleu)
                if i % 10 == 0:
                    print(f"  [{i+1}/{result.total}] BLEU: {bleu:.3f} | {nl[:50]}")
            except Exception as e:
                result.errors.append(f"Sample {i}: {e}")

        return result


# ─────────────────────────────────────────────
# Agent Task Evaluator
# ─────────────────────────────────────────────

class AgentTaskEvaluator:
    BENCHMARK_TASKS = [
        {"query": "list all files larger than 10MB in the current directory", "expected_keywords": ["find", "size", "+10M"]},
        {"query": "show the top 5 processes by CPU usage", "expected_keywords": ["ps", "top", "cpu"]},
        {"query": "count how many lines are in all .py files in the current directory", "expected_keywords": ["find", "wc", ".py"]},
        {"query": "show disk usage of each directory in /home sorted by size", "expected_keywords": ["du", "sort"]},
        {"query": "find all files modified in the last 24 hours", "expected_keywords": ["find", "mtime", "-1"]},
    ]

    def __init__(self, assistant):
        self.assistant = assistant

    def evaluate(self, tasks=None) -> EvalResult:
        tasks = tasks or self.BENCHMARK_TASKS
        result = EvalResult()
        result.total = len(tasks)

        for task in tasks:
            try:
                output = self.assistant.run(task["query"])
                answer = output["answer"].lower()
                steps = output.get("steps", [])
                keywords_hit = sum(
                    1 for kw in task["expected_keywords"]
                    if kw.lower() in answer or any(kw.lower() in str(s) for s in steps)
                )
                if keywords_hit >= len(task["expected_keywords"]) * 0.6:
                    result.success_rate += 1
                print(f"  ✓ {task['query'][:50]} — {keywords_hit}/{len(task['expected_keywords'])} keywords")
            except Exception as e:
                result.errors.append(str(e))
                print(f"  ✗ {task['query'][:50]} — ERROR: {e}")

        return result


# ─────────────────────────────────────────────
# Run full evaluation
# ─────────────────────────────────────────────

def run_eval(nl2bash_test_file: str = None, limit: int = 100):
    # lazy import to avoid circular import at module level
    from agent.linux_agent import LinuxAssistant

    print("=" * 50)
    print("Linux Assistant Evaluation")
    print("=" * 50)

    assistant = LinuxAssistant()

    if nl2bash_test_file:
        print(f"\n[1/2] NL2Bash evaluation (n={limit})...")
        nl_result = NL2BashEvaluator(assistant).evaluate(nl2bash_test_file, limit=limit)
        print("\nNL2Bash Results:")
        print(nl_result.report())

    print("\n[2/2] Agent task evaluation...")
    agent_result = AgentTaskEvaluator(assistant).evaluate()
    print("\nAgent Task Results:")
    print(agent_result.report())

    assistant.cleanup()


if __name__ == "__main__":
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_eval(nl2bash_test_file=test_file)