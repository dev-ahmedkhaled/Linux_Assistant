# main.py
# Entry point — run the assistant or evaluation

import argparse
import sys
from evaluation.eval import run_eval

def main():
    parser = argparse.ArgumentParser(description="Linux Assistant")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--test-file", type=str, help="NL2Bash test file for evaluation")
    parser.add_argument("--limit", type=int, default=100, help="Eval sample limit")
    parser.add_argument("--model", type=str, help="Override Ollama model name")
    parser.add_argument("--no-confirm", action="store_true", help="Skip command confirmation")
    parser.add_argument("--no-safe", action="store_true", help="Disable safe mode")
    args = parser.parse_args()

    # Apply overrides
    from config.settings import config
    if args.model:
        config.model.model_name = args.model
    if args.no_confirm:
        config.agent.confirm_before_run = False
    if args.no_safe:
        config.agent.safe_mode = False

    if args.eval:
        
        run_eval(nl2bash_test_file=args.test_file, limit=args.limit)
    else:
        from cli.terminal import run
        run()

if __name__ == "__main__":
    main()