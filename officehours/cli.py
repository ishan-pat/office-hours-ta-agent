"""Chat with the agent in a terminal: `python -m officehours`."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Office Hours TA agent")
    parser.add_argument("--notes", type=Path, default=ROOT / "notes")
    parser.add_argument("--queue", type=Path, default=ROOT / "ta_queue.jsonl",
                        help="where escalations are logged for human TAs")
    parser.add_argument("--trace", action="store_true", help="show tool calls and token use")
    args = parser.parse_args()

    import anthropic  # imported here so tests and --help work without the SDK configured

    from .agent import OfficeHoursAgent

    agent = OfficeHoursAgent(anthropic.Anthropic(), args.notes, args.queue)
    print("Office Hours TA. Ask about the course notes; Ctrl-D to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        r = agent.ask(q)
        print(f"\nta> {r.text}")
        if r.citations:
            print(f"    sources: {', '.join(r.citations)}")
        if r.status == "escalated":
            print(f"    [escalated: {r.escalation_reason}]")
        if args.trace:
            print(f"    trace: {' -> '.join(r.tool_calls)} | {r.steps} model calls, "
                  f"{r.input_tokens} in / {r.output_tokens} out tokens, {r.seconds}s")
        print()
