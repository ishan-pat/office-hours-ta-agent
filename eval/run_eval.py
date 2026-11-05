"""Score the agent on eval/cases.jsonl.

    python eval/run_eval.py --retrieval-only   # offline: does search find the right notes?
    python eval/run_eval.py                    # live: runs the full agent (needs ANTHROPIC_API_KEY)

Grading is deterministic on purpose (no LLM judge): did it route correctly
(answer vs. escalate), did it cite the right notes file, does the answer contain
the key fact or number, and did it use the tool we expected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from officehours.corpus import BM25Index, load_chunks  # noqa: E402


def load_cases() -> list[dict]:
    return [json.loads(l) for l in (ROOT / "eval/cases.jsonl").read_text().splitlines() if l.strip()]


def retrieval_only(cases: list[dict], k: int = 3) -> None:
    index = BM25Index(load_chunks(ROOT / "notes"))
    rows = [c for c in cases if c["expect"] == "answer"]
    hits = 0
    for c in rows:
        got = [ch.id.split("#")[0] for ch, _ in index.search(c["question"], k=k)]
        ok = any(s in got for s in c["sources"])
        hits += ok
        print(f"{'PASS' if ok else 'MISS'}  {c['id']:<20} top{k}: {got}")
    print(f"\nRetrieval recall@{k} on the raw question text: {hits}/{len(rows)}")


def grade(case: dict, r) -> dict:
    checks = {"route": r.status == ("answered" if case["expect"] == "answer" else "escalated")}
    if case["expect"] == "answer":
        cited = {c.split("#")[0] for c in r.citations}
        checks["cited_source"] = any(s in cited for s in case["sources"])
        text = r.text.lower()
        checks["key_fact"] = any(s.lower() in text for s in case["include_any"])
        if "tool" in case:
            checks["used_tool"] = case["tool"] in r.tool_calls
    else:
        checks["reason"] = r.escalation_reason == case["reason"]
    return checks


def live(cases: list[dict], only: str | None) -> None:
    import anthropic

    from officehours.agent import OfficeHoursAgent

    client = anthropic.Anthropic()
    out_dir = ROOT / "eval/results"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for c in cases:
        if only and c["id"] != only:
            continue
        agent = OfficeHoursAgent(client, ROOT / "notes")  # fresh memory per case
        r = agent.ask(c["question"])
        checks = grade(c, r)
        rows.append({"id": c["id"], "expect": c["expect"], "status": r.status,
                     "checks": checks, "tool_calls": r.tool_calls, "rejections": r.rejections,
                     "steps": r.steps, "input_tokens": r.input_tokens,
                     "output_tokens": r.output_tokens, "seconds": r.seconds,
                     "answer": r.text, "citations": r.citations,
                     "escalation_reason": r.escalation_reason})
        mark = "PASS" if all(checks.values()) else "FAIL"
        print(f"{mark}  {c['id']:<20} {r.status:<9} {checks}  [{' -> '.join(r.tool_calls)}]")

    n = len(rows)
    if not n:
        return
    def rate(key):
        vals = [row["checks"][key] for row in rows if key in row["checks"]]
        return f"{sum(vals)}/{len(vals)}" if vals else "-"
    summary = {
        "cases": n,
        "all_checks_pass": f"{sum(all(r['checks'].values()) for r in rows)}/{n}",
        "routing": rate("route"),
        "cited_right_source": rate("cited_source"),
        "key_fact_present": rate("key_fact"),
        "used_expected_tool": rate("used_tool"),
        "escalation_reason_match": rate("reason"),
        "answers_bounced_by_citation_check": sum(r["rejections"] for r in rows),
        "avg_model_calls": round(sum(r["steps"] for r in rows) / n, 2),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows),
    }
    print("\n" + json.dumps(summary, indent=2))
    path = out_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--case", help="run a single case id")
    a = p.parse_args()
    cases = load_cases()
    retrieval_only(cases) if a.retrieval_only else live(cases, a.case)
