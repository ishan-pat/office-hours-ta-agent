"""Drive the agent loop with a scripted fake model: no network, no API key."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

from officehours.agent import OfficeHoursAgent

NOTES = Path(__file__).resolve().parent.parent / "notes"
F1 = "06_classification_metrics#f1-score"


def tool(name, **inp):
    tool.n = getattr(tool, "n", 0) + 1
    return NS(type="tool_use", id=f"tu_{tool.n}", name=name, input=inp)


def text(t):
    return NS(type="text", text=t)


def turn(*blocks, stop="tool_use"):
    return NS(stop_reason=stop, content=list(blocks), usage=NS(input_tokens=100, output_tokens=20))


class FakeClient:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []
        self.messages = self

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw["messages"], default=vars)))
        return self.turns.pop(0)


def make(*turns, queue=None):
    client = FakeClient(*turns)
    return OfficeHoursAgent(client, NOTES, queue), client


def test_grounded_answer_is_delivered():
    agent, _ = make(
        turn(tool("search_notes", query="F1 score")),
        turn(tool("run_python", code="print(2*.8*.5/1.3)")),
        turn(tool("submit_answer", answer="F1 is about 0.615.", citations=[F1])),
    )
    r = agent.ask("precision 0.8 recall 0.5, F1?")
    assert r.status == "answered"
    assert r.citations == [F1]
    assert r.tool_calls == ["search_notes", "run_python", "submit_answer"]
    assert r.steps == 3 and r.input_tokens == 300


def test_citation_that_was_never_retrieved_is_bounced():
    agent, client = make(
        turn(tool("submit_answer", answer="Made up.", citations=[F1])),
        turn(tool("search_notes", query="F1 score")),
        turn(tool("submit_answer", answer="F1 is the harmonic mean.", citations=[F1])),
    )
    r = agent.ask("What is F1?")
    assert r.status == "answered" and r.rejections == 1
    bounced = client.requests[1][-1]["content"][0]
    assert bounced["is_error"] is True and "not retrieved" in bounced["content"]


def test_two_bounces_hand_off_to_a_ta(tmp_path):
    q = tmp_path / "queue.jsonl"
    agent, _ = make(
        turn(tool("submit_answer", answer="a", citations=[])),
        turn(tool("submit_answer", answer="b", citations=["made-up#id"])),
        queue=q,
    )
    r = agent.ask("Explain SVM margins")
    assert r.status == "escalated" and r.escalation_reason == "not_in_notes"
    assert json.loads(q.read_text())["question"] == "Explain SVM margins"


def test_model_escalation_is_logged(tmp_path):
    q = tmp_path / "queue.jsonl"
    agent, _ = make(turn(tool("escalate_to_ta", reason="policy", summary="Extension request",
                              message_to_student="A TA will reply about your extension.")), queue=q)
    r = agent.ask("Can I get an extension?")
    assert r.status == "escalated" and r.text.startswith("A TA will reply")
    assert json.loads(q.read_text())["reason"] == "policy"


def test_plain_text_reply_gets_one_nudge_then_hands_off():
    agent, client = make(turn(text("Sure!"), stop="end_turn"), turn(text("Done."), stop="end_turn"))
    r = agent.ask("hi")
    assert r.status == "escalated" and r.steps == 2
    assert "submit_answer" in client.requests[1][-1]["content"][0]["text"]


def test_refusal_hands_off():
    agent, _ = make(turn(stop="refusal"))
    assert agent.ask("?").status == "escalated"


def test_followup_question_carries_the_owed_tool_result():
    agent, client = make(
        turn(tool("search_notes", query="softmax")),
        turn(tool("submit_answer", answer="Probabilities.",
                  citations=["09_neural_networks#softmax"])),
        turn(tool("submit_answer", answer="Yes, they sum to 1.",
                  citations=["09_neural_networks#softmax"])),
    )
    agent.ask("What does softmax do?")
    r = agent.ask("Do they sum to 1?")
    assert r.status == "answered"
    first = client.requests[2][-1]["content"]
    assert first[0]["type"] == "tool_result" and first[-1]["type"] == "text"
