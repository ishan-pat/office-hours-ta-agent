from pathlib import Path

from officehours.corpus import BM25Index, load_chunks, tokenize
from officehours.sandbox import run_python

NOTES = Path(__file__).resolve().parent.parent / "notes"


def test_chunks_have_stable_ids():
    ids = [c.id for c in load_chunks(NOTES)]
    assert "06_classification_metrics#precision-and-recall" in ids
    assert len(ids) == len(set(ids))


def test_search_ranks_the_right_section_first():
    top = BM25Index(load_chunks(NOTES)).search("precision recall F1 harmonic mean", k=1)[0][0]
    assert top.id.startswith("06_classification_metrics")


def test_search_returns_nothing_for_unrelated_words():
    assert BM25Index(load_chunks(NOTES)).search("transformers self-attention") == []


def test_tokenize_drops_stopwords():
    assert tokenize("What is the learning rate?") == ["learning", "rate"]


def test_sandbox_runs_code():
    ok, out = run_python("print(2 * 0.8 * 0.5 / (0.8 + 0.5))")
    assert ok and out.startswith("0.615")


def test_sandbox_reports_errors_and_timeouts():
    ok, out = run_python("1/0")
    assert not ok and "ZeroDivisionError" in out
    ok, out = run_python("while True: pass")
    assert not ok and "Timed out" in out


def test_sandbox_hides_api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    ok, out = run_python("import os; print(os.environ.get('ANTHROPIC_API_KEY'))")
    assert ok and out == "None"
