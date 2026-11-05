"""Course notes: load markdown, split into sections, and rank them with BM25.

Each `## ` section of a notes file becomes one chunk. Its id is
`<file-stem>#<section-slug>`, which is what the agent must cite.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from how i in is it of on or that the this to"
    " was what when which why with you your do does can my".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _slug(text: str) -> str:
    return "-".join(_TOKEN.findall(text.lower()))


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str


def load_chunks(notes_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(notes_dir.glob("*.md")):
        doc_title = path.stem
        section, lines = None, []
        for line in path.read_text().splitlines() + ["## "]:
            if line.startswith("# ") and not line.startswith("## "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                if section and "".join(lines).strip():
                    chunks.append(Chunk(
                        id=f"{path.stem}#{_slug(section)}",
                        title=f"{doc_title} > {section}",
                        text=" ".join(l.strip() for l in lines if l.strip()),
                    ))
                section, lines = line[3:].strip(), []
            else:
                lines.append(line)
    return chunks


class BM25Index:
    """Okapi BM25 over chunk title + text. Small enough to rebuild on every start."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.title + " " + c.text) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored = []
        for chunk, tf, doc in zip(self.chunks, self.tf, self.docs):
            s = 0.0
            for t in terms:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            if s > 0:
                scored.append((chunk, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
