from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import re

from src.modules.hotpot_baseline_runner import HotpotBaselineRunner, RetrievedChunk


@dataclass(frozen=True)
class TextPerturbation:
    start: int
    end: int
    removed_preview: str
    perturbed_text: str


class RAGExTextPerturber:
    def __init__(self, window_tokens: int = 30, max_perturbations: int = 40):
        self.window_tokens = window_tokens
        self.max_perturbations = max_perturbations

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\S+", text or "")

    def _detokenize(self, toks: List[str]) -> str:
        return " ".join(toks)

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for c in chunks:
            parts.append(f"[{c.chunk_id}] {c.title} :: {c.content}")
        return "\n".join(parts)

    def perturb(self, context: str) -> List[TextPerturbation]:
        toks = self._tokenize(context)
        if not toks:
            return []

        w = max(1, int(self.window_tokens))
        positions = list(range(0, len(toks), w))
        positions = positions[: self.max_perturbations]

        out: List[TextPerturbation] = []
        for s in positions:
            e = min(len(toks), s + w)
            kept = toks[:s] + toks[e:]
            removed = self._detokenize(toks[s:e])[:120]
            out.append(TextPerturbation(start=s, end=e, removed_preview=removed, perturbed_text=self._detokenize(kept)))
        return out

    def answer_with_context_override(self, runner: HotpotBaselineRunner, question: str, context_override: str) -> tuple[str, int, int, int]:
        fake_chunk = RetrievedChunk(
            chunk_id="ctx-1",
            score=1.0,
            title="Perturbed Context",
            content=context_override,
            metadata={"source": "rag-ex"},
        )
        return runner.answer(question=question, retrieved=[fake_chunk])
