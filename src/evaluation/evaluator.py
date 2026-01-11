from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


class Evaluator:
    """
    Jaccard-based evaluator for comparing generated answers with a baseline answer.
    Accuracy defaults to mean Jaccard unless a threshold is provided.
    """

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        return text.lower() if self.lowercase else text

    def _tokenize(self, text: str) -> List[str]:
        normalized = self._normalize(text)
        return re.findall(r"\b\w+\b", normalized)

    def jaccard(self, a: str, b: str) -> float:
        a_tokens = set(self._tokenize(a))
        b_tokens = set(self._tokenize(b))
        if not a_tokens and not b_tokens:
            return 1.0
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    def f1(self, a: str, b: str) -> float:
        a_tokens = set(self._tokenize(a))
        b_tokens = set(self._tokenize(b))
        if not a_tokens and not b_tokens:
            return 1.0
        if not a_tokens or not b_tokens:
            return 0.0
        intersection = len(a_tokens & b_tokens)
        precision = intersection / len(a_tokens)
        recall = intersection / len(b_tokens)
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def evaluate(
        self,
        answers: Sequence[str],
        baseline_answer: str,
        jaccard_threshold: float | None = None,
    ) -> Dict[str, Any]:
        details: List[Dict[str, Any]] = []
        jaccard_scores: List[float] = []
        f1_scores: List[float] = []
        correct_flags: List[int] = []

        for answer in answers:
            jacc = self.jaccard(answer, baseline_answer)
            f1 = self.f1(answer, baseline_answer)
            jaccard_scores.append(jacc)
            f1_scores.append(f1)
            if jaccard_threshold is not None:
                correct_flags.append(1 if jacc >= jaccard_threshold else 0)

            details.append(
                {
                    "answer": answer,
                    "jaccard": jacc,
                    "f1": f1,
                }
            )

        mean_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        if jaccard_threshold is None:
            accuracy = mean_jaccard
        else:
            accuracy = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0

        return {
            "accuracy": accuracy,
            "f1": mean_f1,
            "mean_jaccard": mean_jaccard,
            "details": details,
        }
