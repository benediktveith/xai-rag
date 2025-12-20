from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from rapidfuzz.distance import Levenshtein as RFLevenshtein
from sentence_transformers import SentenceTransformer, util
from .explainable_module import ExplainableModule
from .llm_client import LLMClient


@dataclass
class RAGExConfig:
    """
    Central configuration for the RAG-Ex style explainer.
    Only leave-one-token-out is active by default; more strategies can be added later.
    """

    perturbation_strategies: Tuple[str, ...] = ("leave_one_out",)
    max_tokens: int | None = None
    embedding_model: str = "all-MiniLM-L6-v2"
    weight_levenshtein: float = 0.4
    weight_semantic: float = 0.6
    aggregation: str = "max"  # how to merge multiple perturbations per token: "max" or "mean"

    def normalized_weights(self) -> Tuple[float, float]:
        total = self.weight_levenshtein + self.weight_semantic
        if total <= 0:
            return 0.5, 0.5
        return self.weight_levenshtein / total, self.weight_semantic / total


def _levenshtein_similarity(a: str, b: str) -> float:
    sim = RFLevenshtein.normalized_similarity(a, b)  # typically 0..1, but guard if 0..100
    if sim > 1.0:
        sim = sim / 100.0
    return max(0.0, min(1.0, sim))


def _tokenize_context(text: str) -> Tuple[List[str], List[int]]:
    """
    Splits the context into tokens while keeping whitespace tokens.
    Returns the full token list and the indices that correspond to non-whitespace tokens.
    """

    tokens = re.split(r"(\s+)", text)
    non_space_indices = [i for i, tok in enumerate(tokens) if tok and not tok.isspace()]
    return tokens, non_space_indices


def _extract_answer_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "content" in response:
            return str(response.get("content") or "")
        if "text" in response:
            return str(response.get("text") or "")
    if hasattr(response, "content"):
        return str(getattr(response, "content") or "")
    return str(response)


class RAGExExplainable(ExplainableModule):
    """
    Perturbation-based explainer following the RAG-Ex paper.
    Uses leave-one-word-out perturbations on the context and compares perturbed answers
    against the baseline answer via Levenshtein + SBERT cosine similarity.
    """

    def __init__(self, llm_client: LLMClient, config: RAGExConfig | None = None):
        self.llm_client = llm_client
        self.config = config or RAGExConfig()
        self._sbert_model: SentenceTransformer | None = None
        self._llm = self.llm_client.get_llm()

    def _ensure_sbert(self) -> None:
        if self._sbert_model or SentenceTransformer is None:
            return
        self._sbert_model = SentenceTransformer(self.config.embedding_model)

    def _semantic_similarity(self, base_answer: str, perturbed_answer: str) -> float:
        if not base_answer and not perturbed_answer:
            return 1.0
        self._ensure_sbert()
        if not self._sbert_model:
            return 0.0

        embeddings = self._sbert_model.encode([base_answer, perturbed_answer])
        score = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
        return max(0.0, min(1.0, score))

    def _perturb_context(self, tokens: List[str], drop_index: int) -> str:
        return "".join(tokens[:drop_index] + tokens[drop_index + 1 :])

    def _generate_perturbations(self, tokens: List[str], token_index: int) -> List[Tuple[str, str]]:
        """
        Returns a list of (strategy_name, perturbed_context) for the given token.
        Only leave_one_out is implemented now; extend here for additional strategies.
        """
        perturbations: List[Tuple[str, str]] = []

        if "leave_one_out" in self.config.perturbation_strategies:
            perturbations.append(("leave_one_out", self._perturb_context(tokens, token_index)))

        return perturbations

    def _aggregate_importance(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        if self.config.aggregation == "mean":
            return sum(scores) / len(scores)
        # default max
        return max(scores)

    def explain(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Explains the given answer by perturbing the context (word-level leave-one-out).
        Returns a dict with token-level importance scores and intermediate similarities.
        """

        tokens, candidate_indices = _tokenize_context(context)

        if self.config.max_tokens is not None:
            candidate_indices = candidate_indices[: self.config.max_tokens]

        weights = self.config.normalized_weights()
        results: List[Dict[str, Any]] = []
        raw_scores: List[float] = []

        for id, idx in enumerate(candidate_indices):
            print(f'Pertubating {id} of {len(candidate_indices)}')

            per_strategy_scores: List[float] = []
            per_strategy_details: List[Dict[str, Any]] = []

            for strategy_name, perturbed_context in self._generate_perturbations(tokens, idx):
                prompt = self.llm_client._create_final_answer_prompt(query, perturbed_context)
                response = self._llm.invoke(prompt)
                perturbed_answer = _extract_answer_text(response)

                lev_sim = _levenshtein_similarity(answer, perturbed_answer)
                sem_sim = self._semantic_similarity(answer, perturbed_answer)
                similarity = (weights[0] * lev_sim) + (weights[1] * sem_sim)
                similarity = max(0.0, min(1.0, similarity))
                importance = max(0.0, 1.0 - similarity)

                per_strategy_scores.append(importance)
                per_strategy_details.append(
                    {
                        "strategy": strategy_name,
                        "levenshtein_similarity": lev_sim,
                        "semantic_similarity": sem_sim,
                        "combined_similarity": similarity,
                        "importance_raw": importance,
                    }
                )

            aggregated_importance = self._aggregate_importance(per_strategy_scores)
            raw_scores.append(aggregated_importance)
            results.append(
                {
                    "token": tokens[idx],
                    "token_index": idx,
                    "importance_raw": aggregated_importance,
                    "details": per_strategy_details,
                }
            )

        max_score = max(raw_scores) if raw_scores else 0.0
        if max_score > 0:
            for item in results:
                item["importance"] = item["importance_raw"] / max_score
        else:
            for item in results:
                item["importance"] = 0.0

        return {
            "baseline_answer": answer,
            "context": context,
            "results": results,
        }
