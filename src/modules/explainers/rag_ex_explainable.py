from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from rapidfuzz.distance import Levenshtein as RFLevenshtein
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel, Field
from .explainable_module import ExplainableModule
from ..llm.llm_client import LLMClient


class RAGExAnswer(BaseModel):
    final_answer: str = Field(..., description="The final answer text without additional reasoning.")


@dataclass
class RAGExConfig:
    """
    Central configuration for the RAG-Ex style explainer.
    Only leave-one-token-out is active by default; more strategies can be added later.
    """

    perturbation_strategies: Tuple[str, ...] = ("leave_one_out", "random_noise")
    max_tokens: int | None = None
    embedding_model: str = "all-MiniLM-L6-v2"
    weight_levenshtein: float = 0.0
    weight_semantic: float = 1.0
    aggregation: str = "max"  # how to merge multiple perturbations per token: "max" or "mean"
    importance_normalization: str = "sum"  # "sum" or "max"
    include_query_tokens: bool = False
    # If empty/None, noise tokens will be generated on the fly via LLM.
    noise_tokens: Tuple[str, ...] = ()

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
    Uses leave-one-token-out perturbations on the input and compares perturbed answers
    against the baseline answer via configured comparison strategies.
    """

    def __init__(self, llm_client: LLMClient, config: RAGExConfig | None = None):
        super().__init__()
        self.llm_client = llm_client
        self.config = config or RAGExConfig()
        self._sbert_model: SentenceTransformer | None = None
        self._llm = self.llm_client._init_base_llm()
        self._answer_llm = None
        self._noise_words: List[str] = []

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

    def _generate_noise_words_via_llm(self, k: int = 15) -> List[str]:
        prompt = (
            "Provide a comma-separated list of random, unrelated single words. "
            "Do not number them. Example: apple, river, neon, cobalt"
        )
        try:
            response = self._llm_call(self._llm, prompt)
            text = _extract_answer_text(response)
            raw_tokens = re.split(r"[,\n]", text)
            cleaned = [tok.strip() for tok in raw_tokens if tok.strip()]
            if cleaned:
                return cleaned[:k]
        except Exception:
            pass
        return ["foo", "bar", "baz", "qux", "noise"]

    def _ensure_answer_llm(self) -> None:
        if self._answer_llm is None:
            self._answer_llm = self._llm.with_structured_output(RAGExAnswer, include_raw=True)

    def _answer_prompt(self, query: str, context: str) -> str:
        extra = (
            "Return a JSON object with a single key \"final_answer\".\n"
            "Do not include any explanations, reasoning, or extra fields.\n"
            "Example: {\"final_answer\": \"B: Housing\"}"
        )
        return self.llm_client._create_final_answer_prompt(query, context, extra=extra)

    def _extract_final_answer(self, response: Any) -> str:
        if isinstance(response, RAGExAnswer):
            return response.final_answer
        if isinstance(response, dict):
            parsed = response.get("parsed")
            if isinstance(parsed, RAGExAnswer):
                return parsed.final_answer
            if isinstance(parsed, dict) and "final_answer" in parsed:
                return str(parsed.get("final_answer") or "")
            if "final_answer" in response:
                return str(response.get("final_answer") or "")
        if hasattr(response, "final_answer"):
            return str(getattr(response, "final_answer") or "")
        return _extract_answer_text(response)

    def _sample_noise_word(self) -> str:
        if self.config.noise_tokens:
            pool = list(self.config.noise_tokens)
        else:
            if not self._noise_words:
                self._noise_words = self._generate_noise_words_via_llm()
            pool = self._noise_words
        return random.choice(pool) if pool else "noise"

    def _perturb_random_noise(self, tokens: List[str], token_index: int) -> str:
        noise = self._sample_noise_word()
        new_tokens = list(tokens)
        new_tokens[token_index] = f"{noise} {new_tokens[token_index]} {noise}"
        return "".join(new_tokens)

    def _generate_perturbations(self, tokens: List[str], token_index: int) -> List[Tuple[str, str]]:
        """
        Returns a list of (strategy_name, perturbed_context) for the given token.
        Supports leave_one_out, random_noise.
        """
        perturbations: List[Tuple[str, str]] = []

        if "leave_one_out" in self.config.perturbation_strategies:
            perturbations.append(("leave_one_out", self._perturb_context(tokens, token_index)))

        if "random_noise" in self.config.perturbation_strategies:
            perturbations.append(("random_noise", self._perturb_random_noise(tokens, token_index)))

        return perturbations

    def _aggregate_importance(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        if self.config.aggregation == "mean":
            return sum(scores) / len(scores)
        # default max
        return max(scores)

    def _explain(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Explains the given answer by perturbing the input (question and/or context).
        Returns a dict with token-level importance scores and intermediate similarities.
        """

        query_tokens, query_indices = _tokenize_context(query)
        context_tokens, context_indices = _tokenize_context(context)

        candidates: List[Tuple[str, int]] = []
        if self.config.include_query_tokens:
            candidates.extend([("query", idx) for idx in query_indices])
        candidates.extend([("context", idx) for idx in context_indices])

        if self.config.max_tokens is not None:
            candidates = candidates[: self.config.max_tokens]

        weights = self.config.normalized_weights()
        results: List[Dict[str, Any]] = []
        raw_scores: List[float] = []

        for step, (segment, idx) in enumerate(candidates):
            print(f"Perturbating {step + 1} of {len(candidates)}")

            per_strategy_scores: List[float] = []
            per_strategy_details: List[Dict[str, Any]] = []

            if segment == "query":
                base_tokens = query_tokens
                base_query = query
                base_context = context
            else:
                base_tokens = context_tokens
                base_query = query
                base_context = context

            for strategy_name, perturbed_segment in self._generate_perturbations(base_tokens, idx)[:5]:
                if segment == "query":
                    perturbed_query = perturbed_segment
                    perturbed_context = base_context
                else:
                    perturbed_query = base_query
                    perturbed_context = perturbed_segment

                self._ensure_answer_llm()
                prompt = self._answer_prompt(perturbed_query, perturbed_context)
                response = self._llm_call(self._answer_llm, prompt)
                perturbed_answer = self._extract_final_answer(response)

                lev_sim = _levenshtein_similarity(answer, perturbed_answer)
                sem_sim = self._semantic_similarity(answer, perturbed_answer)
                similarity = (weights[0] * lev_sim) + (weights[1] * sem_sim)
                similarity = max(0.0, min(1.0, similarity))
                importance = max(0.0, 1.0 - similarity)

                per_strategy_scores.append(importance)
                per_strategy_details.append(
                    {
                        "strategy": strategy_name,
                        "perturbed_answer": perturbed_answer,
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
                    "segment": segment,
                    "token": base_tokens[idx],
                    "token_index": idx,
                    "importance_raw": aggregated_importance,
                    "details": per_strategy_details,
                }
            )

        if self.config.importance_normalization == "sum":
            denom = sum(raw_scores)
        else:
            denom = max(raw_scores) if raw_scores else 0.0

        if denom > 0:
            for item in results:
                item["importance"] = item["importance_raw"] / denom
        else:
            for item in results:
                item["importance"] = 0.0

        return {
            "query": query,
            "baseline_answer": answer,
            "context": context,
            "results": results,
        }

    @staticmethod
    def prettify(explanation: Dict[str, Any]) -> str:
        """
        Returns a formatted string listing tokens with their normalized importance and per-strategy details.
        """
        lines = ["segment\ttoken\timportance\tstrategies"]
        for item in explanation.get("results", []):
            token_display = item.get("token", "").replace("\n", "\\n")
            segment = item.get("segment", "")
            importance = item.get("importance", 0.0)
            details = item.get("details", [])
            detail_parts = []
            for d in details:
                detail_parts.append(
                    f"{d.get('strategy')}:raw={d.get('importance_raw', 0.0):.3f},sim={d.get('combined_similarity', 0.0):.3f}"
                )
            lines.append(f"{segment}\t{token_display}\t{importance:.3f}\t{' | '.join(detail_parts)}")

        return "\n".join(lines)
