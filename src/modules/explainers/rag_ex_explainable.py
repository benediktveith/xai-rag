from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from rapidfuzz.distance import Levenshtein as RFLevenshtein
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pydantic import BaseModel, Field
from .explainable_module import ExplainableModule
from ..llm.llm_client import LLMClient

class RAGExAnswer(BaseModel):
    final_answer: str = Field(..., description="The final answer text without additional reasoning.")

@dataclass
class RAGExConfig:
    """
    Central configuration for the RAG-Ex style explainer.
    """

    perturbation_strategies: Tuple[str, ...] = ("leave_one_out", "random_noise")
    max_tokens: int | None = None
    embedding_model: str = "all-MiniLM-L6-v2"
    weight_levenshtein: float = 0.4
    weight_semantic: float = 0.6
    aggregation: str = "max" # "mean" or "max"
    importance_normalization: str = "max"  # "sum" or "max"
    include_query_tokens: bool = False
    pertubation_depth: int = 1
    pertubation_mode: str = 'word'  # 'word' or 'sentences' or 'paragraphs'
    noise_tokens: Tuple[str, ...] = ()
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    enable_nli_evaluation: bool = True

    def normalized_weights(self) -> Tuple[float, float]:
        total = self.weight_levenshtein + self.weight_semantic
        if total <= 0:
            return 0.5, 0.5
        return self.weight_levenshtein / total, self.weight_semantic / total


def _levenshtein_similarity(a: str, b: str) -> float:
    sim = RFLevenshtein.normalized_similarity(a, b)
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

def _tokenize_paragraphs(text: str) -> Tuple[List[str], List[int]]:
    parts = re.split(r'(\n{2,})', text)
    
    tokens = []
    paragraph_indices = []
    
    for part in parts:
        if not part:
            continue

        if re.match(r'^\n{2,}$', part):
            tokens.append(part)
        else:
            tokens.append(part)
            paragraph_indices.append(len(tokens) - 1)
    
    return tokens, paragraph_indices

def _tokenize_sentences(text: str) -> Tuple[List[str], List[int]]:
    pattern = r'([^.!?\n]+[.!?]+|[^.!?\n]+)(\s*)'
    
    tokens = []
    sentence_indices = []
    
    for match in re.finditer(pattern, text):
        sentence_part = match.group(1).strip()
        whitespace_part = match.group(2)
        
        if sentence_part:
            tokens.append(sentence_part)
            sentence_indices.append(len(tokens) - 1)
            
        if whitespace_part:
            tokens.append(whitespace_part)
    
    return tokens, sentence_indices

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
    """

    def __init__(self, llm_client: LLMClient, config: RAGExConfig | None = None):
        super().__init__()
        self.llm_client = llm_client
        self.config = config or RAGExConfig()
        self._sbert_model: SentenceTransformer | None = None
        self._llm = self.llm_client._init_base_llm()
        self._answer_llm = None
        self._noise_words: List[str] = []
        self._nli_tokenizer = None
        self._nli_model = None
        self._device = "cpu"

    def _ensure_sbert(self) -> None:
        if self._sbert_model or SentenceTransformer is None:
            return
        self._sbert_model = SentenceTransformer(self.config.embedding_model)

    def _ensure_nli_model(self) -> None:
        """Initialize NLI model for explanation quality evaluation."""
        if self._nli_tokenizer is None or self._nli_model is None:
            self._nli_tokenizer = AutoTokenizer.from_pretrained(self.config.nli_model)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(self.config.nli_model)
            self._nli_model.to(self._device)
            self._nli_model.eval()

    def _evaluate_nli(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Evaluate if the hypothesis (answer) can be derived from the premise (context).
        
        Args:
            premise: The context text
            hypothesis: The answer text
            
        Returns:
            Dictionary with NLI scores and label
        """
        if not self.config.enable_nli_evaluation:
            return {}
        
        self._ensure_nli_model()
        
        inputs = self._nli_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self._nli_model(**inputs)
        
        prediction = torch.softmax(output.logits[0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        label_scores = {name: float(score) for score, name in zip(prediction, label_names)}
        
        predicted_label = max(label_scores, key=label_scores.get)
        entailment_score = label_scores["entailment"]
        
        return {
            "nli_entailment": entailment_score,
            "nli_neutral": label_scores["neutral"],
            "nli_contradiction": label_scores["contradiction"],
            "nli_label": predicted_label,
            "can_explain": predicted_label == "entailment" and entailment_score > 0.5
        }

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
        """
        Leave-one-out perturbation.
        - In 'word' mode: removes the word at drop_index and the next (depth-1) words
        - In 'sentences' mode: removes the entire sentence at drop_index
        """
        depth = self.config.pertubation_depth
        
        if self.config.pertubation_mode == 'sentences':
            return "".join(tokens[:drop_index] + tokens[drop_index + 1:])
        
        return "".join(tokens[:drop_index] + tokens[drop_index + depth:])


    def _generate_noise_words_via_llm(self, k: int = 25) -> List[str]:
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
            "You are only allowed to answer based on the provided context.\n"
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
        """
        Random noise perturbation.
        - In 'word' mode: inserts noise words around the target word
        - In 'sentences' mode: inserts pertubation_depth random words at random positions in the sentence
        """
        new_tokens = list(tokens)
        
        if self.config.pertubation_mode == 'sentences':
            sentence = tokens[token_index]
            words = sentence.split()
            
            for _ in range(self.config.pertubation_depth):
                noise_word = self._sample_noise_word()
                if words:
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, noise_word)
                else:
                    words.append(noise_word)
            
            new_tokens[token_index] = ' '.join(words)
        else:
            noise_words = [self._sample_noise_word() for _ in range(self.config.pertubation_depth)]
            new_tokens[token_index] = f"{' '.join(noise_words)} {new_tokens[token_index]} {' '.join(noise_words)}"
        
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

        return max(scores)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r"\b\w+\b", text.lower())

    def _calculate_interpretability(
        self,
        results: List[Dict[str, Any]],
        ground_truth_evidence: str | None = None,
    ) -> Dict[str, Any]:
        """
        Calculate interpretability using Jaccard coefficient.
        Compares the token with highest feature importance against ground truth evidence.
        
        Args:
            results: List of token results with importance scores
            ground_truth_evidence: Expert-annotated ground truth evidence (optional)
            
        Returns:
            Dictionary with interpretability scores
        """
        if not ground_truth_evidence:
            return {
                "interpretability_score": None,
                "explanation": "No ground truth evidence provided"
            }
        
        if not results:
            return {
                "interpretability_score": 0.0,
                "jaccard_score": 0.0,
                "top_token": "",
                "ground_truth_evidence": ground_truth_evidence,
                "explanation": "No results generated"
            }
        
        # Find token with highest importance
        top_result = max(results, key=lambda x: x.get("importance", 0.0))
        top_token = top_result.get("token", "")
        top_importance = top_result.get("importance", 0.0)
        
        # Tokenize both top token and ground truth evidence
        token_tokens = set(self._tokenize(top_token))
        truth_tokens = set(self._tokenize(ground_truth_evidence))
        
        # Calculate Jaccard coefficient: |A ∩ B| / |A ∪ B|
        if not token_tokens and not truth_tokens:
            jaccard_score = 1.0
        elif not token_tokens or not truth_tokens:
            jaccard_score = 0.0
        else:
            intersection = len(token_tokens & truth_tokens)
            union = len(token_tokens | truth_tokens)
            jaccard_score = intersection / union if union > 0 else 0.0
        
        return {
            "interpretability_score": float(jaccard_score),
            "jaccard_score": float(jaccard_score),
            "top_token": top_token,
            "top_importance": float(top_importance),
            "ground_truth_evidence": ground_truth_evidence,
            "intersection_size": len(token_tokens & truth_tokens) if token_tokens and truth_tokens else 0,
            "union_size": len(token_tokens | truth_tokens) if token_tokens and truth_tokens else 0
        }

    def _explain(self, query: str, answer: str, context: str, ground_truth_evidence: str | None = None) -> Dict[str, Any]:
        """
        Explains the given answer by perturbing the input (question and/or context).
        Returns a dict with token-level importance scores and intermediate similarities.
        """

        if self.config.pertubation_mode == 'sentences':
            tokenize_fn = _tokenize_sentences
        elif self.config.pertubation_mode == 'paragraphs':
            tokenize_fn = _tokenize_paragraphs
        else:
            tokenize_fn = _tokenize_context
        
        query_tokens, query_indices = tokenize_fn(query)
        context_tokens, context_indices = tokenize_fn(context)

        candidates: List[Tuple[str, int]] = []
        if self.config.include_query_tokens:
            candidates.extend([("query", idx) for idx in query_indices])
        candidates.extend([("context", idx) for idx in context_indices])

        if self.config.max_tokens is not None:
            candidates = candidates[: self.config.max_tokens]

        weights = self.config.normalized_weights()
        results: List[Dict[str, Any]] = []
        raw_scores: List[float] = []

        try:
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

                for strategy_name, perturbed_segment in self._generate_perturbations(base_tokens, idx):
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

                    nli_scores = self._evaluate_nli(perturbed_context, answer)

                    per_strategy_scores.append(importance)
                    detail_entry = {
                        "strategy": strategy_name,
                        "perturbed_answer": perturbed_answer,
                        "perturbed_context": perturbed_context,
                        "levenshtein_similarity": lev_sim,
                        "semantic_similarity": sem_sim,
                        "combined_similarity": similarity,
                        "importance_raw": importance,
                    }

                    detail_entry.update(nli_scores)
                    per_strategy_details.append(detail_entry)

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

            for item in results:
                item["importance"] = item["importance_raw"]

            if denom > 0:
                for item in results:
                    item["importance"] = item["importance_raw"] / denom
            else:
                for item in results:
                    item["importance"] = 0.0
        except Exception as e:
            print(f"Error during explanation: {e}")
            pass

        result_dict = {
            "query": query,
            "baseline_answer": answer,
            "context": context,
            "results": results,
        }
        
        if ground_truth_evidence:
            result_dict["interpretability"] = self._calculate_interpretability(results, ground_truth_evidence)
        
        return result_dict

    @staticmethod
    def prettify(explanation: Dict[str, Any]) -> str:
        """
        Returns a formatted string listing tokens with their normalized importance and per-strategy details including NLI scores.
        """
        lines = ["=" * 100]
        lines.append("RAG-Ex Explanation Results")
        lines.append("=" * 100)
        lines.append("")
        
        if "interpretability" in explanation:
            interp = explanation["interpretability"]
            score = interp.get('interpretability_score')
            if score is not None:
                lines.append("🎯 Interpretability (Feature Importance vs Ground Truth):")
                lines.append(f"  • Jaccard Score: {score:.4f}")
                if 'top_importance' in interp:
                    lines.append(f"  • Top Importance: {interp['top_importance']:.4f}")
                if 'intersection_size' in interp:
                    lines.append(f"  • Intersection: {interp['intersection_size']} tokens")
                if 'union_size' in interp:
                    lines.append(f"  • Union: {interp['union_size']} tokens")
                lines.append("")
        
        for item in explanation.get("results", []):
            token_display = item.get("token", "").replace("\n", "\\n")
            segment = item.get("segment", "")
            importance = item.get("importance", 0.0)
            token_index = item.get("token_index", 0)
            
            if len(token_display) > 80:
                token_display = token_display[:77] + "..."
            
            lines.append(f"[{segment.upper()}] Token {token_index}: \"{token_display}\"")
            lines.append(f"  Importance: {importance:.4f}")
            
            details = item.get("details", [])
            if details:
                lines.append("  Per-Strategy Details:")
                for d in details:
                    strategy = d.get('strategy', 'unknown')
                    imp_raw = d.get('importance_raw', 0.0)
                    sim = d.get('combined_similarity', 0.0)
                    
                    lines.append(f"    • {strategy}:")
                    lines.append(f"      - Importance (raw): {imp_raw:.4f}")
                    lines.append(f"      - Similarity: {sim:.4f}")
                    
                    if 'nli_label' in d:
                        nli_ent = d.get('nli_entailment', 0.0)
                        nli_label = d.get('nli_label', 'N/A')
                        can_explain = d.get('can_explain', False)
                        lines.append(f"      - NLI: {nli_label.upper()} (ent: {nli_ent:.3f}) - Can explain: {'✓' if can_explain else '✗'}")
            
            lines.append("")
        
        lines.append("=" * 100)
        return "\n".join(lines)
