from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

try:
    from langchain_core.documents import Document
except Exception:
    Document = None

from src.modules.explainers.cot_explainable import ExplainableAnswer

NO_SOURCE = "no-source"

class CoTEvaluator:
    def __init__(
        self,
        embedding_model: str = "multi-qa-MiniLM-L6-cos-v1",
        nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        similarity_threshold: float = 0.6,
        lowercase: bool = True,
    ) -> None:
        self.embedding_model = embedding_model
        self.nli_model = nli_model
        self.similarity_threshold = similarity_threshold
        self.lowercase = lowercase
        self._model: SentenceTransformer | None = None
        self._nli_tokenizer = None
        self._nli_model = None
        self._device = "cpu"

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model)

    def _ensure_nli_model(self) -> None:
        if self._nli_tokenizer is None or self._nli_model is None:
            self._nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model)
            self._nli_model.to(self._device)
            self._nli_model.eval()

    def _embed(self, texts: Sequence[str]):
        self._ensure_model()
        return self._model.encode(list(texts), normalize_embeddings=True)

    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        return text.lower() if self.lowercase else text

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r"\b\w+\b", self._normalize(text))

    def _doc_text(self, item: Any) -> str:
        if Document is not None and isinstance(item, Document):
            return (item.page_content or "").strip()
        if isinstance(item, dict):
            return str(item.get("content") or item.get("page_content") or "").strip()
        if isinstance(item, str):
            return item.strip()
        return str(item).strip()

    def _build_doc_map(self, documents: Sequence[Any], from_hop: int = 1) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for i, item in enumerate(documents, start=1):
            doc_id = f"chunk-{from_hop}-{i}"
            mapping[doc_id] = self._doc_text(item)
        return mapping

    def _to_model(self, obj: Any) -> ExplainableAnswer:
        if isinstance(obj, ExplainableAnswer):
            return obj
        if isinstance(obj, dict):
            return ExplainableAnswer.model_validate(obj)
        raise TypeError(f"Unsupported explanation type: {type(obj)!r}")

    def similarity_matrix(
        self,
        explanation: Any,
        documents: Sequence[Any],
        doc_id_map: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """
        Compute similarity matrix between evidence spans and documents.
        
        Args:
            explanation: The explainable answer object
            documents: List of source documents
            doc_id_map: Optional mapping of document IDs to text content
            
        Returns:
            Dictionary containing evidence_spans, doc_ids, and similarity scores matrix
        """
        expl = self._to_model(explanation)
        doc_map = doc_id_map or self._build_doc_map(documents)
        doc_ids = list(doc_map.keys())
        doc_texts = [doc_map[doc_id] for doc_id in doc_ids]

        spans = [ev.span or "" for ev in expl.evidence]
        if not spans or not doc_texts:
            return {"evidence_spans": spans, "doc_ids": doc_ids, "scores": []}

        span_embs = self._embed(spans)
        doc_embs = self._embed(doc_texts)
        scores = util.cos_sim(span_embs, doc_embs).cpu().tolist()
        return {"evidence_spans": spans, "doc_ids": doc_ids, "scores": scores}

    def explanation_quality(
        self,
        explanation: Any,
        answer: str,
        documents: Sequence[Any],
        doc_id_map: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluate explanation quality using NLI-style reasoning.
        Can the answer be derived from the provided evidence?
        
        Args:
            explanation: The explainable answer object
            documents: List of source documents
            doc_id_map: Optional mapping of document IDs to text content
            
        Returns:
            Dictionary with NLI-based quality metrics
        """
        self._ensure_nli_model()
        expl = self._to_model(explanation)

        # Get all evidence spans
        evidence_spans = [ev.span or "" for ev in expl.evidence if ev.span]
        if not evidence_spans:
            return {
                "nli_score": 0.0,
                "entailment_score": 0.0,
                "can_derive_answer": False,
                "label": "no_evidence",
                "per_evidence_scores": []
            }
        
        hypothesis = answer
        per_evidence_scores = []
        
        # Calculate NLI score for each evidence span individually
        for i, evidence_span in enumerate(evidence_spans):
            premise = evidence_span
            
            # Use NLI to check if answer can be derived from this evidence
            inputs = self._nli_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self._nli_model(**inputs)
            
            # Get probabilities with softmax
            prediction = torch.softmax(output.logits[0], -1).tolist()
            label_names = ["entailment", "neutral", "contradiction"]
            label_scores = {name: float(score) for score, name in zip(prediction, label_names)}
            
            # Get the label with highest score
            predicted_label = max(label_scores, key=label_scores.get)
            
            per_evidence_scores.append({
                "evidence_index": i,
                "evidence_span": evidence_span,
                "entailment": float(label_scores["entailment"]),
                "neutral": float(label_scores["neutral"]),
                "contradiction": float(label_scores["contradiction"]),
                "label": predicted_label
            })
        
        # Calculate aggregate scores
        avg_entailment = sum(ev["entailment"] for ev in per_evidence_scores) / len(per_evidence_scores)
        max_entailment = max(ev["entailment"] for ev in per_evidence_scores)
        
        # Overall assessment based on best evidence
        overall_label = per_evidence_scores[0]["label"] if per_evidence_scores else "no_evidence"
        can_derive = max_entailment > 0.5 and any(ev["label"] == "entailment" for ev in per_evidence_scores)
        
        return {
            "nli_score": float(max_entailment),
            "entailment_score": float(avg_entailment),
            "max_entailment": float(max_entailment),
            "neutral_score": float(sum(ev["neutral"] for ev in per_evidence_scores) / len(per_evidence_scores)),
            "contradiction_score": float(sum(ev["contradiction"] for ev in per_evidence_scores) / len(per_evidence_scores)),
            "can_derive_answer": bool(can_derive),
            "label": overall_label,
            "per_evidence_scores": per_evidence_scores
        }

    def interpretability(
        self,
        explanation: Any,
        ground_truth_evidence: str | None = None,
    ) -> Dict[str, Any]:
        """
        Calculate interpretability score using Jaccard coefficient.
        Compares system-generated evidence highlights with expert-annotated evidence.
        
        Args:
            explanation: The explainable answer object
            ground_truth_evidence: Expert-annotated ground truth evidence (optional)
            
        Returns:
            Dictionary with interpretability scores
        """
        if not ground_truth_evidence:
            return {
                "interpretability_score": None,
                "explanation": "No ground truth evidence provided"
            }
        
        expl = self._to_model(explanation)
        
        # Extract system-generated evidence spans
        evidence_spans = [ev.span or "" for ev in expl.evidence if ev.span]
        if not evidence_spans:
            return {
                "interpretability_score": 0.0,
                "jaccard_score": 0.0,
                "system_evidence": "",
                "ground_truth_evidence": ground_truth_evidence,
                "explanation": "No system evidence generated"
            }
        
        # Combine all evidence spans into one text
        system_evidence = " ".join(evidence_spans)
        
        # Tokenize both system and ground truth evidence
        system_tokens = set(self._tokenize(system_evidence))
        truth_tokens = set(self._tokenize(ground_truth_evidence))
        
        # Calculate Jaccard coefficient: |A ∩ B| / |A ∪ B|
        if not system_tokens and not truth_tokens:
            jaccard_score = 1.0
        elif not system_tokens or not truth_tokens:
            jaccard_score = 0.0
        else:
            intersection = len(system_tokens & truth_tokens)
            union = len(system_tokens | truth_tokens)
            jaccard_score = intersection / union if union > 0 else 0.0
        
        return {
            "interpretability_score": float(jaccard_score),
            "jaccard_score": float(jaccard_score),
            "system_evidence": system_evidence,
            "ground_truth_evidence": ground_truth_evidence,
            "intersection_size": len(system_tokens & truth_tokens) if system_tokens and truth_tokens else 0,
            "union_size": len(system_tokens | truth_tokens) if system_tokens and truth_tokens else 0
        }

    def evaluate(
        self,
        explanation: Any,
        documents: Sequence[Any],
        predicted_answer: str | None = None,
        ground_truth: str | None = None,
        ground_truth_evidence: str | None = None,
        doc_id_map: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """
        Run all evaluation metrics on the explanation.
        
        Args:
            explanation: The explainable answer object
            documents: List of source documents
            predicted_answer: The predicted answer (optional)
            ground_truth: The ground truth answer (optional)
            ground_truth_evidence: Ground truth evidence for interpretability (optional)
            doc_id_map: Optional mapping of document IDs to text content
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # Add explanation quality if we have an answer
        if predicted_answer:
            results["explanation_quality"] = self.explanation_quality(explanation, predicted_answer, documents, doc_id_map)
        
        # Add interpretability if we have ground truth evidence
        if ground_truth_evidence:
            results["interpretability"] = self.interpretability(explanation, ground_truth_evidence)
        
        return results

    def prettify(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics in a nice, readable way.
        
        Args:
            metrics: Dictionary of metrics from evaluate()
            
        Returns:
            Formatted string representation of the metrics
        """
        lines = ["=" * 60, "Evaluation Metrics", "=" * 60, ""]
        
        # Explanation Quality
        if "explanation_quality" in metrics:
            lines.append("📊 Explanation Quality (NLI):")
            eq = metrics["explanation_quality"]
            nli_score = eq.get('nli_score', 0.0)
            if isinstance(nli_score, float):
                lines.append(f"  • NLI Score (Max): {nli_score:.4f}")
            else:
                lines.append(f"  • NLI Score: {nli_score}")
            lines.append(f"  • Can Derive Answer: {'✓ Yes' if eq.get('can_derive_answer') else '✗ No'}")
            lines.append(f"  • Overall Label: {eq.get('label', 'N/A')}")
            if 'entailment_score' in eq:
                lines.append(f"  • Avg Entailment: {eq['entailment_score']:.4f}")
            if 'max_entailment' in eq:
                lines.append(f"  • Max Entailment: {eq['max_entailment']:.4f}")
            if 'neutral_score' in eq:
                lines.append(f"  • Avg Neutral: {eq['neutral_score']:.4f}")
            if 'contradiction_score' in eq:
                lines.append(f"  • Avg Contradiction: {eq['contradiction_score']:.4f}")
            
            # Show per-evidence scores
            if eq.get('per_evidence_scores'):
                lines.append("  • Per-Evidence Scores:")
                for ev in eq['per_evidence_scores']:
                    idx = ev.get('evidence_index', 0)
                    ent = ev.get('entailment', 0.0)
                    label = ev.get('label', 'N/A')
                    span = ev.get('evidence_span', '')[:80] + "..." if len(ev.get('evidence_span', '')) > 80 else ev.get('evidence_span', '')
                    lines.append(f"    [{idx}] {label.upper()} (ent: {ent:.3f})")
                    lines.append(f"        \"{span}\"")
            lines.append("")
        
        # Interpretability
        if "interpretability" in metrics:
            lines.append("🎯 Interpretability (Evidence Overlap):")
            interp = metrics["interpretability"]
            score = interp.get('interpretability_score')
            if score is not None:
                lines.append(f"  • Jaccard Score: {score:.4f}")
                if 'intersection_size' in interp:
                    lines.append(f"  • Intersection: {interp['intersection_size']} tokens")
                if 'union_size' in interp:
                    lines.append(f"  • Union: {interp['union_size']} tokens")
            else:
                lines.append(f"  • {interp.get('explanation', 'N/A')}")
            lines.append("")
    
        
        lines.append("=" * 60)
        return "\n".join(lines)
