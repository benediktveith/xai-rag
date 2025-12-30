from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

from src.modules.kgrag_ex_pipeline import KGRAGExPipeline, KGRAGRun
from src.modules.kgrag_ex_perturbations import KGPerturbationFactory, Perturbation
from src.modules.kgrag_ex_metrics import KGRAGGraphMetrics
from src.modules.ragex_text_perturber import RAGExTextPerturber


def _norm(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


@dataclass
class PerturbationOutcome:
    kind: str
    removed: str
    answer_changed: bool
    answer: str


@dataclass
class ExplanationReport:
    most_influential_node: Optional[str]
    most_influential_edge: Optional[str]
    most_influential_subpath: Optional[str]
    outcomes: List[PerturbationOutcome]
    rq2_positions: List[Dict[str, Any]]
    rq3_node_types: List[Dict[str, Any]]
    rq4_graph_metrics: List[Dict[str, Any]]
    cost: Dict[str, Any]
    comparison_rag_ex: Optional[Dict[str, Any]]


class KGRAGExExplainer:
    def __init__(self, pipeline: KGRAGExPipeline):
        self.pipeline = pipeline
        self.factory = KGPerturbationFactory()
        self.metrics = KGRAGGraphMetrics(pipeline.kg)

    def _run_answer_with_path_override(self, base: KGRAGRun, path_override) -> tuple[str, int, int, int]:
        kg_context = self.pipeline.path.pseudo_paragraph(path_override)
        retrieved = self.pipeline.baseline.retrieve(query=f"{base.question}\n\nKG Context:\n{kg_context}", k=len(base.retrieved) - 1)
        merged = [base.retrieved[0]] + retrieved if base.retrieved and base.retrieved[0].chunk_id == "kg-1" else retrieved

        if merged and merged[0].chunk_id == "kg-1":
            merged[0].content = kg_context

        return self.pipeline.baseline.answer(question=base.question, retrieved=merged)

    def explain(self, run: KGRAGRun, do_rag_ex_comparison: bool = True, rag_ex_window_tokens: int = 30, rag_ex_max_perturbations: int = 40) -> ExplanationReport:
        base_answer = _norm(run.answer)

        outcomes: List[PerturbationOutcome] = []
        rq2_positions: List[Dict[str, Any]] = []
        rq3_node_types: List[Dict[str, Any]] = []
        rq4_graph_metrics: List[Dict[str, Any]] = []

        node_influence: Dict[str, int] = {}
        edge_influence: Dict[str, int] = {}
        subpath_influence: Dict[str, int] = {}

        total_calls = int(run.llm_calls)
        total_in = int(run.tokens_in)
        total_out = int(run.tokens_out)

        if not run.path:
            return ExplanationReport(
                most_influential_node=None,
                most_influential_edge=None,
                most_influential_subpath=None,
                outcomes=[],
                rq2_positions=[],
                rq3_node_types=[],
                rq4_graph_metrics=[],
                cost={"llm_calls": total_calls, "tokens_in": total_in, "tokens_out": total_out},
                comparison_rag_ex=None,
            )

        perturbations: List[Perturbation] = []
        perturbations.extend(self.factory.subpath_perturbations(run.path))
        perturbations.extend(self.factory.node_perturbations(run.path))
        perturbations.extend(self.factory.edge_perturbations(run.path))

        for idx, p in enumerate(perturbations):
            ans, tin, tout, calls = self._run_answer_with_path_override(run, p.path)
            total_calls += calls
            total_in += tin
            total_out += tout

            changed = _norm(ans) != base_answer
            outcomes.append(PerturbationOutcome(kind=p.kind, removed=p.removed, answer_changed=changed, answer=ans))

            if changed:
                if p.kind == "node":
                    node_influence[p.removed] = node_influence.get(p.removed, 0) + 1
                elif p.kind == "edge":
                    edge_influence[p.removed] = edge_influence.get(p.removed, 0) + 1
                elif p.kind == "subpath":
                    subpath_influence[p.removed] = subpath_influence.get(p.removed, 0) + 1

            if p.kind == "subpath":
                rel_pos = float(idx) / float(max(1, len(perturbations) - 1))
                rq2_positions.append({"kind": p.kind, "removed": p.removed, "relative_position_proxy": rel_pos, "changed": changed})

        for s in run.path:
            rq3_node_types.append({"node": s.subject, "type": self.metrics.node_type(s.subject)})
            rq3_node_types.append({"node": s.object, "type": self.metrics.node_type(s.object)})

        for s in run.path:
            u = s.subject
            v = s.object
            edge_key = f"{u}::{s.relation}::{v}"
            rq4_graph_metrics.append(
                {
                    "node": u,
                    "degree": self.metrics.node_degree(u),
                    "edge": edge_key,
                    "edge_betweenness": self.metrics.edge_betweenness(u, v),
                    "subpath_score": self.metrics.subpath_score(u, v),
                }
            )

        def argmax(d: Dict[str, int]) -> Optional[str]:
            if not d:
                return None
            return sorted(d.items(), key=lambda x: x[1], reverse=True)[0][0]

        comp_rag_ex: Optional[Dict[str, Any]] = None
        if do_rag_ex_comparison:
            perturber = RAGExTextPerturber(window_tokens=rag_ex_window_tokens, max_perturbations=rag_ex_max_perturbations)
            context = perturber.build_context(run.retrieved)
            text_perts = perturber.perturb(context)

            rag_ex_changes = 0
            rag_ex_calls = 0
            rag_ex_in = 0
            rag_ex_out = 0

            for tp in text_perts:
                ans2, tin2, tout2, calls2 = perturber.answer_with_context_override(self.pipeline.baseline, run.question, tp.perturbed_text)
                rag_ex_calls += calls2
                rag_ex_in += tin2
                rag_ex_out += tout2
                if _norm(ans2) != base_answer:
                    rag_ex_changes += 1

            comp_rag_ex = {
                "window_tokens": rag_ex_window_tokens,
                "num_perturbations": len(text_perts),
                "num_changes": rag_ex_changes,
                "llm_calls": rag_ex_calls,
                "tokens_in": rag_ex_in,
                "tokens_out": rag_ex_out,
            }

        return ExplanationReport(
            most_influential_node=argmax(node_influence),
            most_influential_edge=argmax(edge_influence),
            most_influential_subpath=argmax(subpath_influence),
            outcomes=outcomes,
            rq2_positions=rq2_positions,
            rq3_node_types=rq3_node_types,
            rq4_graph_metrics=rq4_graph_metrics,
            cost={"llm_calls": total_calls, "tokens_in": total_in, "tokens_out": total_out},
            comparison_rag_ex=comp_rag_ex,
        )
