from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

from src.modules.kgrag_ex_pipeline import KGRAGExPipeline, KGRAGRun
from src.modules.kgrag_ex_perturbations import KGPerturbationFactory, Perturbation
from src.modules.kgrag_ex_metrics import KGRAGGraphMetrics


def _norm(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


@dataclass(frozen=True)
class PerturbationOutcome:
    kind: str
    removed: str
    answer_changed: bool
    answer: str


@dataclass(frozen=True)
class ExplanationReport:
    most_influential_node: Optional[str]
    most_influential_edge: Optional[str]
    most_influential_subpath: Optional[str]

    outcomes: List[PerturbationOutcome]

    rq2_positions: List[Dict[str, Any]]
    rq3_node_types: List[Dict[str, Any]]
    rq4_graph_metrics: List[Dict[str, Any]]

    cost: Dict[str, Any]


class KGRAGExExplainer:
    """
    KG level perturbation explainer.
    No RAG Ex comparison, only demonstrates KGRAG Ex behavior.
    """

    def __init__(self, pipeline: KGRAGExPipeline):
        self.pipeline = pipeline
        self.factory = KGPerturbationFactory()
        self.metrics = KGRAGGraphMetrics(pipeline.kg)

    def _answer_for_path_override(self, base: KGRAGRun, path_override) -> tuple[str, int, int, int]:
        kg_context = self.pipeline.path.pseudo_paragraph(path_override)
        ans, tin, tout, calls, _retrieved = self.pipeline.answer_with_kg_context(
            question=base.question,
            kg_context=kg_context,
            top_k_docs=max(1, len(base.retrieved) - 1),
            extra="",
        )
        return ans, tin, tout, calls

    def explain(self, run: KGRAGRun) -> ExplanationReport:
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
            )

        perturbations: List[Perturbation] = []
        perturbations.extend(self.factory.subpath_perturbations(run.path))
        perturbations.extend(self.factory.node_perturbations(run.path))
        perturbations.extend(self.factory.edge_perturbations(run.path))

        for idx, p in enumerate(perturbations):
            ans, tin, tout, calls = self._answer_for_path_override(run, p.path)
            total_calls += calls
            total_in += tin
            total_out += tout

            changed = _norm(ans) != base_answer
            outcomes.append(
                PerturbationOutcome(kind=p.kind, removed=p.removed, answer_changed=changed, answer=ans)
            )

            if changed:
                if p.kind == "node":
                    node_influence[p.removed] = node_influence.get(p.removed, 0) + 1
                elif p.kind == "edge":
                    edge_influence[p.removed] = edge_influence.get(p.removed, 0) + 1
                elif p.kind == "subpath":
                    subpath_influence[p.removed] = subpath_influence.get(p.removed, 0) + 1

            if p.kind == "subpath":
                rel_pos = float(idx) / float(max(1, len(perturbations) - 1))
                rq2_positions.append(
                    {
                        "kind": p.kind,
                        "removed": p.removed,
                        "relative_position_proxy": rel_pos,
                        "changed": changed,
                    }
                )

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

        return ExplanationReport(
            most_influential_node=argmax(node_influence),
            most_influential_edge=argmax(edge_influence),
            most_influential_subpath=argmax(subpath_influence),
            outcomes=outcomes,
            rq2_positions=rq2_positions,
            rq3_node_types=rq3_node_types,
            rq4_graph_metrics=rq4_graph_metrics,
            cost={"llm_calls": total_calls, "tokens_in": total_in, "tokens_out": total_out},
        )
