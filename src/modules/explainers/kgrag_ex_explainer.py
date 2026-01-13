# src/modules/knowledge_graph/kgrag_ex_explainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.modules.knowledge_graph.kgrag_ex_pipeline import KGRAGExPipeline, KGRAGRun
from src.modules.knowledge_graph.kgrag_ex_perturbations import KGPerturbationFactory, ChainPerturbation
from src.modules.knowledge_graph.kg_path_service import PathAsLists


@dataclass(frozen=True)
class RQ1Sensitivity:
    node_changed: int
    edge_changed: int
    subpath_changed: int
    node_total: int
    edge_total: int
    subpath_total: int


@dataclass(frozen=True)
class PerturbationOutcome:
    path_idx: int
    kind: str
    removed: str
    chain_str: str
    paragraph: str
    answer: str
    answer_changed: bool
    important_entities: List[str]


@dataclass
class KGRAGExplainReport:
    base_answer: str
    base_chains: List[str]
    base_paragraph: str
    outcomes: List[PerturbationOutcome]

    most_influential_node: Optional[str]
    most_influential_edge: Optional[str]
    most_influential_subpath: Optional[str]

    rq1_sensitivity: RQ1Sensitivity


class KGRAGExExplainer:
    """
    Notebook konform:
    - Perturbations für ALLE Pfade
    - Pro Perturbation wird genau eine Chain ersetzt, alle anderen bleiben
    - Paragraph wird pro Chain generiert, dann gejoint
    - changed, wenn Antwort ungleich Base Antwort
    """

    def __init__(self, pipeline: KGRAGExPipeline):
        self.pipeline = pipeline
        self.factory = KGPerturbationFactory()

    def _rq1(self, outcomes: List[PerturbationOutcome]) -> RQ1Sensitivity:
        def _counts(kind: str) -> tuple[int, int]:
            xs = [o for o in outcomes if o.kind == kind]
            return sum(1 for o in xs if o.answer_changed), len(xs)

        nc, nt = _counts("node")
        ec, et = _counts("edge")
        sc, st = _counts("subpath")
        return RQ1Sensitivity(
            node_changed=nc,
            edge_changed=ec,
            subpath_changed=sc,
            node_total=nt,
            edge_total=et,
            subpath_total=st,
        )

    def _build_joined_paragraph_from_chains(self, chains: List[str]) -> tuple[str, int]:
        llm_calls = 0
        paragraphs: List[str] = []
        for ch in chains:
            p, c = self.pipeline._generate_pseudo_paragraph(ch)
            llm_calls += int(c)
            if p:
                paragraphs.append(p)
        joined = "\n".join(paragraphs).strip()
        return joined, llm_calls

    def explain(self, run: KGRAGRun, options: str = "") -> KGRAGExplainReport:
        base_answer = (run.answer or "").strip()
        base_paragraph = run.kg_paragraph or ""

        base_chains: List[str] = []
        for pl in (run.path_lists or []):
            base_chains.append(pl.chain_str or "")

        if not run.path_lists:
            rq1 = self._rq1([])
            return KGRAGExplainReport(
                base_answer=base_answer,
                base_chains=base_chains,
                base_paragraph=base_paragraph,
                outcomes=[],
                most_influential_node=None,
                most_influential_edge=None,
                most_influential_subpath=None,
                rq1_sensitivity=rq1,
            )

        outcomes: List[PerturbationOutcome] = []

        for path_idx, pl in enumerate(run.path_lists):
            perturbations: List[ChainPerturbation] = self.factory.generate(pl)

            for p in perturbations:
                temp_chains = base_chains[:]
                if path_idx >= len(temp_chains):
                    continue
                temp_chains[path_idx] = p.chain_str

                joined_par, _ = self._build_joined_paragraph_from_chains(temp_chains)
                ans, _ = self.pipeline._answer_from_paragraph(joined_par, run.question, options=options)

                changed = (base_answer in ("A", "B", "C", "D") and ans in ("A", "B", "C", "D") and (ans != base_answer))
                
                outcomes.append(
                    PerturbationOutcome(
                        path_idx=path_idx,
                        kind=p.kind,
                        removed=p.removed,
                        chain_str=p.chain_str,
                        paragraph=joined_par,
                        answer=ans,
                        answer_changed=changed,
                        important_entities=p.important_entities,
                    )
                )

        most_node = next((o.removed for o in outcomes if o.kind == "node" and o.answer_changed), None)
        most_edge = next((o.removed for o in outcomes if o.kind == "edge" and o.answer_changed), None)
        most_sub = next((o.removed for o in outcomes if o.kind == "subpath" and o.answer_changed), None)

        rq1 = self._rq1(outcomes)

        return KGRAGExplainReport(
            base_answer=base_answer,
            base_chains=base_chains,
            base_paragraph=base_paragraph,
            outcomes=outcomes,
            most_influential_node=most_node,
            most_influential_edge=most_edge,
            most_influential_subpath=most_sub,
            rq1_sensitivity=rq1,
        )
