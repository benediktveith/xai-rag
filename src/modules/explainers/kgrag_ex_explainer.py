# src/modules/knowledge_graph/kgrag_ex_explainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.modules.knowledge_graph.kgrag_ex_pipeline import KGRAGExPipeline, KGRAGRun
from src.modules.knowledge_graph.kgrag_ex_perturbations import KGPerturbationFactory, ChainPerturbation
from src.modules.knowledge_graph.kg_path_service import PathAsLists


@dataclass(frozen=True)
class Sensitivity:
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
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0


@dataclass
class KGRAGExplainReport:
    base_answer: str
    base_chains: List[str]
    base_paragraph: str
    outcomes: List[PerturbationOutcome]

    most_influential_node: Optional[str]
    most_influential_edge: Optional[str]
    most_influential_subpath: Optional[str]

    sensitivity: Sensitivity
    
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


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

    def _sensitivity(self, outcomes: List[PerturbationOutcome]) -> Sensitivity:
        def _counts(kind: str) -> tuple[int, int]:
            xs = [o for o in outcomes if o.kind == kind]
            return sum(1 for o in xs if o.answer_changed), len(xs)

        nc, nt = _counts("node")
        ec, et = _counts("edge")
        sc, st = _counts("subpath")
        return Sensitivity(
            node_changed=nc,
            edge_changed=ec,
            subpath_changed=sc,
            node_total=nt,
            edge_total=et,
            subpath_total=st,
        )

    def _build_joined_paragraph_from_chains(self, chains: List[str]) -> tuple[str, int, int, int]:
        llm_calls = 0
        tokens_in = 0
        tokens_out = 0
        paragraphs: List[str] = []
        for ch in chains:
            p, c, tin, tout = self.pipeline._generate_pseudo_paragraph(ch)
            llm_calls += int(c)
            tokens_in += int(tin)
            tokens_out += int(tout)
            if p:
                paragraphs.append(p)
        joined = "\n".join(paragraphs).strip()
        return joined, llm_calls, tokens_in, tokens_out

    def explain(self, run: KGRAGRun, options: str = "") -> KGRAGExplainReport:
        base_answer = (run.answer or "").strip()
        base_paragraph = run.kg_paragraph or ""

        base_chains: List[str] = [pl.chain_str or "" for pl in (run.path_lists or [])]

        if not run.path_lists:
            sens = self._sensitivity([])
            return KGRAGExplainReport(
                base_answer=base_answer,
                base_chains=base_chains,
                base_paragraph=base_paragraph,
                outcomes=[],
                most_influential_node=None,
                most_influential_edge=None,
                most_influential_subpath=None,
                sensitivity=sens,
                llm_calls=0,
                tokens_in=0,
                tokens_out=0,
            )

        outcomes: List[PerturbationOutcome] = []

        total_calls = 0
        total_in = 0
        total_out = 0

        for path_idx, pl in enumerate(run.path_lists):
            perturbations: List[ChainPerturbation] = self.factory.generate(pl)

            for p in perturbations:
                temp_chains = base_chains[:]
                if path_idx >= len(temp_chains):
                    continue
                temp_chains[path_idx] = p.chain_str

                joined_par, calls_par, in_par, out_par = self._build_joined_paragraph_from_chains(temp_chains)
                ans, calls_ans, in_ans, out_ans = self.pipeline._answer_from_paragraph(joined_par, run.question, options=options)

                calls = int(calls_par) + int(calls_ans)
                tin = int(in_par) + int(in_ans)
                tout = int(out_par) + int(out_ans)

                total_calls += calls
                total_in += tin
                total_out += tout

                changed = (
                    base_answer in ("A", "B", "C", "D")
                    and ans in ("A", "B", "C", "D")
                    and (ans != base_answer)
                )

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
                        tokens_in=tin,
                        tokens_out=tout,
                        llm_calls=calls,
                    )
                )

        most_node = next((o.removed for o in outcomes if o.kind == "node" and o.answer_changed), None)
        most_edge = next((o.removed for o in outcomes if o.kind == "edge" and o.answer_changed), None)
        most_sub = next((o.removed for o in outcomes if o.kind == "subpath" and o.answer_changed), None)

        sens = self._sensitivity(outcomes)

        return KGRAGExplainReport(
            base_answer=base_answer,
            base_chains=base_chains,
            base_paragraph=base_paragraph,
            outcomes=outcomes,
            most_influential_node=most_node,
            most_influential_edge=most_edge,
            most_influential_subpath=most_sub,
            sensitivity=sens,
            llm_calls=total_calls,
            tokens_in=total_in,
            tokens_out=total_out,
        )

