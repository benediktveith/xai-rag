from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.modules.knowledge_graph.kgrag_ex_pipeline import KGRAGExPipeline, KGRAGRun
from src.modules.knowledge_graph.kgrag_ex_perturbations import KGPerturbationFactory, ChainPerturbation


@dataclass(frozen=True)
class Sensitivity:
    # Aggregate statistics describing how often perturbations change the final answer.
    node_changed: int
    edge_changed: int
    subpath_changed: int
    node_total: int
    edge_total: int
    subpath_total: int


@dataclass(frozen=True)
class PerturbationOutcome:
    # Result of a single perturbation applied to one path and its impact on the answer.
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
    # Explanation report, containing the baseline run and all perturbation outcomes.
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
    def __init__(self, pipeline: KGRAGExPipeline):
        self.pipeline = pipeline
        self.factory = KGPerturbationFactory()

    def _sensitivity(self, outcomes: List[PerturbationOutcome]) -> Sensitivity:
        # Compute per-perturbation-kind counts, changed vs total.
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
        # Generate pseudo-paragraphs for each chain and join them, aggregating usage metrics.
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
        # Perturb each path chain, recompose the paragraph, re-answer, and record deltas vs baseline.
        base_answer = (run.answer or "").strip()
        base_paragraph = run.kg_paragraph or ""

        # One chain string per path list entry, empty string if missing.
        base_chains: List[str] = [pl.chain_str or "" for pl in (run.path_lists or [])]

        # No paths means no perturbations, return an empty report with zeroed metrics.
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

        # Iterate each path, generate perturbations, swap the perturbed chain into place, then re-run answer.
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

                # Treat "change" only as multiple-choice label flips.
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

        # Pick the first element per kind that causes an answer flip as a simple "most influential" proxy.
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
