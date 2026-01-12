# src/modules/knowledge_graph/kgrag_ex_explainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from src.modules.knowledge_graph.kgrag_ex_pipeline import KGRAGExPipeline, KGRAGRun
from src.modules.knowledge_graph.kgrag_ex_perturbations import KGPerturbationFactory
from src.modules.knowledge_graph.kg_store import KGStore

import networkx as nx


@dataclass(frozen=True)
class RQ1Sensitivity:
    node_changed: int
    edge_changed: int
    subpath_changed: int
    node_total: int
    edge_total: int
    subpath_total: int


@dataclass(frozen=True)
class RQ2PositionRecord:
    kind: str
    removed: str
    index: int
    rel_pos: float
    answer_changed: bool


@dataclass(frozen=True)
class RQ3NodeTypeRecord:
    node: str
    node_type: str
    answer_changed: bool


@dataclass(frozen=True)
class RQ4MetricsRecord:
    kind: str
    removed: str
    answer_changed: bool

    node_degree: Optional[int] = None
    node_degree_rank_rel: Optional[float] = None

    edge_betweenness: Optional[float] = None
    edge_betweenness_rank_rel: Optional[float] = None

    subpath_score: Optional[float] = None
    subpath_score_rank_rel: Optional[float] = None


@dataclass
class KGRAGExplainReport:
    base_answer: str
    base_chain: str
    base_paragraph: str
    outcomes: List["PerturbationOutcome"]

    most_influential_node: Optional[str]
    most_influential_edge: Optional[str]
    most_influential_subpath: Optional[str]

    rq1_sensitivity: Optional[RQ1Sensitivity] = None
    rq2_positions: Optional[List[RQ2PositionRecord]] = None
    rq3_node_types: Optional[List[RQ3NodeTypeRecord]] = None
    rq4_graph_metrics: Optional[List[RQ4MetricsRecord]] = None


@dataclass(frozen=True)
class PerturbationOutcome:
    kind: str
    removed: str
    chain_str: str
    paragraph: str
    answer: str
    answer_changed: bool
    important_entities: List[str]


def _safe_rel_pos(idx: int, n: int) -> float:
    if n <= 1:
        return 0.0
    return float(idx) / float(n - 1)


def _parse_edge_removed(removed: str) -> Optional[tuple[str, str, str]]:
    parts = (removed or "").split("::")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _parse_subpath_removed(removed: str) -> Optional[tuple[str, str, str]]:
    s = (removed or "").strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None
    inner = s[1:-1]
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _relative_rank(value: float, values: List[float], higher_is_better: bool = True) -> float:
    if not values:
        return 0.0
    uniq = sorted(set(values), reverse=higher_is_better)
    if len(uniq) == 1:
        return 0.0
    try:
        idx = uniq.index(value)
    except ValueError:
        idx = min(range(len(uniq)), key=lambda i: abs(uniq[i] - value))
    return float(idx) / float(len(uniq) - 1)


class KGRAGExExplainer:
    """
    Paper nah:
    - Perturbations verändern KG chain
    - Daraus wird pseudo paragraph generiert
    - Retrieval läuft ausschließlich über pseudo paragraph, nicht über question
    - Answer wird auf Basis retrieved docs plus question generiert
    """

    def __init__(self, pipeline: KGRAGExPipeline, betweenness_k: int = 0, betweenness_seed: int = 7):
        self.pipeline = pipeline
        self.factory = KGPerturbationFactory()
        self.betweenness_k = int(betweenness_k)
        self.betweenness_seed = int(betweenness_seed)
        self._edge_betweenness_cache: Optional[Dict[Tuple[str, str], float]] = None

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

    def _rq2(self, run: KGRAGRun, outcomes: List[PerturbationOutcome]) -> List[RQ2PositionRecord]:
        pl = run.path_lists
        nodes = pl.node_list or []
        edges = pl.edge_list or []
        subs = pl.subpath_list or []

        out: List[RQ2PositionRecord] = []

        for o in outcomes:
            if o.kind == "node":
                try:
                    idx = nodes.index(o.removed)
                except ValueError:
                    continue
                rel_pos = _safe_rel_pos(idx, len(nodes))
                out.append(RQ2PositionRecord(o.kind, o.removed, idx, rel_pos, o.answer_changed))

            elif o.kind == "edge":
                parsed = _parse_edge_removed(o.removed)
                if not parsed:
                    continue
                u, rel, v = parsed
                idx = None
                for i in range(len(edges)):
                    if i + 1 < len(nodes) and nodes[i] == u and edges[i] == rel and nodes[i + 1] == v:
                        idx = i
                        break
                if idx is None:
                    continue
                rel_pos = _safe_rel_pos(idx, len(edges))
                out.append(RQ2PositionRecord(o.kind, o.removed, idx, rel_pos, o.answer_changed))

            elif o.kind == "subpath":
                parsed = _parse_subpath_removed(o.removed)
                if not parsed:
                    continue
                tup = (parsed[0], parsed[1], parsed[2])
                try:
                    idx = subs.index(tup)
                except ValueError:
                    continue
                rel_pos = _safe_rel_pos(idx, len(subs))
                out.append(RQ2PositionRecord(o.kind, o.removed, idx, rel_pos, o.answer_changed))

        return out

    def _rq3(self, kg: KGStore, outcomes: List[PerturbationOutcome]) -> List[RQ3NodeTypeRecord]:
        out: List[RQ3NodeTypeRecord] = []
        for o in outcomes:
            if o.kind != "node":
                continue
            n = o.removed
            ntype = kg.node_type(n) if n in kg.g else "Unknown"
            out.append(RQ3NodeTypeRecord(node=n, node_type=str(ntype), answer_changed=o.answer_changed))
        return out

    def _edge_betweenness(self) -> Dict[Tuple[str, str], float]:
        if self._edge_betweenness_cache is not None:
            return self._edge_betweenness_cache

        G = self.pipeline.kg.g.to_undirected()

        if self.betweenness_k and self.betweenness_k > 0:
            eb_raw = nx.edge_betweenness_centrality(G, k=self.betweenness_k, seed=self.betweenness_seed)
        else:
            eb_raw = nx.edge_betweenness_centrality(G)

        out: Dict[Tuple[str, str], float] = {}

        for key, val in eb_raw.items():
            if isinstance(key, tuple) and len(key) == 3:
                u, v, _k = key
            else:
                u, v = key

            a, b = (u, v) if str(u) <= str(v) else (v, u)
            kk = (str(a), str(b))
            prev = out.get(kk, 0.0)
            if float(val) > prev:
                out[kk] = float(val)

        self._edge_betweenness_cache = out
        return out

    def _rq4(self, run: KGRAGRun, outcomes: List[PerturbationOutcome]) -> List[RQ4MetricsRecord]:
        kg = self.pipeline.kg
        pl = run.path_lists
        nodes = pl.node_list or []
        edges = pl.edge_list or []
        subs = pl.subpath_list or []

        G_und = kg.g.to_undirected(as_view=True)
        node_degrees: Dict[str, int] = {n: int(G_und.degree(n)) for n in nodes if n in G_und}
        node_degree_vals = [float(v) for v in node_degrees.values()]

        eb = self._edge_betweenness()
        edge_b_vals: List[float] = []
        edge_b_map: Dict[Tuple[str, str, str], float] = {}
        for i in range(len(edges)):
            if i + 1 >= len(nodes):
                continue
            u, rel, v = nodes[i], edges[i], nodes[i + 1]
            a, b = (u, v) if str(u) <= str(v) else (v, u)
            val = float(eb.get((str(a), str(b)), 0.0))
            edge_b_map[(u, rel, v)] = val
            edge_b_vals.append(val)

        sub_scores: Dict[Tuple[str, str, str], float] = {}
        sub_vals: List[float] = []
        for (u, rel, v) in subs:
            bet = float(edge_b_map.get((u, rel, v), 0.0))
            du = float(node_degrees.get(u, 0))
            dv = float(node_degrees.get(v, 0))
            denom = du + dv
            score = (bet / denom) if denom > 0 else 0.0
            sub_scores[(u, rel, v)] = score
            sub_vals.append(score)

        out: List[RQ4MetricsRecord] = []

        for o in outcomes:
            if o.kind == "node":
                n = o.removed
                deg = node_degrees.get(n)
                deg_rank = None
                if deg is not None and node_degree_vals:
                    deg_rank = _relative_rank(float(deg), node_degree_vals, higher_is_better=True)
                out.append(
                    RQ4MetricsRecord(
                        kind="node",
                        removed=o.removed,
                        answer_changed=o.answer_changed,
                        node_degree=deg,
                        node_degree_rank_rel=deg_rank,
                    )
                )

            elif o.kind == "edge":
                parsed = _parse_edge_removed(o.removed)
                if not parsed:
                    continue
                u, rel, v = parsed
                b = edge_b_map.get((u, rel, v))
                b_rank = None
                if b is not None and edge_b_vals:
                    b_rank = _relative_rank(float(b), edge_b_vals, higher_is_better=True)
                out.append(
                    RQ4MetricsRecord(
                        kind="edge",
                        removed=o.removed,
                        answer_changed=o.answer_changed,
                        edge_betweenness=b,
                        edge_betweenness_rank_rel=b_rank,
                    )
                )

            elif o.kind == "subpath":
                parsed = _parse_subpath_removed(o.removed)
                if not parsed:
                    continue
                tup = (parsed[0], parsed[1], parsed[2])
                score = sub_scores.get(tup)
                s_rank = None
                if score is not None and sub_vals:
                    s_rank = _relative_rank(float(score), sub_vals, higher_is_better=True)
                out.append(
                    RQ4MetricsRecord(
                        kind="subpath",
                        removed=o.removed,
                        answer_changed=o.answer_changed,
                        subpath_score=score,
                        subpath_score_rank_rel=s_rank,
                    )
                )

        return out

    def explain(self, run: KGRAGRun, options: str = "", top_k_docs: int = 4) -> KGRAGExplainReport:
        base_answer = run.answer or ""
        base_chain = run.kg_chain or ""
        base_paragraph = run.kg_paragraph or ""

        if not (run.path_lists and run.path_lists.node_list):
            return KGRAGExplainReport(
                base_answer=base_answer,
                base_chain=base_chain,
                base_paragraph=base_paragraph,
                outcomes=[],
                most_influential_node=None,
                most_influential_edge=None,
                most_influential_subpath=None,
            )

        perturbations = self.factory.generate(run.path_lists)
        outcomes: List[PerturbationOutcome] = []

        for p in perturbations:
            par, _ = self.pipeline._generate_pseudo_paragraph(p.chain_str)

            # Paper konform, Retrieval query ist pseudo paragraph, Answer nutzt question plus retrieved docs
            ans, _, _ = self.pipeline._rag_answer_with_optional_kg_paragraph(
                question=run.question,
                kg_paragraph=par,
                top_k_docs=top_k_docs,
            )

            changed = bool(base_answer) and bool(ans) and (ans != base_answer)

            outcomes.append(
                PerturbationOutcome(
                    kind=p.kind,
                    removed=p.removed,
                    chain_str=p.chain_str,
                    paragraph=par,
                    answer=ans,
                    answer_changed=changed,
                    important_entities=p.important_entities,
                )
            )

        most_node = next((o.removed for o in outcomes if o.kind == "node" and o.answer_changed), None)
        most_edge = next((o.removed for o in outcomes if o.kind == "edge" and o.answer_changed), None)
        most_sub = next((o.removed for o in outcomes if o.kind == "subpath" and o.answer_changed), None)

        rq1 = self._rq1(outcomes)
        rq2 = self._rq2(run, outcomes)
        rq3 = self._rq3(self.pipeline.kg, outcomes)
        rq4 = self._rq4(run, outcomes)

        return KGRAGExplainReport(
            base_answer=base_answer,
            base_chain=base_chain,
            base_paragraph=base_paragraph,
            outcomes=outcomes,
            most_influential_node=most_node,
            most_influential_edge=most_edge,
            most_influential_subpath=most_sub,
            rq1_sensitivity=rq1,
            rq2_positions=rq2,
            rq3_node_types=rq3,
            rq4_graph_metrics=rq4,
        )
