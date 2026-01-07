# src/modules/kg_path_service.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

import networkx as nx

from src.modules.knowledge_graph.kg_schema import KGStep
from src.modules.knowledge_graph.kg_store import KGStore


@dataclass(frozen=True)
class PathAsLists:
    node_list: List[str]
    edge_list: List[str]
    subpath_list: List[Tuple[str, str, str]]
    chain_str: str


class KGPathService:
    """
    DiGraph kompatibel.
    Außerdem liefert es Path Listen, wie im Paper Code verwendet:
      - node_list: [n0, n1, ..., nk]
      - edge_list: [rel0, rel1, ..., rel{k-1}]
      - subpath_list: [(n0, rel0, n1), ...]
      - chain_str: "n0->[rel0]->n1->[rel1]->...->nk"
    """

    def __init__(self, kg: KGStore):
        self.kg = kg

    def shortest_path_nodes(self, start: str, end: str, cutoff: Optional[int] = None) -> List[str]:
        if start not in self.kg.g or end not in self.kg.g:
            return []
        G_und = self.kg.g.to_undirected(as_view=True)
        try:
            nodes = nx.shortest_path(G_und, start, end)
            if cutoff is not None and len(nodes) - 1 > int(cutoff):
                return []
            return [str(n) for n in nodes]
        except Exception:
            return []

    def _edge_relation_for_display(self, u: str, v: str) -> str:
        data = self.kg.edge_attr(u, v)
        if data:
            rel = str(data.get("relation") or "").strip()
            if rel:
                return rel
            rels = data.get("relations") or []
            if rels:
                return str(rels[0])
        return "related_to"

    def shortest_path(self, start: str, end: str, cutoff: Optional[int] = None) -> List[KGStep]:
        nodes = self.shortest_path_nodes(start, end, cutoff=cutoff)
        if len(nodes) < 2:
            return []

        steps: List[KGStep] = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            # In undirected shortest path, a-b could correspond to either direction in DiGraph,
            # but we want a directed edge attribute for explanation. Prefer a->b, else b->a.
            if self.kg.g.has_edge(a, b):
                u, v = a, b
            elif self.kg.g.has_edge(b, a):
                u, v = b, a
            else:
                u, v = a, b

            ed = self.kg.edge_attr(u, v) or {}
            rel = str(ed.get("relation") or "") or self._edge_relation_for_display(u, v)

            steps.append(
                KGStep(
                    subject=u,
                    relation=rel,
                    object=v,
                    edge=ed,
                )
            )
        return steps

    def as_lists(self, steps: List[KGStep]) -> PathAsLists:
        if not steps:
            return PathAsLists(node_list=[], edge_list=[], subpath_list=[], chain_str="")

        node_list = [steps[0].subject]
        edge_list: List[str] = []
        subpath_list: List[Tuple[str, str, str]] = []

        for s in steps:
            edge_list.append(str(s.relation or "related_to"))
            node_list.append(str(s.object))
            subpath_list.append((str(s.subject), str(s.relation or "related_to"), str(s.object)))

        chain = ""
        for i in range(len(node_list) - 1):
            chain += f"{node_list[i]}->[{edge_list[i]}]->"
        chain += node_list[-1]
        return PathAsLists(node_list=node_list, edge_list=edge_list, subpath_list=subpath_list, chain_str=chain)
