from __future__ import annotations
from typing import List

import networkx as nx

from src.modules.kg_schema import KGStep
from src.modules.kg_store import KGStore


class KGPathService:
    def __init__(self, kg: KGStore):
        self.kg = kg

    def shortest_path(self, start: str, end: str) -> List[KGStep]:
        G_und = self.kg.g.to_undirected(as_view=True)
        try:
            nodes = nx.shortest_path(G_und, start, end)
        except Exception:
            return []

        steps: List[KGStep] = []

        for a, b in zip(nodes[:-1], nodes[1:]):
            u, v = None, None
            data = None

            if self.kg.g.has_edge(a, b):
                u, v = a, b
                data = self.kg.g.get_edge_data(a, b)
            elif self.kg.g.has_edge(b, a):
                u, v = b, a
                data = self.kg.g.get_edge_data(b, a)
            else:
                # Should not happen if undirected path is derived from this graph, but keep robust
                continue

            # MultiDiGraph returns dict of keys -> attr dict
            if isinstance(data, dict):
                first_key = sorted(data.keys())[0]
                data = data[first_key]
            data = data or {}

            steps.append(
                KGStep(
                    subject=u,
                    relation=str(data.get("relation", "")),
                    object=v,
                    edge=data,
                )
            )

        return steps

    def pseudo_paragraph(self, path: List[KGStep]) -> str:
        if not path:
            return "No KG path available."
        sents = []
        for s in path:
            st = self.kg.node_type(s.subject)
            ot = self.kg.node_type(s.object)
            rel = s.relation.replace("_", " ").strip()
            sents.append(f"({s.subject} [{st}]) {rel} ({s.object} [{ot}]).")
        return " ".join(sents)
