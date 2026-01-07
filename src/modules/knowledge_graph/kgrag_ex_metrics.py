from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx

from src.modules.knowledge_graph.kg_schema import KGStep
from src.modules.knowledge_graph.kg_store import KGStore


@dataclass(frozen=True)
class CriticalPosition:
    kind: str
    removed: str
    relative_pos: float


class KGRAGGraphMetrics:
    def __init__(self, kg: KGStore):
        self.kg = kg
        self._edge_betweenness: Dict[Tuple[str, str], float] = {}
        self._computed = False

    def compute_global_metrics(self) -> None:
        if self._computed:
            return

        G = self.kg.g.to_undirected()
        raw = nx.edge_betweenness_centrality(G)

        merged: Dict[Tuple[str, str], float] = {}
        for key, val in raw.items():
            if isinstance(key, tuple) and len(key) == 3:
                u, v, _k = key
            else:
                u, v = key

            a, b = (u, v) if str(u) <= str(v) else (v, u)
            prev = merged.get((str(a), str(b)), 0.0)
            if float(val) > prev:
                merged[(str(a), str(b))] = float(val)

        self._edge_betweenness = merged
        self._computed = True

    def node_degree(self, node: str) -> int:
        return int(self.kg.g.degree(node))

    def edge_betweenness(self, u: str, v: str) -> float:
        self.compute_global_metrics()
        return float(self._edge_betweenness.get((u, v), 0.0))

    def subpath_score(self, u: str, v: str) -> float:
        eb = self.edge_betweenness(u, v)
        deg_sum = float(self.node_degree(u) + self.node_degree(v))
        if deg_sum <= 0:
            return 0.0
        return float(eb / deg_sum)

    @staticmethod
    def relative_positions_for_path(kind: str, path_len: int, removed_index: int, removed_label: str) -> CriticalPosition:
        if path_len <= 1:
            pos = 0.0
        else:
            pos = float(removed_index) / float(path_len - 1)
        return CriticalPosition(kind=kind, removed=removed_label, relative_pos=pos)

    def node_type(self, node: str) -> str:
        return self.kg.node_type(node)
