# src/modules/knowledge_graph/kg_path_service.py

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

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
    MultiDiGraph kompatibel.

    Liefert:
      - shortest_path_nodes: Knotenpfad (undirected)
      - shortest_path: eine deterministische Variante (best edge pro hop)

    Hinweis, Varianten sind hier weiterhin vorhanden für andere Teile deines Codes,
    die Notebook konforme Pipeline nutzt sie nicht mehr.
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

    def _score_edge_data(self, data: Dict[str, Any]) -> Tuple[int, int, str]:
        d = dict(data or {})
        rel = str(d.get("relation") or "").strip()

        observations = list(d.get("observations") or [])
        obs_n = len(observations)

        synthetic_n = 0
        for o in observations:
            meta = o.get("meta") if isinstance(o, dict) else {}
            if isinstance(meta, dict) and bool(meta.get("synthetic", False)):
                synthetic_n += 1

        has_nonsynth = 1 if (obs_n - synthetic_n) > 0 else 0
        return (has_nonsynth, obs_n, rel)

    def _candidate_edges_between(self, u: str, v: str, top_n: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        if not u or not v:
            return [("related_to", {})]

        if not self.kg.g.has_edge(u, v):
            return [("related_to", {})]

        all_data = self.kg.g.get_edge_data(u, v) or {}
        scored: List[Tuple[Tuple[int, int, str, str], str, Dict[str, Any]]] = []

        for k, data in all_data.items():
            d = dict(data or {})
            rel = str(d.get("relation") or "").strip()
            if not rel:
                rels = d.get("relations") or []
                if rels:
                    rel = str(rels[0] or "").strip()

            base = self._score_edge_data(d)
            score = (base[0], base[1], base[2], str(k))
            scored.append((score, rel or "related_to", d))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Tuple[str, Dict[str, Any]]] = []
        for _score, rel, d in scored[: max(1, int(top_n))]:
            out.append((rel or "related_to", d))
        return out if out else [("related_to", {})]

    def _best_edge_between(self, u: str, v: str) -> Tuple[str, Dict[str, Any]]:
        return self._candidate_edges_between(u, v, top_n=1)[0]

    def shortest_path(self, start: str, end: str, cutoff: Optional[int] = None) -> List[KGStep]:
        nodes = self.shortest_path_nodes(start, end, cutoff=cutoff)
        if len(nodes) < 2:
            return []

        steps: List[KGStep] = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            if self.kg.g.has_edge(a, b):
                u, v = a, b
            elif self.kg.g.has_edge(b, a):
                u, v = b, a
            else:
                u, v = a, b

            rel, ed = self._best_edge_between(u, v)
            steps.append(KGStep(subject=u, relation=rel, object=v, edge=ed))
        return steps

    def _steps_to_lists(self, steps: List[KGStep]) -> PathAsLists:
        if not steps:
            return PathAsLists(node_list=[], edge_list=[], subpath_list=[], chain_str="")

        node_list = [str(steps[0].subject)]
        edge_list: List[str] = []
        subpath_list: List[Tuple[str, str, str]] = []

        for s in steps:
            rel = str(s.relation or "related_to")
            u = str(s.subject)
            v = str(s.object)
            edge_list.append(rel)
            node_list.append(v)
            subpath_list.append((u, rel, v))

        chain = ""
        for i in range(len(node_list) - 1):
            chain += f"{node_list[i]}->[{edge_list[i]}]->"
        chain += node_list[-1]

        return PathAsLists(
            node_list=node_list,
            edge_list=edge_list,
            subpath_list=subpath_list,
            chain_str=chain,
        )

    def shortest_path_variants(
        self,
        start: str,
        end: str,
        cutoff: Optional[int] = None,
        per_hop_top_n: int = 3,
        max_chains: int = 12,
    ) -> List[PathAsLists]:
        nodes = self.shortest_path_nodes(start, end, cutoff=cutoff)
        if len(nodes) < 2:
            return []

        hop_candidates: List[List[Tuple[str, Dict[str, Any], str, str]]] = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            if self.kg.g.has_edge(a, b):
                u, v = a, b
            elif self.kg.g.has_edge(b, a):
                u, v = b, a
            else:
                u, v = a, b

            cands = self._candidate_edges_between(u, v, top_n=per_hop_top_n)
            hop_candidates.append([(rel, ed, u, v) for (rel, ed) in cands])

        variants: List[List[KGStep]] = [[]]
        for hop in hop_candidates:
            new_variants: List[List[KGStep]] = []
            for partial in variants:
                for rel, ed, u, v in hop:
                    steps = partial + [KGStep(subject=u, relation=rel, object=v, edge=ed)]
                    new_variants.append(steps)
                    if len(new_variants) >= int(max_chains):
                        break
                if len(new_variants) >= int(max_chains):
                    break
            variants = new_variants
            if not variants:
                break

        out: List[PathAsLists] = []
        seen = set()
        for steps in variants:
            pal = self._steps_to_lists(steps)
            if pal.chain_str in seen:
                continue
            seen.add(pal.chain_str)
            out.append(pal)

        return out
