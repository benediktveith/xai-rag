# src/modules/knowledge_graph/kgrag_ex_perturbations.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

from src.modules.knowledge_graph.kg_path_service import PathAsLists


@dataclass(frozen=True)
class ChainPerturbation:
    kind: str
    removed: str
    chain_str: str
    important_entities: List[str]


class KGPerturbationFactory:
    """
    Notebook konform:
    - node, ersetze genau einen Knoten durch "__", inklusive Endpunkte
    - edge, ersetze genau eine Relation durch "__"
    - subpath, ersetze ein Tripel (u, rel, v) als Block ("__","__","__"), dann Notebook Cleaning und format_edge_path
    """

    def _build_chain(self, nodes: List[str], edges: List[str]) -> str:
        if len(nodes) < 2 or len(edges) < 1:
            return ""
        out = ""
        for i in range(len(nodes) - 1):
            rel = edges[i] if i < len(edges) else ""
            out += f"{nodes[i]}->[{rel}]->"
        out += nodes[-1]
        return out

    def _format_edge_path(self, elements: List[str]) -> str:
        if not elements:
            return ""
        out = ""
        for i in range(0, len(elements) - 2, 2):
            out += f"{elements[i]}->[{elements[i + 1]}]->"
        out += elements[-1]
        return out

    def _remove_duplicates_except_underscores(self, lst: List[str]) -> List[str]:
        seen = set()
        res = []
        for x in lst:
            if x == "__":
                res.append(x)
            elif x not in seen:
                seen.add(x)
                res.append(x)
        return res

    def _remove_direct_neighbors_only(self, data: List[str]) -> List[str]:
        idx_rm = set()
        for i, v in enumerate(data):
            if v == "__":
                if i > 0 and data[i - 1] != "__":
                    idx_rm.add(i - 1)
                if i < len(data) - 1 and data[i + 1] != "__":
                    idx_rm.add(i + 1)
        return [v for i, v in enumerate(data) if i not in idx_rm]

    def generate(self, path_lists: PathAsLists) -> List[ChainPerturbation]:
        node_list = list(path_lists.node_list or [])
        edge_list = list(path_lists.edge_list or [])
        subpath_list = list(path_lists.subpath_list or [])

        if len(node_list) < 2 or len(edge_list) < 1:
            return []

        out: List[ChainPerturbation] = []
        seen: Set[Tuple[str, str]] = set()

        for i in range(len(node_list)):
            pert_nodes = node_list[:]
            pert_nodes[i] = "__"
            chain = self._build_chain(pert_nodes, edge_list)
            key = ("node", chain)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                ChainPerturbation(
                    kind="node",
                    removed=node_list[i],
                    chain_str=chain,
                    important_entities=[node_list[i]],
                )
            )

        for i in range(len(edge_list)):
            pert_edges = edge_list[:]
            pert_edges[i] = "__"
            chain = self._build_chain(node_list, pert_edges)
            removed = f"{node_list[i]}::{edge_list[i]}::{node_list[i + 1]}"
            key = ("edge", chain)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                ChainPerturbation(
                    kind="edge",
                    removed=removed,
                    chain_str=chain,
                    important_entities=[node_list[i], node_list[i + 1]],
                )
            )

        for i in range(len(subpath_list)):
            pert = subpath_list[:]
            removed_trip = subpath_list[i]
            pert[i] = ("__", "__", "__")

            elements: List[str] = []
            for (u, rel, v) in pert:
                elements.append(u)
                elements.append(rel)
                elements.append(v)

            elements = self._remove_duplicates_except_underscores(elements)
            elements = self._remove_direct_neighbors_only(elements)

            chain = self._format_edge_path(elements)
            removed_str = f"({removed_trip[0]}, {removed_trip[1]}, {removed_trip[2]})"

            key = ("subpath", chain)
            if key in seen:
                continue
            seen.add(key)

            out.append(
                ChainPerturbation(
                    kind="subpath",
                    removed=removed_str,
                    chain_str=chain,
                    important_entities=[removed_trip[0], removed_trip[2]],
                )
            )

        return out
