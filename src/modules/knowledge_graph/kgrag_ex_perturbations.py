from dataclasses import dataclass
from typing import List, Set, Tuple

from src.modules.knowledge_graph.kg_path_service import PathAsLists


@dataclass(frozen=True)
class ChainPerturbation:
    """
    Paper nah:
    Perturbiere die Chain Darstellung mit "__" Platzhaltern.
    """
    kind: str  # "node" | "edge" | "subpath"
    removed: str
    chain_str: str
    important_entities: List[str]


class KGPerturbationFactory:
    def __init__(self, skip_endpoints_in_node_perturb: bool = True):
        self.skip_endpoints_in_node_perturb = bool(skip_endpoints_in_node_perturb)

    def _build_chain(self, nodes: List[str], edges: List[str]) -> str:
        if len(nodes) < 2 or len(edges) < 1:
            return ""
        parts: List[str] = []
        for i in range(len(nodes) - 1):
            rel = edges[i] if i < len(edges) else ""
            parts.append(f"{nodes[i]}->[{rel}]->")
        parts.append(nodes[-1])
        return "".join(parts)

    def generate(self, path_lists: PathAsLists) -> List[ChainPerturbation]:
        node_list = list(path_lists.node_list or [])
        edge_list = list(path_lists.edge_list or [])
        subpath_list = list(path_lists.subpath_list or [])

        if len(node_list) < 2 or len(edge_list) < 1:
            return []

        out: List[ChainPerturbation] = []
        seen: Set[Tuple[str, str]] = set()  # (kind, chain_str)

        # Node perturbations: ersetze genau einen Knoten durch "__"
        idxs = range(len(node_list))
        if self.skip_endpoints_in_node_perturb and len(node_list) >= 2:
            idxs = range(1, len(node_list) - 1)  # ohne Start/End

        for i in idxs:
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

        # Edge perturbations: ersetze genau eine Relation durch "__"
        for i in range(len(edge_list)):
            pert_edges = edge_list[:]
            pert_edges[i] = "__"
            chain = self._build_chain(node_list, pert_edges)
            removed = f"{node_list[i]}::{edge_list[i]}::{node_list[i+1]}"
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

        # Subpath perturbations: Tripel Removal als "Relation blanken" an genau dieser Stelle
        # Das bleibt strukturell näher am Gedanken "remove one (u, rel, v)" ohne zusätzliche Node Removal.
        for i, (u, rel, v) in enumerate(subpath_list):
            if i >= len(edge_list):
                continue
            pert_edges = edge_list[:]
            pert_edges[i] = "__"
            chain = self._build_chain(node_list, pert_edges)

            removed_str = f"({u}, {rel}, {v})"
            key = ("subpath", chain)
            if key in seen:
                continue
            seen.add(key)

            out.append(
                ChainPerturbation(
                    kind="subpath",
                    removed=removed_str,
                    chain_str=chain,
                    important_entities=[u, v],
                )
            )

        return out
