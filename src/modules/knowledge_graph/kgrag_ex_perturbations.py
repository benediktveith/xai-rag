from dataclasses import dataclass
from typing import List, Tuple

from src.modules.knowledge_graph.kg_path_service import PathAsLists


@dataclass(frozen=True)
class ChainPerturbation:
    """
    Paper-nah:
    Wir perturbieren die Relation-Chain Darstellung (mit "__" Platzhalter),
    nicht den Graph strukturell, und lassen das LLM daraus einen Pseudoparagraphen generieren.
    """
    kind: str  # "node" | "edge" | "subpath"
    removed: str
    chain_str: str
    important_entities: List[str]


class KGPerturbationFactory:
    def generate(self, path_lists: PathAsLists) -> List[ChainPerturbation]:
        if not path_lists.node_list or not path_lists.edge_list:
            return []

        node_list = path_lists.node_list
        edge_list = path_lists.edge_list
        subpath_list = path_lists.subpath_list

        out: List[ChainPerturbation] = []

        # Node perturbations: replace node with "__" in node_list, then reconstruct chain
        for i in range(len(node_list)):
            pert_nodes = node_list[:i] + ["__"] + node_list[i + 1 :]
            chain = ""
            for j in range(len(pert_nodes) - 1):
                chain += f"{pert_nodes[j]}->[{edge_list[j]}]->"
            chain += pert_nodes[-1]
            out.append(
                ChainPerturbation(
                    kind="node",
                    removed=node_list[i],
                    chain_str=chain,
                    important_entities=[node_list[i]],
                )
            )

        # Edge perturbations: replace edge relation with "__"
        for i in range(len(edge_list)):
            pert_edges = edge_list[:i] + ["__"] + edge_list[i + 1 :]
            chain = ""
            for j in range(len(node_list) - 1):
                chain += f"{node_list[j]}->[{pert_edges[j]}]->"
            chain += node_list[-1]
            removed = f"{node_list[i]}::{edge_list[i]}::{node_list[i+1]}"
            out.append(
                ChainPerturbation(
                    kind="edge",
                    removed=removed,
                    chain_str=chain,
                    important_entities=[node_list[i], node_list[i + 1]],
                )
            )

        # Subpath perturbations: replace one triple (u, rel, v) with "__" markers in chain
        for i, (u, rel, v) in enumerate(subpath_list):
            pert_nodes = node_list[:]
            pert_edges = edge_list[:]
            # In chain representation, removing subpath is best approximated by blanking both endpoints and relation
            pert_nodes[i] = "__"
            pert_edges[i] = "__"
            pert_nodes[i + 1] = "__"

            chain = ""
            for j in range(len(pert_nodes) - 1):
                chain += f"{pert_nodes[j]}->[{pert_edges[j]}]->"
            chain += pert_nodes[-1]

            out.append(
                ChainPerturbation(
                    kind="subpath",
                    removed=f"({u}, {rel}, {v})",
                    chain_str=chain,
                    important_entities=[u, v],
                )
            )

        return out
