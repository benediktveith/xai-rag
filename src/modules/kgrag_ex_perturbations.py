from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from src.modules.kg_schema import KGStep


@dataclass(frozen=True)
class Perturbation:
    kind: str
    removed: str
    path: List[KGStep]


class KGPerturbationFactory:
    def node_perturbations(self, path: List[KGStep]) -> List[Perturbation]:
        nodes = []
        for s in path:
            nodes.append(s.subject)
            nodes.append(s.object)
        uniq = []
        for n in nodes:
            if n not in uniq:
                uniq.append(n)

        out: List[Perturbation] = []
        for n in uniq:
            new_path = [s for s in path if s.subject != n and s.object != n]
            out.append(Perturbation(kind="node", removed=n, path=new_path))
        return out

    def edge_perturbations(self, path: List[KGStep]) -> List[Perturbation]:
        out: List[Perturbation] = []
        for s in path:
            key = f"{s.subject}::{s.relation}::{s.object}"
            new_path = [x for x in path if f"{x.subject}::{x.relation}::{x.object}" != key]
            out.append(Perturbation(kind="edge", removed=key, path=new_path))
        return out

    def subpath_perturbations(self, path: List[KGStep]) -> List[Perturbation]:
        out: List[Perturbation] = []
        for i, s in enumerate(path):
            removed = f"({s.subject}, {s.relation}, {s.object})"
            new_path = path[:i] + path[i + 1 :]
            out.append(Perturbation(kind="subpath", removed=removed, path=new_path))
        return out
