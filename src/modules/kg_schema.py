from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class KGTriple:
    subject: str
    relation: str
    object: str
    subject_type: str
    object_type: str
    source: str
    chunk_id: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class KGStep:
    subject: str
    relation: str
    object: str
    edge: Dict[str, Any]


@dataclass(frozen=True)
class KGNodeStats:
    degree: int
    in_degree: int
    out_degree: int


@dataclass(frozen=True)
class KGEdgeStats:
    betweenness: float


@dataclass(frozen=True)
class KGSubpathStats:
    score: float
