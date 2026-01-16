from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class KGTriple:
    # Atomic knowledge graph fact as extracted from source text.
    # Represents a directed relation with typed subject and object, plus provenance metadata.
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
    # Single step in a KG path, binding subject, relation, and object,
    # with edge-level metadata used for scoring or explanation.
    subject: str
    relation: str
    object: str
    edge: Dict[str, Any]
