# src/modules/kg_build_service.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

from src.modules.kg_schema import KGTriple
from src.modules.kg_store import KGStore
from src.modules.kg_triplet_extractor import KGTripletExtractor


@dataclass(frozen=True)
class KGBuildStats:
    docs_seen: int
    docs_with_triples: int
    triples_forward_total: int
    triples_reverse_total: int
    triples_written_total: int


_REVERSE_RELATION_MAP: Dict[str, str] = {
    "has_symptom": "is_symptom_of",
    "affects": "is_affected_by",
    "diagnosed_by": "diagnoses",
    "treated_with": "treats",
    "managed_by": "manages",
    "has_risk_factor": "is_risk_factor_for",
    "caused_by": "causes",
    "indicates": "is_indicated_by",
    "occurs_in": "has_occurrence_of",
    "used_for": "has_use",
    "involves_medication": "is_involved_in_treatment",
    "has_side_effect": "is_side_effect_of",
    "contraindicated_for": "has_contraindication",
    "detects": "is_detected_by",
    "measures": "is_measured_by",
    "increases_risk_of": "has_increased_risk_due_to",
    "part_of": "has_part",
    "can_affect": "can_be_affected_by",
}

_SYMMETRIC_RELATIONS = {
    "comorbid_with",
    "related_to",
}


def _triple_key(t: KGTriple) -> Tuple[str, str, str, str, str]:
    return (t.subject.lower(), t.relation.lower(), t.object.lower(), t.subject_type, t.object_type)


def _swap_meta_labels(meta: Dict) -> Dict:
    """
    Wenn der Extractor label_subject und label_object gesetzt hat,
    muessen sie fuer synthetische Reverse Kanten getauscht werden.
    """
    if not isinstance(meta, dict):
        return {}

    out = dict(meta)

    ls = out.get("label_subject")
    lo = out.get("label_object")
    if isinstance(ls, str) and isinstance(lo, str) and ls and lo:
        out["label_subject"] = lo
        out["label_object"] = ls

    return out


class KGBuildService:
    def __init__(
        self,
        extractor: KGTripletExtractor,
        add_reverse_edges: bool = True,
        add_symmetric_relations: bool = True,
    ):
        self.extractor = extractor
        self.add_reverse_edges = bool(add_reverse_edges)
        self.add_symmetric_relations = bool(add_symmetric_relations)

    def build_or_load(
        self,
        docs: List[Document],
        cache_path: Path,
        limit: Optional[int] = None,
        chunk_id_prefix: str = "kg",
        force_rebuild: bool = False,
        source_name: str = "medical",
    ) -> Tuple[KGStore, KGBuildStats]:
        kg = KGStore()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not force_rebuild:
            kg.load_jsonl(cache_path)
            stats = KGBuildStats(
                docs_seen=0,
                docs_with_triples=0,
                triples_forward_total=0,
                triples_reverse_total=0,
                triples_written_total=kg.g.number_of_edges(),
            )
            return kg, stats

        docs_seen = 0
        docs_with_triples = 0
        forward_total = 0
        reverse_total = 0

        seen: Set[Tuple[str, str, str, str, str]] = set()

        for i, doc in enumerate(docs):
            if limit is not None and i >= int(limit):
                break

            docs_seen += 1

            meta = doc.metadata or {}
            base_chunk_id = str(meta.get("chunk_id") or f"{chunk_id_prefix}-{i}")

            res = self.extractor.extract(doc, chunk_id=base_chunk_id, source=source_name)
            triples_fwd = res.triples or []
            if not triples_fwd:
                continue

            docs_with_triples += 1
            to_add: List[KGTriple] = []

            for t in triples_fwd:
                k = _triple_key(t)
                if k in seen:
                    continue

                seen.add(k)
                to_add.append(t)
                forward_total += 1

                if self.add_symmetric_relations and t.relation in _SYMMETRIC_RELATIONS:
                    meta_syn = _swap_meta_labels(dict(t.meta or {}))
                    meta_syn.update(
                        {
                            "synthetic": True,
                            "reverse_of": t.relation,
                            "symmetric": True,
                            "relation_raw": str(t.relation).upper(),
                        }
                    )

                    rev = KGTriple(
                        subject=t.object,
                        relation=t.relation,
                        object=t.subject,
                        subject_type=t.object_type,
                        object_type=t.subject_type,
                        source=t.source,
                        chunk_id=t.chunk_id,
                        meta=meta_syn,
                    )

                    rk = _triple_key(rev)
                    if rk not in seen:
                        seen.add(rk)
                        to_add.append(rev)
                        reverse_total += 1
                    continue

                if self.add_reverse_edges:
                    rev_rel = _REVERSE_RELATION_MAP.get(t.relation)
                    if not rev_rel or rev_rel == t.relation:
                        continue

                    meta_syn = _swap_meta_labels(dict(t.meta or {}))
                    meta_syn.update(
                        {
                            "synthetic": True,
                            "reverse_of": t.relation,
                            "relation_raw": str(rev_rel).upper(),
                        }
                    )

                    rev = KGTriple(
                        subject=t.object,
                        relation=rev_rel,
                        object=t.subject,
                        subject_type=t.object_type,
                        object_type=t.subject_type,
                        source=t.source,
                        chunk_id=t.chunk_id,
                        meta=meta_syn,
                    )

                    rk = _triple_key(rev)
                    if rk not in seen:
                        seen.add(rk)
                        to_add.append(rev)
                        reverse_total += 1

            if to_add:
                kg.add_triples(to_add)

        kg.save_jsonl(cache_path)
        stats = KGBuildStats(
            docs_seen=docs_seen,
            docs_with_triples=docs_with_triples,
            triples_forward_total=forward_total,
            triples_reverse_total=reverse_total,
            triples_written_total=kg.g.number_of_edges(),
        )
        return kg, stats
