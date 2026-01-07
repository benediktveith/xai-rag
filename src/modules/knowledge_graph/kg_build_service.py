from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

from src.modules.knowledge_graph.kg_schema import KGTriple
from src.modules.knowledge_graph.kg_store import KGStore
from src.modules.knowledge_graph.kg_triplet_extractor import KGTripletExtractor
from src.modules.knowledge_graph.relation_registry import RelationRegistry, ProposedRelation

@dataclass(frozen=True)
class KGBuildStats:
    docs_seen: int
    docs_with_triples: int
    triples_forward_total: int
    triples_reverse_total: int
    triples_written_total: int


_REVERSE_RELATION_MAP: Dict[str, str] = {
    "HAS_SYMPTOM": "IS_SYMPTOM_OF",
    "AFFECTS": "IS_AFFECTED_BY",
    "DIAGNOSED_BY": "DIAGNOSES",
    "TREATED_WITH": "TREATS",
    "MANAGED_BY": "MANAGES",
    "HAS_RISK_FACTOR": "IS_RISK_FACTOR_FOR",
    "CAUSED_BY": "CAUSES",
    "INDICATES": "IS_INDICATED_BY",
    "OCCURS_IN": "HAS_OCCURRENCE_OF",
    "USED_FOR": "HAS_USE",
    "INVOLVES_MEDICATION": "IS_INVOLVED_IN_TREATMENT",
    "HAS_SIDE_EFFECT": "IS_SIDE_EFFECT_OF",
    "CONTRAINDICATED_FOR": "HAS_CONTRAINDICATION",
    "DETECTS": "IS_DETECTED_BY",
    "MEASURES": "IS_MEASURED_BY",
    "INCREASES_RISK_OF": "HAS_INCREASED_RISK_DUE_TO",
    "PART_OF": "HAS_PART",
    "CAN_AFFECT": "CAN_BE_AFFECTED_BY",
}

_SYMMETRIC_RELATIONS = {
    "COMORBID_WITH",
    "RELATED_TO",
}

def _triple_key(t: KGTriple) -> Tuple[str, str, str, str, str]:
    return (t.subject.lower(), t.relation.upper(), t.object.lower(), t.subject_type, t.object_type)


def _swap_meta_labels(meta: Dict) -> Dict:
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
        relation_registry: RelationRegistry,
        registry_cache_path: Path,
        add_reverse_edges: bool = True,
        add_symmetric_relations: bool = True,
        auto_accept_reverse_relations: bool = True,
    ):
        self.extractor = extractor
        self.registry = relation_registry
        self.registry_cache_path = registry_cache_path
        self.add_reverse_edges = bool(add_reverse_edges)
        self.add_symmetric_relations = bool(add_symmetric_relations)
        self.auto_accept_reverse_relations = bool(auto_accept_reverse_relations)

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

        # Load registry if present
        if self.registry_cache_path.exists():
            loaded = RelationRegistry.load(self.registry_cache_path)
            self.registry.allowed = loaded.allowed
            self.registry.aliases = loaded.aliases
            self.registry.pending = loaded.pending

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

                rel = str(t.relation or "").upper()

                # symmetric
                if self.add_symmetric_relations and rel in _SYMMETRIC_RELATIONS:
                    meta_syn = _swap_meta_labels(dict(t.meta or {}))
                    meta_syn.update(
                        {
                            "synthetic": True,
                            "reverse_of": rel,
                            "symmetric": True,
                            "relation_raw": rel,
                        }
                    )

                    rev = KGTriple(
                        subject=t.object,
                        relation=rel,
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

                # reverse
                if self.add_reverse_edges:
                    rev_rel = _REVERSE_RELATION_MAP.get(rel)
                    if not rev_rel or rev_rel == rel:
                        continue

                    if self.auto_accept_reverse_relations:
                        self.registry.accept(rev_rel)
                    else:
                        # mindestens als pending aufnehmen
                        self.registry.propose(
                            ProposedRelation(name=rev_rel, description=f"Auto reverse of {rel}"),
                            alias_threshold=0.95,
                        )

                    meta_syn = _swap_meta_labels(dict(t.meta or {}))
                    meta_syn.update(
                        {
                            "synthetic": True,
                            "reverse_of": rel,
                            "relation_raw": rev_rel,
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

            self.registry.save(self.registry_cache_path)

        kg.save_jsonl(cache_path)
        self.registry.save(self.registry_cache_path)

        stats = KGBuildStats(
            docs_seen=docs_seen,
            docs_with_triples=docs_with_triples,
            triples_forward_total=forward_total,
            triples_reverse_total=reverse_total,
            triples_written_total=kg.g.number_of_edges(),
        )
        return kg, stats
