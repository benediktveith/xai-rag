from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.modules.kg_store import KGStore
from src.modules.kg_triplet_extractor import KGTripletExtractor


@dataclass(frozen=True)
class KGBuildStats:
    docs_seen: int
    docs_with_triples: int
    triples_total: int


class KGBuildService:
    def __init__(self, extractor: KGTripletExtractor):
        self.extractor = extractor

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
            stats = KGBuildStats(docs_seen=0, docs_with_triples=0, triples_total=kg.g.number_of_edges())
            return kg, stats

        n = 0
        docs_with_triples = 0
        triples_total = 0

        for i, doc in enumerate(docs):
            if limit is not None and i >= limit:
                break

            chunk_id = f"{chunk_id_prefix}-{i}"
            res = self.extractor.extract(doc, chunk_id=chunk_id, source=source_name)
            n += 1

            if res.triples:
                docs_with_triples += 1
                triples_total += len(res.triples)
                kg.add_triples(res.triples)

        kg.save_jsonl(cache_path)
        stats = KGBuildStats(docs_seen=n, docs_with_triples=docs_with_triples, triples_total=triples_total)
        return kg, stats
