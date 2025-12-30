from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document

from src.modules.kg_store import KGStore
from src.modules.kg_triplet_extractor import KGTripletExtractor
from src.modules.kg_schema import KGTriple


@dataclass
class KGBuildStats:
    docs_processed: int
    triples_extracted: int


class HotpotKGBuildService:
    def __init__(self, extractor: KGTripletExtractor):
        self.extractor = extractor

    def build_or_load(
        self,
        docs: List[Document],
        cache_path: Path,
        limit: Optional[int] = None,
        chunk_id_prefix: str = "hp",
        force_rebuild: bool = False,
    ) -> tuple[KGStore, KGBuildStats]:
        store = KGStore()

        if cache_path.exists() and not force_rebuild:
            store.load_jsonl(cache_path)
            return store, KGBuildStats(docs_processed=0, triples_extracted=store.g.number_of_edges())

        triples_all: List[KGTriple] = []
        docs_use = docs[:limit] if isinstance(limit, int) else docs

        for i, doc in enumerate(docs_use):
            meta = doc.metadata or {}
            qid = str(meta.get("question_id", "unknown"))
            source = str(meta.get("title", "hotpotqa"))
            chunk_id = f"{chunk_id_prefix}-{qid}-{i}"

            res = self.extractor.extract(doc, chunk_id=chunk_id, source=source)
            triples_all.extend(res.triples)

        store.add_triples(triples_all)
        store.save_jsonl(cache_path)

        return store, KGBuildStats(docs_processed=len(docs_use), triples_extracted=len(triples_all))
