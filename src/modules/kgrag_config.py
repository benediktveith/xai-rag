from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class KGRAGConfig:
    kg_cache_path: Path
    vector_db_path: Path

    max_docs_for_kg: Optional[int] = None
    max_docs_for_vectorstore: Optional[int] = None

    chunk_id_prefix: str = "hp"
    entity_k: int = 6
    top_k_docs: int = 4

    rag_ex_window_tokens: int = 30
    rag_ex_max_perturbations: int = 40

    llm_max_retries: int = 2
