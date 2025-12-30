from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.modules.hotpot_baseline_runner import HotpotBaselineRunner, RetrievedChunk
from src.modules.kg_store import KGStore
from src.modules.kg_query_service import KGQueryService
from src.modules.kg_path_service import KGPathService
from src.modules.kg_schema import KGStep


@dataclass
class KGRAGRun:
    question_id: str
    question: str
    gold_answer: Optional[str]
    entities: List[str]
    start: Optional[str]
    end: Optional[str]
    path: List[KGStep]
    kg_context: str
    retrieved: List[RetrievedChunk]
    answer: str
    tokens_in: int
    tokens_out: int
    llm_calls: int


class KGRAGExPipeline:
    def __init__(self, baseline: HotpotBaselineRunner, kg: KGStore, query_llm=None):
        self.baseline = baseline
        self.kg = kg
        self.query = KGQueryService(kg, llm_client=query_llm)
        self.path = KGPathService(kg)

    def run(self, question_id: str, question: str, gold_answer: Optional[str], entity_k: int = 6, top_k_docs: int = 4) -> KGRAGRun:
        qents = self.query.extract_entities(question, k=entity_k).entities
        pair = self.query.pick_pair_with_path(qents)
        start, end = (pair if pair else (None, None))

        path_steps: List[KGStep] = []
        if start and end:
            path_steps = self.path.shortest_path(start, end)

        kg_context = self.path.pseudo_paragraph(path_steps)

        retrieved = self.baseline.retrieve(query=f"{question}\n\nKG Context:\n{kg_context}", k=top_k_docs)

        kg_chunk = RetrievedChunk(
            chunk_id="kg-1",
            score=1.0,
            title="KG Path Context",
            content=kg_context,
            metadata={"source": "kg", "question_id": question_id},
        )

        merged = [kg_chunk] + retrieved
        ans, tin, tout, calls = self.baseline.answer(question=question, retrieved=merged)

        return KGRAGRun(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            entities=qents,
            start=start,
            end=end,
            path=path_steps,
            kg_context=kg_context,
            retrieved=merged,
            answer=ans,
            tokens_in=tin,
            tokens_out=tout,
            llm_calls=calls,
        )
