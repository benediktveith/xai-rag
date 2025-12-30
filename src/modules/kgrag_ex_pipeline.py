from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from src.modules.kg_store import KGStore
from src.modules.kg_query_service import KGQueryService
from src.modules.kg_path_service import KGPathService
from src.modules.kg_schema import KGStep
from src.modules.llm_client import LLMClient
from src.modules.rag_engine import RAGEngine


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _extract_usage(response: Any) -> Tuple[int, int]:
    """
    Best effort token extraction across providers.
    Returns tokens_in, tokens_out.
    Falls back to 0, 0 if not available.
    """
    try:
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            tin = usage.get("input_tokens", usage.get("prompt_tokens", 0))
            tout = usage.get("output_tokens", usage.get("completion_tokens", 0))
            return _safe_int(tin, 0), _safe_int(tout, 0)
    except Exception:
        pass

    try:
        resp_md = getattr(response, "response_metadata", None)
        if isinstance(resp_md, dict):
            usage = resp_md.get("token_usage") or resp_md.get("usage")
            if isinstance(usage, dict):
                tin = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                tout = usage.get("completion_tokens", usage.get("output_tokens", 0))
                return _safe_int(tin, 0), _safe_int(tout, 0)
    except Exception:
        pass

    return 0, 0


@dataclass
class RetrievedChunk:
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]


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
    """
    KGRAG Ex style pipeline without Hotpot dependencies.

    Steps
    1. Extract entities from question
    2. Pick pair with path, compute shortest path in KG
    3. Convert path into pseudo paragraph
    4. Use KG context to guide retrieval in RAGEngine
    5. Generate answer via LLMClient using retrieved context
    """

    def __init__(
        self,
        rag: RAGEngine,
        kg: KGStore,
        llm_client: LLMClient,
        query_llm: Optional[LLMClient] = None,
    ):
        self.rag = rag
        self.kg = kg
        self.llm_client = llm_client

        self.query = KGQueryService(kg, llm_client=query_llm or llm_client)
        self.path = KGPathService(kg)

    def _build_context_from_retrieved(self, retrieved: Sequence[RetrievedChunk], max_chars: int = 12000) -> str:
        """
        Builds a single text context from retrieved chunks.
        Limits size to reduce token cost.
        """
        parts: List[str] = []
        total = 0
        for c in retrieved:
            block = f"[{c.id}] {c.content}".strip()
            if not block:
                continue
            if total + len(block) > max_chars:
                remaining = max(0, max_chars - total)
                if remaining > 0:
                    parts.append(block[:remaining])
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)

    def _retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        """
        Uses RAGEngine retrieval with scores when available.
        Falls back to retrieve_documents if needed.
        """
        try:
            trace = self.rag.retrieve_with_scores(query, k=k)
            out: List[RetrievedChunk] = []
            for item in trace:
                out.append(
                    RetrievedChunk(
                        id=str(item.get("id", "")),
                        score=float(item.get("score", 0.0)),
                        content=str(item.get("content", "")),
                        metadata=dict(item.get("metadata", {}) or {}),
                    )
                )
            return out
        except Exception:
            docs: List[Document] = self.rag.retrieve_documents(query)
            out = []
            for i, d in enumerate(docs[:k], start=1):
                out.append(
                    RetrievedChunk(
                        id=f"chunk-{i}",
                        score=0.0,
                        content=str(d.page_content or ""),
                        metadata=dict(d.metadata or {}),
                    )
                )
            return out

    def answer_with_kg_context(
        self,
        question: str,
        kg_context: str,
        top_k_docs: int = 4,
        extra: str = "",
    ) -> Tuple[str, int, int, int, List[RetrievedChunk]]:
        """
        Executes retrieval and answer generation for a given KG context.
        Returns answer, tokens_in, tokens_out, llm_calls, retrieved_chunks
        """
        retrieval_query = f"{question}\n\nKG Context:\n{kg_context}".strip()
        retrieved = self._retrieve(retrieval_query, k=top_k_docs)

        kg_chunk = RetrievedChunk(
            id="kg-1",
            score=1.0,
            content=kg_context,
            metadata={"source": "kg"},
        )
        merged = [kg_chunk] + retrieved

        context_text = self._build_context_from_retrieved(merged)

        llm = self.llm_client.get_llm()
        prompt = self.llm_client._create_final_answer_prompt(question, context_text, extra=extra)
        resp = llm.invoke(prompt)
        ans = (getattr(resp, "content", str(resp)) or "").strip()

        tin, tout = _extract_usage(resp)
        return ans, tin, tout, 1, merged

    def run(
        self,
        question_id: str,
        question: str,
        gold_answer: Optional[str],
        entity_k: int = 6,
        top_k_docs: int = 4,
        extra: str = "",
    ) -> KGRAGRun:
        qents = self.query.extract_entities(question, k=entity_k).entities
        pair = self.query.pick_pair_with_path(qents)
        start, end = (pair if pair else (None, None))

        path_steps: List[KGStep] = []
        if start and end:
            path_steps = self.path.shortest_path(start, end)

        kg_context = self.path.pseudo_paragraph(path_steps)

        ans, tin, tout, calls, merged = self.answer_with_kg_context(
            question=question,
            kg_context=kg_context,
            top_k_docs=top_k_docs,
            extra=extra,
        )

        return KGRAGRun(
            question_id=str(question_id),
            question=str(question),
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
