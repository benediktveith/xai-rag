from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.modules.data_loader import DataLoader
from src.modules.rag_engine import RAGEngine
from src.modules.llm_client import LLMClient
from src.modules.strict_llm_formatter import StrictLLMFormatter


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    title: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class BaselineRun:
    question_id: str
    question: str
    gold_answer: Optional[str]
    retrieved: List[RetrievedChunk]
    answer: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    llm_calls: int = 0


class HotpotBaselineRunner:
    def __init__(self, rag: RAGEngine, llm_client: LLMClient, formatter: Optional[StrictLLMFormatter] = None):
        self.rag = rag
        self.llm_client = llm_client
        self.formatter = formatter or StrictLLMFormatter()

    def load_hotpot(self, limit: Optional[int] = None, as_documents: bool = True):
        loader = DataLoader()
        return loader.setup(as_documents=as_documents, limit=limit)

    def setup_vectorstore(self, documents, reset: bool = False):
        self.rag.setup(documents=documents, reset=reset)

    def retrieve(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        trace = self.rag.retrieve_with_scores(query, k=k)
        out: List[RetrievedChunk] = []
        for t in trace:
            out.append(
                RetrievedChunk(
                    chunk_id=str(t.get("id")),
                    score=float(t.get("score", 0.0)),
                    title=str(t.get("title", "")),
                    content=str(t.get("content", "")),
                    metadata=dict(t.get("metadata", {})),
                )
            )
        return out

    def answer(self, question: str, retrieved: List[RetrievedChunk]) -> tuple[str, int, int, int]:
        retrieval_trace = [
            {"id": c.chunk_id, "score": c.score, "title": c.title, "content": c.content, "metadata": c.metadata}
            for c in retrieved
        ]
        messages = self.formatter.build_messages(question=question, retrieval_trace=retrieval_trace)

        llm = self.llm_client.get_llm()
        resp = llm.invoke(messages)
        content = (getattr(resp, "content", str(resp)) or "").strip()

        token_in = None
        token_out = None
        llm_calls = 1

        try:
            usage = getattr(resp, "usage_metadata", None) or getattr(resp, "response_metadata", None)
            if isinstance(usage, dict):
                token_in = usage.get("input_tokens") or usage.get("prompt_tokens")
                token_out = usage.get("output_tokens") or usage.get("completion_tokens")
        except Exception:
            pass

        return content, int(token_in) if token_in else 0, int(token_out) if token_out else 0, llm_calls

    def run(self, question_id: str, question: str, gold_answer: Optional[str] = None, k: int = 4) -> BaselineRun:
        retrieved = self.retrieve(query=question, k=k)
        ans, tin, tout, calls = self.answer(question=question, retrieved=retrieved)
        return BaselineRun(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            retrieved=retrieved,
            answer=ans,
            tokens_in=tin,
            tokens_out=tout,
            llm_calls=calls,
        )
