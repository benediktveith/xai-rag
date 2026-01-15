# src/modules/kgragex/kgragex_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal

import inspect
from pydantic import BaseModel, Field

from src.modules.rag.rag_engine import RAGEngine
from src.modules.llm.llm_client import LLMClient

from src.modules.knowledge_graph.kg_path_service import PathAsLists
from src.modules.knowledge_graph.kg_store import KGStore
from src.modules.knowledge_graph.kg_query_service import KGQueryService
from src.modules.knowledge_graph.kg_path_service import KGPathService
from src.modules.knowledge_graph.kg_schema import KGStep


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    title: str
    content: str
    metadata: dict


@dataclass
class KGRAGRun:
    question_id: str
    question: str
    gold_answer: Optional[str]

    entities: List[str]
    start: Optional[str]
    end: Optional[str]
    path: List[KGStep]

    kg_chain: str
    kg_paragraph: str
    path_lists: List[PathAsLists]

    kg_context: str

    retrieved: List[RetrievedChunk]
    answer: str
    tokens_in: int
    tokens_out: int
    llm_calls: int


class _MCQAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D"] = Field(...)


class KGRAGExPipeline:
    """
    Notebook konform:
    - LLM extrahiert Entity Paare
    - Für jedes Paar shortest path im KG
    - Pfad zu pseudo paragraph, join über alle Pfade
    - Antwort nur aus joined paragraph, question, options
    """

    def __init__(self, kg: KGStore, llm_client: LLMClient):
        self.kg = kg
        self.llm_client = llm_client

        self.query = KGQueryService(kg, llm_client=llm_client)
        self.path = KGPathService(kg)

    def _create_mcq_prompt(self, paragraph: str, question: str, options: str = "") -> str:
        full_question = question if not options else f"{question}\n\nOptions:\n{options}"
        return f"""
        You are a knowledgeable medical assistant.

        Use the following medical paragraph to answer the multiple-choice question below.
        Choose the best answer from the provided options based solely on the paragraph.

        Medical Paragraph:
        {paragraph}

        Question:
        {full_question}

        Answer:
        Provide only the letter (A, B, C or D) corresponding to the corrected choice.
        Do not provide any additional explanation or commentary.
        """.strip()

    def _generate_pseudo_paragraph(self, chain_str: str) -> tuple[str, int, int, int]:
        llm = self.llm_client.get_llm()
        if llm is None:
            raise RuntimeError("LLMClient.get_llm() returned None")

        chain = (chain_str or "").strip()
        if not chain:
            return "", 0, 0, 0

        s = chain.replace("__", "")
        s = (
            s.replace("->", "")
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace("{", "")
            .replace("}", "")
            .replace("_", "")
            .replace(" ", "")
            .replace("\n", "")
            .replace("\t", "")
        )
        if not any(ch.isalnum() for ch in s):
            return "", 0, 0, 0

        prompt = f"""
            You are a medical instructor helping to generate educational materials.
                                                                
            Given a relationship chain connecting medical concepts, write a short, clear and medically accurate paragraph that explains the connections in a way understandable to students.
                                                                    
            Input chain:
            {chain}
                                                                    
            Output:
            A single, coherent paragraph explaining the relationships. Do not include any further comments, only your generated answer.
            """.strip()

        resp = llm.invoke(prompt)
        tin, tout = self.llm_client._extract_token_usage(resp)
        txt = (getattr(resp, "content", str(resp)) or "").strip()
        return txt, 1, tin, tout

    def _answer_from_paragraph(self, paragraph: str, question: str, options: str = "") -> tuple[str, int, int, int]:
        llm = self.llm_client.get_llm()
        if llm is None:
            raise RuntimeError("LLMClient.get_llm() returned None")

        ctx = (paragraph or "").strip()
        if not ctx:
            return "", 0, 0, 0

        msg = self._create_mcq_prompt(ctx, question, options=options)
        if hasattr(llm, "with_structured_output"):
            try:
                llm_so = llm.with_structured_output(_MCQAnswer, include_raw=True)
                resp = llm_so.invoke(msg)
                tin, tout = self.llm_client._extract_token_usage(resp)
                ans = None
                if isinstance(resp, dict):
                    parsed = resp.get("parsed")
                    ans = getattr(parsed, "answer", None)
                else:
                    ans = getattr(resp, "answer", None)
                if isinstance(ans, str) and ans in ("A", "B", "C", "D"):
                    return ans, 1, tin, tout
            except Exception:
                pass

        resp = llm.invoke(msg)
        tin, tout = self.llm_client._extract_token_usage(resp)
        txt = (getattr(resp, "content", str(resp)) or "").strip()
        return (txt[:1].upper() if txt else ""), 1, tin, tout

    def _path_as_chain_str(self, path_lists: PathAsLists) -> str:
        nl = path_lists.node_list or []
        el = path_lists.edge_list or []
        if len(nl) < 2 or len(el) < 1:
            return ""

        chain = ""
        for j in range(len(nl) - 1):
            rel = el[j] if j < len(el) else ""
            chain += f"{nl[j]}->[{rel}]->"
        chain += nl[-1]
        return chain

    def _steps_to_path_as_lists(self, steps: List[KGStep]) -> PathAsLists:
        if not steps:
            return PathAsLists(node_list=[], edge_list=[], subpath_list=[], chain_str="")

        def _s(x: object) -> str:
            return str(x or "").strip()

        node_list: List[str] = [_s(steps[0].subject)]
        edge_list: List[str] = []
        subpath_list: List[tuple[str, str, str]] = []

        for st in steps:
            rel = _s(st.relation) or "related_to"
            obj = _s(st.object)
            edge_list.append(rel)
            subpath_list.append((_s(st.subject), rel, obj))
            node_list.append(obj)

        chain = ""
        for i in range(len(node_list) - 1):
            chain += f"{node_list[i]}->[{edge_list[i]}]->"
        chain += node_list[-1] if node_list else ""
    
        return PathAsLists(
            node_list=node_list,
            edge_list=edge_list,
            subpath_list=subpath_list,
            chain_str=chain,
        )

    def _path_lists_to_steps(self, path_lists: PathAsLists) -> List[KGStep]:
        out: List[KGStep] = []

        try:
            sig = inspect.signature(KGStep)
            params = set(sig.parameters.keys())
        except Exception:
            params = {"subject", "relation", "object", "edge"}

        for (u, rel, v) in (path_lists.subpath_list or []):
            kwargs: Dict[str, Any] = {}

            if "subject" in params:
                kwargs["subject"] = str(u)
            if "relation" in params:
                kwargs["relation"] = str(rel)
            if "object" in params:
                kwargs["object"] = str(v)

            if "u" in params and "subject" not in kwargs:
                kwargs["u"] = str(u)
            if "v" in params and "object" not in kwargs:
                kwargs["v"] = str(v)

            if "edge" in params and "edge" not in kwargs:
                kwargs["edge"] = {"relation": str(rel), "u": str(u), "v": str(v)}

            try:
                out.append(KGStep(**kwargs))
            except TypeError:
                if "edge" in params:
                    kwargs["edge"] = str(rel)
                out.append(KGStep(**kwargs))

        return out

    def run(
        self,
        question_id: str,
        question: str,
        gold_answer: Optional[str],
        options: str = "",
    ) -> KGRAGRun:
        q = self.query.extract_entities(question)
        qents = q.entities
        pairs = q.pairs

        chains: List[str] = []
        path_lists_used: List[PathAsLists] = []
        primary_steps: List[KGStep] = []

        for a_raw, b_raw in pairs:
            a = self.query.map_to_node(a_raw)
            b = self.query.map_to_node(b_raw)
            if not a or not b or a == b:
                continue

            steps = self.path.shortest_path(a, b)
            if not steps:
                continue

            pl = self._steps_to_path_as_lists(steps)
            chain = self._path_as_chain_str(pl)
            if not chain:
                continue

            if not primary_steps:
                primary_steps = steps

            chains.append(chain)
            path_lists_used.append(pl)

        if not chains:
            mapped: List[str] = []
            for e in qents:
                m = self.query.map_to_node(e)
                if m and m not in mapped:
                    mapped.append(m)

            n = len(mapped)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = mapped[i], mapped[j]
                    steps = self.path.shortest_path(a, b)
                    if not steps:
                        continue

                    pl = self._steps_to_path_as_lists(steps)
                    chain = self._path_as_chain_str(pl)
                    if not chain:
                        continue

                    if not primary_steps:
                        primary_steps = steps

                    chains.append(chain)
                    path_lists_used.append(pl)

        llm_calls = 0
        tokens_in = 0
        tokens_out = 0
        paragraphs: List[str] = []
        for ch in chains:
            p, c, tin, tout = self._generate_pseudo_paragraph(ch)
            llm_calls += int(c)
            tokens_in += int(tin)
            tokens_out += int(tout)
            if p:
                paragraphs.append(p)
        joined_paragraphs = "\n".join(paragraphs).strip()
        ans = ""
        if joined_paragraphs:
            ans, c_ans, tin, tout = self._answer_from_paragraph(joined_paragraphs, question, options=options)
            llm_calls += int(c_ans)
            tokens_in += int(tin)
            tokens_out += int(tout)

        start = str(primary_steps[0].subject) if primary_steps else None
        end = str(primary_steps[-1].object) if primary_steps else None

        primary_chain = chains[0] if chains else ""

        return KGRAGRun(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            entities=qents,
            start=start,
            end=end,
            path=primary_steps,
            kg_chain=primary_chain,
            kg_paragraph=joined_paragraphs,
            path_lists=path_lists_used,
            kg_context=joined_paragraphs,
            retrieved=[],
            answer=ans,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            llm_calls=llm_calls,
        )
