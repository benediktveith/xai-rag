# src/modules/kgrag_ex_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.modules.rag_engine import RAGEngine
from src.modules.llm_client import LLMClient
from src.modules.kg_path_service import PathAsLists


from src.modules.kg_store import KGStore
from src.modules.kg_query_service import KGQueryService
from src.modules.kg_path_service import KGPathService
from src.modules.kg_schema import KGStep


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

    # Paper nah artifacts for explainer
    kg_chain: str
    kg_paragraph: str
    path_lists: PathAsLists

    # Keep for compatibility with your current code
    kg_context: str

    retrieved: List[RetrievedChunk]
    answer: str
    tokens_in: int
    tokens_out: int
    llm_calls: int


class KGRAGExPipeline:
    """
    Paper close usage:
    - Use KG to find a path
    - Convert path to natural language via LLM, that is the pseudo paragraph
    - Retrieval can use KG paragraph as query expansion, but the explainer operates on chain and paragraph only
    """

    def __init__(self, rag: RAGEngine, kg: KGStore, llm_client: LLMClient):
        self.rag = rag
        self.kg = kg
        self.llm_client = llm_client

        self.query = KGQueryService(kg, llm_client=llm_client)
        self.path = KGPathService(kg)

    def _retrieve_with_scores(self, query: str, k: int) -> List[Tuple[Document, float]]:
        trace = self.rag.retrieve_with_scores(query, k=k)
        out: List[Tuple[Document, float]] = []
        for t in trace:
            meta = (t.get("metadata") or {}).copy()
            meta["chunk_id"] = str(t.get("id", "chunk"))
            doc = Document(page_content=t.get("content", "") or "", metadata=meta)
            score = float(t.get("score", 0.0))
            out.append((doc, score))
        return out

    def _to_chunks(self, docs_with_scores: List[Tuple[Document, float]]) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        for i, (doc, score) in enumerate(docs_with_scores):
            meta = doc.metadata or {}
            content = doc.page_content or ""
            title = str(meta.get("title", meta.get("topic_name", meta.get("subject_name", "Document"))))

            out.append(
                RetrievedChunk(
                    chunk_id=str(meta.get("chunk_id", f"chunk-{i+1}")),
                    score=float(score),
                    title=title,
                    content=content,
                    metadata=meta,
                )
            )
        return out

    def _answer(self, question: str, retrieved: List[RetrievedChunk]) -> tuple[str, int, int, int]:
        llm = self.llm_client.get_llm()
        if llm is None:
            raise RuntimeError("LLMClient.get_llm() returned None")

        context_lines = []
        for c in retrieved:
            txt = (c.content or "").replace("\n", " ").strip()
            context_lines.append(f"[{c.chunk_id}] {txt}")
        context = "\n".join(context_lines)

        extra = """
Output rules (MANDATORY):
Respond with the final answer only.
For multiple choice, respond with only the option letter, for example A or B or C or D.
No additional text.
No explanations.
No punctuation.
No line breaks.
""".strip()

        msg = self.llm_client._create_final_answer_prompt(question, context, extra=extra)
        resp = llm.invoke(msg)
        ans = (getattr(resp, "content", str(resp)) or "").strip()
        return ans, 0, 0, 1

    def _path_to_node_list(self, steps: List[KGStep]) -> List[str]:
        """
        Make a node list representation from KGStep list.
        Robust against KGStep attribute naming differences.
        """
        if not steps:
            return []

        def _s(x: object) -> str:
            return str(x or "").strip()

        nodes: List[str] = []

        first = steps[0]
        subj = _s(getattr(first, "subject", getattr(first, "u", "")))
        obj = _s(getattr(first, "object", getattr(first, "v", "")))

        if subj:
            nodes.append(subj)
        if obj:
            nodes.append(obj)

        for st in steps[1:]:
            o = _s(getattr(st, "object", getattr(st, "v", "")))
            if o:
                nodes.append(o)

        return nodes

    def _path_to_chain_str(self, steps: List[KGStep]) -> str:
        """
        Paper style chain string uses '__' separators for perturbations.
        """
        nodes = self._path_to_node_list(steps)
        return " __ ".join([n for n in nodes if n])

    def _generate_pseudo_paragraph(self, chain_str: str) -> tuple[str, int]:
        llm = self.llm_client.get_llm()
        if llm is None:
            raise RuntimeError("LLMClient.get_llm() returned None")

        chain = (chain_str or "").strip()
        if not chain:
            return "", 0

        prompt = f"""
    You are a medical instructor helping to generate educational materials.
                                                    
    Given a relationship chain connecting medical concepts, write a short, clear and medically accurate paragraph that explains the connections in a way understandable to students.


    Entity->[{{
    relation
    }}]->Entity, with possible "__" placeholders for removed nodes or relations.

    Rules:
    - Write conservative statements, do not invent facts.
    - If you see "__", treat it as missing information and keep the paragraph appropriately vague.
    - Output exactly one paragraph, no bullets, no citations.
    
    Output:
    A single, coherent paragraph explaining the relationships. Do not include any further comments, only your generated answer.

    Chain:
    {chain}
    """.strip()

        resp = llm.invoke(prompt)
        txt = (getattr(resp, "content", str(resp)) or "").strip()
        return txt, 1


    def _answer_from_paragraph(self, paragraph: str, question: str, options: str = "") -> tuple[str, int]:
        llm = self.llm_client.get_llm()
        if llm is None:
            raise RuntimeError("LLMClient.get_llm() returned None")

        ctx = (paragraph or "").replace("\n", " ").strip()
        if not ctx:
            return "", 0

        extra = """
    Output rules (MANDATORY):
    Respond with the final answer only.
    For multiple choice, respond with only the option letter, for example A or B or C or D.
    No additional text.
    No explanations.
    No punctuation.
    No line breaks.
    """.strip()

        full_question = question if not options else f"{question}\n\nOptions:\n{options}"
        msg = self.llm_client._create_final_answer_prompt(full_question, ctx, extra=extra)
        resp = llm.invoke(msg)
        ans = (getattr(resp, "content", str(resp)) or "").strip()
        return ans, 1
    
    def _rag_answer_with_optional_kg_paragraph(
        self,
        question: str,
        kg_paragraph: str,
        top_k_docs: int,
    ) -> tuple[str, List[RetrievedChunk], int]:
        # build retrieval query
        retrieval_query = question
        if (kg_paragraph or "").strip():
            retrieval_query = f"{question}\n\nSupporting KG context:\n{kg_paragraph}"

        docs_with_scores = self._retrieve_with_scores(retrieval_query, k=top_k_docs)
        retrieved = self._to_chunks(docs_with_scores)

        ans, tin, tout, calls_ans = self._answer(question, retrieved)
        return ans, retrieved, calls_ans
    
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


    def run(
        self,
        question_id: str,
        question: str,
        gold_answer: Optional[str],
        entity_k: int = 6,
        top_k_docs: int = 4,
    ) -> KGRAGRun:
        qents = self.query.extract_entities(question, k=entity_k).entities
        pair = self.query.pick_pair_with_path(qents)
        print(pair)
        start, end = (pair if pair else (None, None))

        path_steps: List[KGStep] = []
        if start and end:
            path_steps = self.path.shortest_path(start, end)

        path_lists = self._steps_to_path_as_lists(path_steps)
        print(path_lists)
        kg_chain = self._path_as_chain_str(path_lists)
        print(kg_chain)

        kg_paragraph = ""
        calls_para = 0
        if kg_chain:
            kg_paragraph, calls_para = self._generate_pseudo_paragraph(kg_chain)

        retrieval_query = question
        if kg_paragraph.strip():
            retrieval_query = f"{question}\n\nSupporting KG context:\n{kg_paragraph}"

        docs_with_scores = self._retrieve_with_scores(retrieval_query, k=top_k_docs)
        retrieved = self._to_chunks(docs_with_scores)

        ans, tin, tout, calls_ans = self._answer(question, retrieved)

        return KGRAGRun(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            entities=qents,
            start=start,
            end=end,
            path=path_steps,
            kg_chain=kg_chain,
            kg_paragraph=kg_paragraph,
            path_lists=path_lists,
            kg_context=kg_paragraph,
            retrieved=retrieved,
            answer=ans,
            tokens_in=tin,
            tokens_out=tout,
            llm_calls=int(calls_para) + int(calls_ans),
        )
