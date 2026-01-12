# src/modules/kgragex/kgragex_pipeline.py

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import inspect

from src.modules.rag.rag_engine import RAGEngine
from src.modules.llm.llm_client import LLMClient

from src.modules.knowledge_graph.kg_path_service import PathAsLists
from src.modules.knowledge_graph.kg_store import KGStore
from src.modules.knowledge_graph.kg_query_service import KGQueryService
from src.modules.knowledge_graph.kg_path_service import KGPathService
from src.modules.knowledge_graph.kg_schema import KGStep


@dataclass
class RetrievedChunk:
    # bleibt für Kompatibilität, wird in Notebook Logik nicht genutzt
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


class KGRAGExPipeline:
    """
    Notebook konform:
    - LLM extrahiert Entity Paare
    - Für jedes Paar shortest path im KG
    - Pfad zu pseudo paragraph, join über alle Pfade
    - Antwort nur aus joined paragraph, question, options, kein Retrieval
    """

    def __init__(self, rag: RAGEngine, kg: KGStore, llm_client: LLMClient):
        self.rag = rag  # nicht genutzt, bleibt kompatibel
        self.kg = kg
        self.llm_client = llm_client

        self.query = KGQueryService(kg, llm_client=llm_client)
        self.path = KGPathService(kg)

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

Entity->[{{relation}}]->Entity, with possible "__" placeholders for removed nodes or relations.

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
            paragraphs: List[str] = []
            for ch in chains:
                p, c = self._generate_pseudo_paragraph(ch)
                llm_calls += int(c)
                if p:
                    paragraphs.append(p)

            joined_paragraphs = "\n".join(paragraphs).strip()

            ans = ""
            if joined_paragraphs:
                ans, c_ans = self._answer_from_paragraph(joined_paragraphs, question, options=options)
                llm_calls += int(c_ans)

            start = str(primary_steps[0].subject) if primary_steps else None
            end = str(primary_steps[-1].object) if primary_steps else None
            kg_chain = chains[0] if chains else ""

            return KGRAGRun(
                question_id=question_id,
                question=question,
                gold_answer=gold_answer,
                entities=qents,
                start=start,
                end=end,
                path=primary_steps,
                kg_chain=kg_chain,
                kg_paragraph=joined_paragraphs,
                path_lists=path_lists_used,
                kg_context=joined_paragraphs,
                retrieved=[],
                answer=ans,
                tokens_in=0,
                tokens_out=0,
                llm_calls=llm_calls,
            )