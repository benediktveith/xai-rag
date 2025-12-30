from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from typing import List, Optional, Tuple

import networkx as nx

from src.modules.kg_store import KGStore
from src.modules.llm_client import LLMClient
from src.modules.llm_json import LLMJSON


@dataclass(frozen=True)
class QueryEntities:
    entities: List[str]


class KGQueryService:
    def __init__(self, kg: KGStore, llm_client: Optional[LLMClient] = None):
        self.kg = kg
        self.llm_client = llm_client

    def extract_entities(self, question: str, k: int = 6) -> QueryEntities:
        if self.llm_client:
            llm = self.llm_client.get_llm()
            prompt = f"""
Gib ausschließlich JSON zurück, ohne zusätzlichen Text.
Schema: {{ "entities": ["...", "..."] }}

Aufgabe:
Extrahiere 2 bis {k} zentrale Entitäten aus der Frage, als exakt vorkommende Names, keine Erklärungen.

Frage:
{question}
""".strip()
            resp = llm.invoke(prompt)
            raw = (getattr(resp, "content", str(resp)) or "").strip()
            parsed = LLMJSON.extract_json(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("entities"), list):
                ents = [str(x).strip() for x in parsed["entities"] if str(x).strip()]
                return QueryEntities(entities=ents[:k])

        q = (question or "").lower()
        hits = [n for n in self.kg.g.nodes if n.lower() in q]
        if len(hits) >= 2:
            return QueryEntities(entities=hits[:k])

        tokens = set(q.replace("?", " ").replace(",", " ").split())
        ranked = sorted(
            list(self.kg.g.nodes),
            key=lambda n: len(tokens & set(n.lower().split())),
            reverse=True,
        )
        return QueryEntities(entities=ranked[:k])

    def pick_pair_with_path(self, entities: List[str]) -> Optional[Tuple[str, str]]:
        if len(entities) < 2:
            return None

        best_pair: Optional[Tuple[str, str]] = None
        best_len: Optional[int] = None

        for a, b in permutations(entities, 2):
            if a not in self.kg.g or b not in self.kg.g:
                continue

            try:
                path_nodes = nx.shortest_path(self.kg.g, a, b)
                plen = len(path_nodes)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            except Exception:
                continue

            if best_len is None or plen < best_len:
                best_len = plen
                best_pair = (a, b)

        if best_pair is None:
            existing = [e for e in entities if e in self.kg.g]
            if len(existing) >= 2:
                return (existing[0], existing[1])
            return (entities[0], entities[1])

        return best_pair
