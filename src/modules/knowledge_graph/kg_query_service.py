from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, combinations
from typing import List, Optional, Tuple

import re
import difflib
import networkx as nx

from src.modules.knowledge_graph.kg_store import KGStore
from src.modules.llm.llm_client import LLMClient
from src.modules.llm.llm_json import LLMJSON


@dataclass(frozen=True)
class QueryEntities:
    # Output container for extracted entities and the entity pairs to be considered for path search.
    entities: List[str]
    pairs: List[Tuple[str, str]]


# Precompiled regex helpers for canonicalization and tokenization.
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9\s]")


def _canon_text(s: str) -> str:
    # Normalize text for matching: lowercase, strip, remove punctuation, and collapse whitespace.
    s = (s or "").strip().lower()
    s = s.replace("\u00a0", " ")
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _token_set(s: str) -> set[str]:
    # Tokenize normalized text into a set for overlap-based similarity scoring.
    s = _canon_text(s)
    return set([t for t in s.split() if t])


def _pair_key(a: str, b: str) -> tuple[str, str]:
    # Order-independent key for de-duplicating entity pairs.
    ca, cb = _canon_text(a), _canon_text(b)
    return (ca, cb) if ca <= cb else (cb, ca)

class KGQueryService:
    def __init__(self, kg: KGStore, llm_client: Optional[LLMClient] = None):
        self.kg = kg
        self.llm_client = llm_client

        # Snapshot node names and derived lookup structures for fast mapping.
        self._nodes: List[str] = [str(n) for n in self.kg.g.nodes]

        self._node_names: dict[str, str] = {n: n for n in self._nodes}
        self._node_tokens: dict[str, set[str]] = {n: _token_set(self._node_names[n]) for n in self._nodes}
        self._node_canon_name: dict[str, str] = {n: _canon_text(self._node_names[n]) for n in self._nodes}

        # Exact-match index by canonical node name.
        self._node_by_canon: dict[str, str] = {}
        for n in self._nodes:
            c = self._node_canon_name[n]
            if c and c not in self._node_by_canon:
                self._node_by_canon[c] = n


    def extract_entities(self, question: str) -> QueryEntities:
        # Prefer the LLM-based extractor when available, and enforce a JSON-only contract.
        if self.llm_client:
            llm = self.llm_client.get_llm()
            if llm is not None:
                prompt = f"""
                Return ONLY JSON, no extra text.

                Schema:
                {{
                "entities": ["...", "..."],
                "pairs": [["...", "..."], ["...", "..."]]
                }}

                Task:
                Extract central medical entities from the question.
                Entities must be short noun phrases.
                Do not output answer options or single letters.

                Additionally:
                You MAY output pairs, but the system will consider all pairwise combinations of the extracted entities anyway.

                Question:
                {question}
                """.strip()

                resp = llm.invoke(prompt)
                raw = (getattr(resp, "content", str(resp)) or "").strip()
                parsed = LLMJSON.extract_json(raw)

                ents_out: List[str] = []
                pairs_out: List[Tuple[str, str]] = []

                # Parse and sanitize LLM output, filtering out MCQ option letters.
                if isinstance(parsed, dict):
                    ents = parsed.get("entities")
                    if isinstance(ents, list):
                        for x in ents:
                            s = str(x).strip()
                            if len(s) == 1 and s.upper() in {"A", "B", "C", "D"}:
                                continue
                            if s and s not in ents_out:
                                ents_out.append(s)

                    prs = parsed.get("pairs")
                    if isinstance(prs, list):
                        for p in prs:
                            if not isinstance(p, (list, tuple)) or len(p) != 2:
                                continue
                            a = str(p[0]).strip()
                            b = str(p[1]).strip()
                            if not a or not b or a == b:
                                continue
                            if len(a) == 1 and a.upper() in {"A", "B", "C", "D"}:
                                continue
                            if len(b) == 1 and b.upper() in {"A", "B", "C", "D"}:
                                continue
                            pairs_out.append((a, b))

                for a, b in pairs_out:
                    if a and a not in ents_out:
                        ents_out.append(a)
                    if b and b not in ents_out:
                        ents_out.append(b)

                all_pairs = list(combinations(ents_out, 2))

                seen = set()
                pairs_final: List[Tuple[str, str]] = []

                for a, b in pairs_out + all_pairs:
                    key = _pair_key(a, b)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs_final.append((a, b))

                return QueryEntities(entities=ents_out, pairs=pairs_final)

        q = _canon_text(question)
        hits = [self._node_names[n] for n in self._nodes if self._node_canon_name[n] and self._node_canon_name[n] in q]

        # Fallback: token overlap scoring against KG node names when substring hits are insufficient.
        if len(hits) < 2:
            toks = _token_set(question)
            scored: List[Tuple[int, str]] = []
            for n in self._nodes:
                inter = len(toks & (self._node_tokens.get(n) or set()))
                if inter > 0:
                    scored.append((inter, self._node_names[n]))
            scored.sort(key=lambda x: x[0], reverse=True)
            hits = [x[1] for x in scored[:8]]

        pairs = [(a, b) for a, b in combinations(hits, 2)]
        return QueryEntities(entities=hits, pairs=pairs)

    def map_to_node(self, ent: str) -> Optional[str]:
        # Map a free-text entity mention to a KG node via exact canonical match, then fuzzy similarity.
        if not ent:
            return None

        ent_canon = _canon_text(ent)
        if ent_canon in self._node_by_canon:
            return self._node_by_canon[ent_canon]

        ent_toks = _token_set(ent)
        if not ent_toks:
            return None

        best: Optional[str] = None
        best_score = 0.0

        for n in self._nodes:
            nt = self._node_tokens.get(n) or set()
            if not nt:
                continue

            inter = len(ent_toks & nt)
            if inter == 0:
                continue

            overlap = inter / float(len(ent_toks))
            union = len(ent_toks | nt)
            jaccard = (inter / float(union)) if union else 0.0
            seq = difflib.SequenceMatcher(None, ent_canon, self._node_canon_name.get(n, "")).ratio()
            score = max(overlap, jaccard, seq)

            if score > best_score:
                best_score = score
                best = n

        if best_score >= 0.62:
            return best
        return None

    def pick_pair_with_path(self, entities: List[str]) -> Optional[Tuple[str, str]]:
        # Choose the pair of mapped entities with the shortest undirected path, preferring connectivity.
        mapped: List[str] = []
        for e in entities:
            m = self.map_to_node(e)
            if m and m not in mapped:
                mapped.append(m)

        if len(mapped) < 2:
            return None

        best_pair: Optional[Tuple[str, str]] = None
        best_len: Optional[int] = None

        G_und = self.kg.g.to_undirected(as_view=True)

        for a, b in permutations(mapped, 2):
            try:
                path_nodes = nx.shortest_path(G_und, a, b)
                plen = len(path_nodes)
            except Exception:
                continue

            if best_len is None or plen < best_len:
                best_len = plen
                best_pair = (a, b)

        if best_pair is None:
            return (mapped[0], mapped[1])

        return best_pair
