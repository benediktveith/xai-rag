from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Set, Tuple, Optional

import re
from langchain_core.documents import Document

from src.modules.knowledge_graph.kg_schema import KGTriple
from src.modules.llm.llm_client import LLMClient


EntityType = Literal[
    "Disease",
    "RiskFactor",
    "Symptom",
    "DiagnosticTest",
    "Treatment",
    "BodyPart",
    "Medication",
    "Procedure",
    "Condition",
    "Concept",
    "Other",
]

_ALLOWED_LABELS: Set[str] = {
    "Diseases",
    "Medications",
    "Symptoms",
    "Treatments",
    "Diagnostic Tests",
    "Risk Factors",
    "Body Parts",
    "Procedures",
}

_ALLOWED_RELATIONS: Set[str] = {
    "HAS_SYMPTOM",
    "AFFECTS",
    "DIAGNOSED_BY",
    "TREATED_WITH",
    "MANAGED_BY",
    "HAS_RISK_FACTOR",
    "COMORBID_WITH",
    "CAUSED_BY",
    "INDICATES",
    "OCCURS_IN",
    "USED_FOR",
    "INVOLVES_MEDICATION",
    "HAS_SIDE_EFFECT",
    "CONTRAINDICATED_FOR",
    "DETECTS",
    "MEASURES",
    "INCREASES_RISK_OF",
    "PART_OF",
    "CAN_AFFECT",
}

_LABEL_TO_TYPE: Dict[str, EntityType] = {
    "Diseases": "Disease",
    "Medications": "Medication",
    "Symptoms": "Symptom",
    "Treatments": "Treatment",
    "Diagnostic Tests": "DiagnosticTest",
    "Risk Factors": "RiskFactor",
    "Body Parts": "BodyPart",
    "Procedures": "Procedure",
}

_REL_LABEL_RULES: Dict[str, Dict[str, Any]] = {
    "HAS_SYMPTOM": {"sub": {"Diseases"}, "obj": {"Symptoms"}, "swappable": True},
    "HAS_RISK_FACTOR": {"sub": {"Diseases"}, "obj": {"Risk Factors"}, "swappable": True},
    "OCCURS_IN": {"sub": {"Diseases", "Symptoms"}, "obj": {"Body Parts"}, "swappable": True},
    "USED_FOR": {"sub": {"Medications", "Treatments", "Procedures"}, "obj": {"Diseases"}, "swappable": True},
    "DIAGNOSED_BY": {"sub": {"Diseases"}, "obj": {"Diagnostic Tests"}, "swappable": True},
    "TREATED_WITH": {"sub": {"Diseases"}, "obj": {"Treatments", "Medications"}, "swappable": True},
    "MANAGED_BY": {"sub": {"Diseases"}, "obj": {"Treatments", "Medications"}, "swappable": True},
    "INCREASES_RISK_OF": {"sub": {"Diseases", "Risk Factors"}, "obj": {"Diseases"}, "swappable": False},
    "AFFECTS": {"sub": {"Diseases", "Body Parts", "Risk Factors"}, "obj": {"Diseases", "Body Parts"}, "swappable": False},
    "CAN_AFFECT": {"sub": {"Diseases", "Risk Factors", "Body Parts"}, "obj": {"Diseases", "Body Parts"}, "swappable": False},
    "COMORBID_WITH": {"sub": {"Diseases"}, "obj": {"Diseases"}, "swappable": False},
    "PART_OF": {"sub": {"Body Parts"}, "obj": {"Body Parts"}, "swappable": False},
}

_BAD_SINGLE = {"A", "B", "C", "D"}
_WS = re.compile(r"\s+")
_PUNCT_REL = re.compile(r"[^a-z0-9\s_]")
_ANSWER_LINE = re.compile(r"^\s*Answer\s*:\s*(.+?)\s*$", re.IGNORECASE)
_SPLIT_COLON = re.compile(r"\s*:\s*")

_MAX_ENTITY_TOKENS = 10
_MAX_ENTITY_CHARS = 80
_HAS_SENTENCE_PUNCT = re.compile(r"[.!?]")
_MULTI_PUNCT = re.compile(r"[;,]{1,}")

_BAD_ENTITY_PATTERNS = [
    r"\bpatients?\b",
    r"\bwho\b",
    r"\bwhich\b",
    r"\bthat\b",
    r"\bundergoing\b",
    r"\bas part of\b",
    r"\bevaluation of\b",
    r"\bin order to\b",
    r"\bresult(s)? in\b",
    r"\blead(s)? to\b",
]


@dataclass
class TripletExtractionResult:
    triples: List[KGTriple]
    raw: str
    parsed: Dict[str, Any]


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u00a0", " ")
    s = _WS.sub(" ", s)
    return s


def _token_count(s: str) -> int:
    s = _norm_text(s)
    if not s:
        return 0
    return len([t for t in s.split(" ") if t])


def _looks_like_sentence_fragment(s: str) -> bool:
    t = _norm_text(s)
    if not t:
        return True

    if len(t) > _MAX_ENTITY_CHARS:
        return True

    if _token_count(t) > _MAX_ENTITY_TOKENS:
        return True

    if _HAS_SENTENCE_PUNCT.search(t):
        return True

    if _MULTI_PUNCT.search(t):
        return True

    low = t.lower()
    for p in _BAD_ENTITY_PATTERNS:
        if re.search(p, low):
            return True

    return False


def _is_bad_entity(ent: str) -> bool:
    e = _norm_text(ent)
    if not e:
        return True
    if len(e) == 1 and e.upper() in _BAD_SINGLE:
        return True
    if e.lower() in {"true", "false", "none", "unknown"}:
        return True
    if re.fullmatch(r"[0-9]+", e):
        return True
    if re.fullmatch(r"[A-D]\)", e):
        return True
    if _looks_like_sentence_fragment(e):
        return True
    return False


def _clean_entity(ent: str) -> str:
    e = _norm_text(ent)
    e = re.sub(r"^[A-D]\s*:\s*", "", e)
    e = re.sub(r"^[A-D]\)\s*", "", e)
    e = _norm_text(e)
    e = e.strip(" ,;:")
    return e


def _rel_to_snake(rel: str) -> str:
    r = _norm_text(rel).upper()
    r = r.replace(" ", "_")
    return r.lower()


def _strip_mc_options(text: str) -> str:
    lines = (text or "").splitlines()
    kept: List[str] = []
    for ln in lines:
        l = ln.strip()
        if not l:
            continue
        if re.match(r"^[A-D]\s*[:\)]\s*", l):
            continue
        kept.append(l)
    return "\n".join(kept)


def _extract_evidence_span(text: str, e1: str, e2: str, max_words: int = 25) -> str:
    t = _norm_text(text)
    if not t:
        return ""

    low = t.lower()
    a = _norm_text(e1).lower()
    b = _norm_text(e2).lower()
    if not a or not b:
        return ""

    ia = low.find(a)
    ib = low.find(b)
    if ia < 0 or ib < 0:
        return ""

    start = min(ia, ib)
    end = max(ia + len(a), ib + len(b))

    words = t.split()
    offsets: List[Tuple[int, int]] = []
    cur = 0
    for w in words:
        w_start = t.find(w, cur)
        w_end = w_start + len(w)
        offsets.append((w_start, w_end))
        cur = w_end

    def char_to_word_index(pos: int) -> int:
        for i, (s, e) in enumerate(offsets):
            if s <= pos <= e:
                return i
        return 0

    wi0 = char_to_word_index(max(0, start))
    wi1 = char_to_word_index(min(len(t) - 1, end))

    left = max(0, wi0 - 8)
    right = min(len(words), wi1 + 9)

    snippet_words = words[left:right]
    if len(snippet_words) > max_words:
        snippet_words = snippet_words[:max_words]
    return " ".join(snippet_words).strip()


def _clean_relation(rel: str) -> str:
    r = _norm_text(rel).lower()
    r = _PUNCT_REL.sub(" ", r)
    r = _WS.sub(" ", r).strip()
    r = r.replace(" ", "_")
    return r


def _validate_or_swap_by_labels(
    e1: str,
    l1: str,
    rel_up: str,
    e2: str,
    l2: str,
) -> Optional[Tuple[str, str, str, str]]:
    rule = _REL_LABEL_RULES.get(rel_up)
    if rule is None:
        if l1 in _ALLOWED_LABELS and l2 in _ALLOWED_LABELS:
            return e1, l1, e2, l2
        return None

    sub_ok = l1 in rule["sub"] and l2 in rule["obj"]
    if sub_ok:
        return e1, l1, e2, l2

    if bool(rule.get("swappable")):
        swap_ok = l2 in rule["sub"] and l1 in rule["obj"]
        if swap_ok:
            return e2, l2, e1, l1

    return None


class KGTripletExtractor:
    """
    Repo style extraction.
    """

    def __init__(self, llm_client: LLMClient, max_retries: int = 2):
        self.llm_client = llm_client
        self.max_retries = max(0, int(max_retries))

    def _prompt(self, doc: Document, chunk_id: str, source: str) -> str:
        raw = (doc.page_content or "").strip()
        meta = doc.metadata or {}
        title = str(meta.get("title", meta.get("topic_name", ""))).strip()
        source_filename = str(meta.get("source_filename", ""))

        text = _strip_mc_options(raw)

        labels = "\n".join([f"- {x}" for x in sorted(_ALLOWED_LABELS)])
        rels = "\n".join([f"- {x}" for x in sorted(_ALLOWED_RELATIONS)])

        return f"""
You are an expert biomedical information extractor.

Your task is to extract all valid Entity1, Relationship, Entity2 triples from the medical text below.
Each triple must follow strict semantic and formatting rules.

Valid Entity Types (Labels):
{labels}

Valid Relationship Types:
{rels}

Instructions:
1. Only extract triples where:
   - Both Entity1 and Entity2 are explicitly mentioned in the input text.
   - Both entities have a valid label from the list above.
   - The relationship is one of the predefined types and is semantically correct.
2. Only extract forward relationships.
   - Do not include reverse or inverse relationships.
3. Do not infer information not explicitly stated.
4. Avoid duplicates and redundant triples.
5. Entities must be short noun phrases, not full sentences or clauses.
   - Max 10 words and max 80 characters.
   - Do not include punctuation like '.', ';', '?' inside entities.
6. If no valid triples exist, return nothing.

Output format:
- One triple per line.
- Must begin with: Answer:
- Format exactly like: Answer: Entity1 : Label1 : Relationship : Entity2 : Label2
- No numbering and no extra text.

Context:
Title: {title}
Source: {source}
Source filename: {source_filename}
Chunk ID: {chunk_id}

Text:
{text}
""".strip()

    def _parse_lines(self, raw: str) -> List[Tuple[str, str, str, str, str]]:
        out: List[Tuple[str, str, str, str, str]] = []
        if not raw:
            return out

        for line in raw.splitlines():
            m = _ANSWER_LINE.match(line)
            if not m:
                continue

            payload = m.group(1).strip()
            parts = _SPLIT_COLON.split(payload)
            if len(parts) != 5:
                continue

            e1, l1, rel, e2, l2 = [p.strip() for p in parts]
            if not e1 or not l1 or not rel or not e2 or not l2:
                continue

            out.append((e1, l1, rel, e2, l2))

        return out

    def _finalize(
        self,
        tuples: List[Tuple[str, str, str, str, str]],
        doc: Document,
        chunk_id: str,
        source: str,
    ) -> List[KGTriple]:
        meta = doc.metadata or {}
        text = (doc.page_content or "").strip()

        seen: Set[Tuple[str, str, str]] = set()
        triples: List[KGTriple] = []

        for e1, l1, rel, e2, l2 in tuples:
            e1c = _clean_entity(e1)
            e2c = _clean_entity(e2)
            l1c = _norm_text(l1)
            l2c = _norm_text(l2)
            rel_up = _norm_text(rel).upper()

            if rel_up not in _ALLOWED_RELATIONS:
                continue

            if l1c not in _ALLOWED_LABELS or l2c not in _ALLOWED_LABELS:
                continue

            if _is_bad_entity(e1c) or _is_bad_entity(e2c):
                continue
            if e1c.lower() == e2c.lower():
                continue

            checked = _validate_or_swap_by_labels(e1c, l1c, rel_up, e2c, l2c)
            if checked is None:
                continue
            e1f, l1f, e2f, l2f = checked

            rel_snake = _clean_relation(_rel_to_snake(rel_up))
            if not rel_snake:
                continue

            key = (e1f.lower(), rel_snake, e2f.lower())
            if key in seen:
                continue
            seen.add(key)

            ev = _extract_evidence_span(text, e1f, e2f, max_words=25)

            triples.append(
                KGTriple(
                    subject=e1f,
                    relation=rel_snake,
                    object=e2f,
                    subject_type=_LABEL_TO_TYPE.get(l1f, "Other"),
                    object_type=_LABEL_TO_TYPE.get(l2f, "Other"),
                    source=source,
                    chunk_id=chunk_id,
                    meta={
                        "title": meta.get("title", meta.get("topic_name", "")),
                        "question_id": meta.get("question_id", ""),
                        "source_filename": meta.get("source_filename", ""),
                        "chunk_index": meta.get("chunk_index", None),
                        "evidence": ev,
                        "label_subject": l1f,
                        "label_object": l2f,
                        "relation_raw": rel_up,
                    },
                )
            )

        return triples

    def _extract_via_prompt(self, doc: Document, chunk_id: str, source: str) -> TripletExtractionResult:
        llm = self.llm_client.get_llm()
        if llm is None:
            raise RuntimeError("LLMClient.get_llm() returned None")

        last_raw = ""
        last_parsed: Dict[str, Any] = {}

        for _ in range(self.max_retries + 1):
            prompt = self._prompt(doc, chunk_id=chunk_id, source=source)
            resp = llm.invoke(prompt)
            raw = (getattr(resp, "content", str(resp)) or "").strip()
            last_raw = raw

            tuples = self._parse_lines(raw)
            triples = self._finalize(tuples, doc=doc, chunk_id=chunk_id, source=source)

            if triples:
                return TripletExtractionResult(triples=triples, raw=raw, parsed=last_parsed)

        return TripletExtractionResult(triples=[], raw=last_raw, parsed=last_parsed)

    def extract(self, doc: Document, chunk_id: str, source: str) -> TripletExtractionResult:
        return self._extract_via_prompt(doc, chunk_id, source)
