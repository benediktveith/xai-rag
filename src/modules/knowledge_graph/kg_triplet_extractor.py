from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import re
from pydantic import BaseModel, Field, ValidationError
import json

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException

from src.modules.knowledge_graph.kg_schema import KGTriple
from src.modules.llm.llm_client import LLMClient
from src.modules.knowledge_graph.relation_registry import RelationRegistry, ProposedRelation, canon_relation

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

# Canonical entity types used for KGTriple.subject_type / object_type.
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

# Allowed UI-facing labels returned by the extractor and accepted for typing.
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

# Mapping from extractor labels to internal canonical entity types.
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

_WS = re.compile(r"\s+")
_HAS_SENTENCE_PUNCT = re.compile(r"[.!?]")
_MULTI_PUNCT = re.compile(r"[;,]{1,}")

# Single-letter options and common MCQ artifacts that must not leak into entities.
_BAD_SINGLE = {"A", "B", "C", "D"}

_MAX_ENTITY_TOKENS = 6
_MAX_ENTITY_CHARS = 100


# Relations, die du initial als Startset geben willst, du kannst das weiterhin zentral pflegen.
_DEFAULT_ALLOWED_RELATIONS: Set[str] = {
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

# Optional, einfache Typprüfung, kann später erweitert werden.
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


def _extract_first_json_object(text: str) -> Optional[dict]:
    # Best-effort extraction of the first JSON object from arbitrary text (e.g., LLM parser errors).
    if not text:
        return None
    m = _JSON_OBJ.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _coerce_list_of_dicts(x: Any) -> List[dict]:
    # Coerce unknown values into a list of dicts, filtering out non-dicts.
    if not isinstance(x, list):
        return []
    return [t for t in x if isinstance(t, dict)]

def _filter_valid_triples_dicts(triples: List[dict]) -> List[dict]:
    # Filter to only those dicts that contain all required string fields for a triple.
    req = ["subject", "subject_label", "relation", "object", "object_label"]
    out: List[dict] = []
    for t in triples:
        ok = True
        for k in req:
            v = t.get(k)
            if not isinstance(v, str) or not v.strip():
                ok = False
                break
        if ok:
            out.append(t)
    return out

def _norm_text(s: str) -> str:
    # Whitespace normalization used across entity cleanup and metadata normalization.
    s = (s or "").strip()
    s = s.replace("\u00a0", " ")
    s = _WS.sub(" ", s)
    return s


def _token_count(s: str) -> int:
    # Token counter used to enforce compact entity phrases.
    s = _norm_text(s)
    if not s:
        return 0
    return len([t for t in s.split(" ") if t])


def _looks_like_sentence_fragment(s: str) -> bool:
    # Heuristics to reject entities that look like sentences or overly long fragments.
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
    return False


def _is_bad_entity(ent: str) -> bool:
    # Reject MCQ artifacts, booleans, pure numerics, and sentence-like fragments.
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
    # Normalize and strip leading MCQ markers like "A:" or "B)".
    e = _norm_text(ent)
    e = re.sub(r"^[A-D]\s*:\s*", "", e)
    e = re.sub(r"^[A-D]\)\s*", "", e)
    e = _norm_text(e)
    e = e.strip(" ,;:")
    return e


def _strip_mc_options(text: str) -> str:
    # Remove multiple choice option lines (A:, B), etc.) before extraction.
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
    # Extract a short snippet around both entity mentions to store as provenance evidence.
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


def _validate_or_swap_by_labels(
    e1: str,
    l1: str,
    rel_up: str,
    e2: str,
    l2: str,
) -> Optional[Tuple[str, str, str, str]]:
    # Validate subject/object label constraints for a relation, optionally swapping if allowed.
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


class ProposedRelationOut(BaseModel):
    # LLM-facing schema for proposing a new relation when no allowed relation fits.
    name: str = Field(..., description="Neue Relation in UPPER_SNAKE_CASE, z.B. ASSOCIATED_WITH")
    description: str = Field("", description="Kurzbeschreibung der Semantik")
    subject_labels: Optional[List[str]] = Field(default=None, description="Optional, z.B. ['Diseases']")
    object_labels: Optional[List[str]] = Field(default=None, description="Optional, z.B. ['Symptoms']")
    symmetric: Optional[bool] = Field(default=False, description="True wenn symmetrisch")


class ExtractedTripleOut(BaseModel):
    # LLM-facing schema for an extracted triple with label-typed endpoints.
    subject: str
    subject_label: str
    relation: str
    object: str
    object_label: str


class KGExtractionOutput(BaseModel):
    # Structured extraction result: triples plus any newly proposed relations.
    triples: List[ExtractedTripleOut] = Field(default_factory=list)
    new_relations: List[ProposedRelationOut] = Field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class TripletExtractionResult:
    # Concrete extraction result returned by KGTripletExtractor, including parsed payload and raw error context if any.
    triples: List[KGTriple]
    raw: str
    parsed: Dict[str, Any]


class KGTripletExtractor:
    def __init__(
        self,
        llm_client: LLMClient,
        relation_registry: RelationRegistry,
        max_retries: int = 1,
        auto_accept_new_relations: bool = True,
        alias_threshold: float = 0.88,
    ):
        self.llm_client = llm_client
        self.registry = relation_registry
        self.max_retries = max(0, int(max_retries))
        self.auto_accept_new_relations = bool(auto_accept_new_relations)
        self.alias_threshold = float(alias_threshold)

        # Seed default relations when the registry is empty to provide the LLM a stable baseline.
        if not self.registry.allowed:
            for r in _DEFAULT_ALLOWED_RELATIONS:
                self.registry.accept(r)

    def _prompt(self, doc: Document, chunk_id: str, source: str) -> str:
        # Build a strict JSON-schema grounded prompt, including allowed labels and current allowed relations.
        raw = (doc.page_content or "").strip()
        meta = doc.metadata or {}
        title = str(meta.get("title", meta.get("topic_name", ""))).strip()
        source_filename = str(meta.get("source_filename", ""))

        text = _strip_mc_options(raw)

        labels = "\n".join([f"- {x}" for x in sorted(_ALLOWED_LABELS)])
        rels = "\n".join([f"- {x}" for x in self.registry.snapshot_allowed_sorted()])

        return f"""
                You are an expert biomedical information extractor.

                Extract knowledge graph triples from the medical text.

                Valid Entity Types (Labels):
                {labels}

                Allowed Relationship Types (Preferred):
                {rels}

                Rules:
                1. Extract triples only if both entities are explicitly mentioned in the text.
                2. Prefer using an existing relation from the Allowed Relationship Types list.
                3. If none of the allowed relations fits semantically, propose a new relation.
                - New relation must be UPPER_SNAKE_CASE, short, unambiguous.
                4. Avoid duplicates and redundant triples.
                5. Entities must be short noun phrases, not full sentences.
                - Max {_MAX_ENTITY_TOKENS} words.
                - Do not include sentence punctuation inside entities.
                6. Do not infer facts not stated in the text.
                
                Important:
                - Only output complete triple objects. Do not output partial objects.
                - If you cannot fill in a triple completely, omit it entirely.

                Return strictly structured JSON according to the following schema:
                {json.dumps(KGExtractionOutput.model_json_schema(), ensure_ascii=False, indent=2)}

                Context:
                Title: {title}
                Source: {source}
                Source filename: {source_filename}
                Chunk ID: {chunk_id}

                Text:
                {text}
                """

    def _finalize(
        self,
        out: KGExtractionOutput,
        doc: Document,
        chunk_id: str,
        source: str,
    ) -> List[KGTriple]:
        # Post-process structured output: register new relations, validate triples, and attach provenance metadata.
        meta0 = doc.metadata or {}
        text = (doc.page_content or "").strip()

        for pr in (out.new_relations or []):
            proposed = ProposedRelation(
                name=pr.name,
                description=_norm_text(pr.description),
                subject_labels=pr.subject_labels,
                object_labels=pr.object_labels,
                symmetric=bool(pr.symmetric or False),
            )
            cname = self.registry.propose(proposed, alias_threshold=self.alias_threshold)
            if not cname:
                continue
            if self.auto_accept_new_relations:
                aliases = []
                raw_name = canon_relation(pr.name)
                if raw_name and raw_name != cname:
                    aliases.append(raw_name)
                self.registry.accept(cname, aliases=aliases)

        seen: Set[Tuple[str, str, str]] = set()
        triples: List[KGTriple] = []

        for t in out.triples or []:
            e1c = _clean_entity(t.subject)
            e2c = _clean_entity(t.object)
            l1c = _norm_text(t.subject_label)
            l2c = _norm_text(t.object_label)

            # Enforce allowed labels and entity hygiene to avoid polluting the KG.
            if l1c not in _ALLOWED_LABELS or l2c not in _ALLOWED_LABELS:
                continue
            if _is_bad_entity(e1c) or _is_bad_entity(e2c):
                continue
            if e1c.lower() == e2c.lower():
                continue

            # Resolve relation through the registry and store canonical UPPER_SNAKE_CASE.
            rel_canon = self.registry.resolve(t.relation)
            rel_up = canon_relation(rel_canon)
            if not rel_up:
                continue

            checked = _validate_or_swap_by_labels(e1c, l1c, rel_up, e2c, l2c)
            if checked is None:
                continue
            e1f, l1f, e2f, l2f = checked

            key = (e1f.lower(), rel_up, e2f.lower())
            if key in seen:
                continue
            seen.add(key)

            ev = _extract_evidence_span(text, e1f, e2f, max_words=25)

            triples.append(
                KGTriple(
                    subject=e1f,
                    relation=rel_up,
                    object=e2f,
                    subject_type=_LABEL_TO_TYPE.get(l1f, "Other"),
                    object_type=_LABEL_TO_TYPE.get(l2f, "Other"),
                    source=source,
                    chunk_id=chunk_id,
                    meta={
                        "title": meta0.get("title", meta0.get("topic_name", "")),
                        "question_id": meta0.get("question_id", ""),
                        "source_filename": meta0.get("source_filename", ""),
                        "chunk_index": meta0.get("chunk_index", None),
                        "evidence": ev,
                        "label_subject": l1f,
                        "label_object": l2f,
                        "relation_raw": _norm_text(t.relation),
                        "relation_canonical": rel_up,
                        "relation_allowed_at_time": self.registry.is_allowed(rel_up),
                    },
                )
            )

        return triples

    def _extract_structured(self, doc: Document, chunk_id: str, source: str) -> TripletExtractionResult:
        # Invoke the LLM with structured output, retrying and attempting partial recovery on schema errors.
        llm = self.llm_client.get_llm()
        if llm is None:
            raise RuntimeError("LLMClient.get_llm() returned None")

        structured_llm = llm.with_structured_output(KGExtractionOutput)

        last_raw = ""
        last_parsed: Dict[str, Any] = {}

        for _ in range(self.max_retries + 1):
            prompt = self._prompt(doc, chunk_id=chunk_id, source=source)

            try:
                resp = structured_llm.invoke(prompt)
                print(resp)

                out: KGExtractionOutput
                if isinstance(resp, KGExtractionOutput):
                    out = resp
                else:
                    out = KGExtractionOutput.model_validate(resp)

            except (OutputParserException, ValidationError) as e:
                last_raw = str(e)

                # Attempt to recover by extracting a JSON object embedded in the error string.
                raw_obj = _extract_first_json_object(last_raw)
                if not isinstance(raw_obj, dict):
                    continue

                triples_dicts = _coerce_list_of_dicts(raw_obj.get("triples"))
                triples_dicts = _filter_valid_triples_dicts(triples_dicts)

                newrels_dicts = _coerce_list_of_dicts(raw_obj.get("new_relations"))

                if not triples_dicts and not newrels_dicts:
                    continue

                try:
                    out = KGExtractionOutput.model_validate({"triples": triples_dicts, "new_relations": newrels_dicts, "notes": raw_obj.get("notes")})
                except Exception:
                    continue

            except Exception as e:
                last_raw = str(e)
                continue

            last_parsed = out.model_dump()
            triples = self._finalize(out, doc=doc, chunk_id=chunk_id, source=source)
            if triples:
                return TripletExtractionResult(triples=triples, raw=last_raw, parsed=last_parsed)

        return TripletExtractionResult(triples=[], raw=last_raw, parsed=last_parsed)

    def extract(self, doc: Document, chunk_id: str, source: str) -> TripletExtractionResult:
        # Public entrypoint, currently delegating to the structured extraction implementation.
        return self._extract_structured(doc, chunk_id, source)
