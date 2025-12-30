from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.modules.kg_schema import KGTriple
from src.modules.llm_client import LLMClient
from src.modules.llm_json import LLMJSON


EntityType = Literal[
    "Person", "Location", "Organization", "Work", "Event", "Date", "Number", "Concept", "Other"
]


class TripleOut(BaseModel):
    subject: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1, description="Prefer snake_case, e.g. born_in, located_in")
    object: str = Field(..., min_length=1)
    subject_type: EntityType = "Other"
    object_type: EntityType = "Other"


class TriplesOut(BaseModel):
    triples: List[TripleOut] = Field(..., min_items=1, max_items=20)


@dataclass
class TripletExtractionResult:
    triples: List[KGTriple]
    raw: str
    parsed: Dict[str, Any]


class KGTripletExtractor:
    def __init__(self, llm_client: LLMClient, max_retries: int = 2):
        self.llm_client = llm_client
        self.max_retries = max(0, int(max_retries))

    def _prompt(self, doc: Document, chunk_id: str, source: str) -> str:
        text = (doc.page_content or "").strip()
        title = str((doc.metadata or {}).get("title", "")).strip()

        return f"""
Du extrahierst faktische Wissens-Tripel aus einem Text.

Gib ausschließlich ein JSON Objekt zurück, ohne zusätzlichen Text.
Schema:
{{
  "triples": [
    {{
      "subject": "...",
      "relation": "...",
      "object": "...",
      "subject_type": "...",
      "object_type": "..."
    }}
  ]
}}

Regeln:
- Extrahiere 3 bis 12 Tripel.
- Relation als kurze, sprechende Phrase, bevorzugt snake_case, zum Beispiel "located_in", "born_in", "is_capital_of".
- Subjekt und Objekt als genaue Entitätsnamen aus dem Text.
- Types als eine der Kategorien: Person, Location, Organization, Work, Event, Date, Number, Concept, Other.
- Keine Halluzinationen, nur was im Text steht.

Kontext:
Titel: {title}
Chunk ID: {chunk_id}
Quelle: {source}

Text:
{text}
""".strip()

    def _to_kg_triples(self, parsed_obj: TriplesOut, doc: Document, chunk_id: str, source: str) -> List[KGTriple]:
        meta = doc.metadata or {}
        out: List[KGTriple] = []
        for t in parsed_obj.triples:
            out.append(
                KGTriple(
                    subject=t.subject.strip(),
                    relation=t.relation.strip(),
                    object=t.object.strip(),
                    subject_type=str(t.subject_type),
                    object_type=str(t.object_type),
                    source=source,
                    chunk_id=chunk_id,
                    meta={"title": meta.get("title", ""), "question_id": meta.get("question_id", "")},
                )
            )
        return out

    def _extract_via_prompt_and_json(self, doc: Document, chunk_id: str, source: str) -> TripletExtractionResult:
        llm = self.llm_client.get_llm()
        last_raw = ""
        last_parsed: Dict[str, Any] = {}

        for _ in range(self.max_retries + 1):
            prompt = self._prompt(doc, chunk_id=chunk_id, source=source)
            resp = llm.invoke(prompt)
            raw = (getattr(resp, "content", str(resp)) or "").strip()
            last_raw = raw

            parsed = LLMJSON.extract_json(raw)
            if not (isinstance(parsed, dict) and isinstance(parsed.get("triples"), list)):
                continue

            triples: List[KGTriple] = []
            meta = doc.metadata or {}
            for t in parsed["triples"]:
                if not isinstance(t, dict):
                    continue
                s = str(t.get("subject", "")).strip()
                r = str(t.get("relation", "")).strip()
                o = str(t.get("object", "")).strip()
                st = str(t.get("subject_type", "Other")).strip() or "Other"
                ot = str(t.get("object_type", "Other")).strip() or "Other"
                if s and r and o:
                    triples.append(
                        KGTriple(
                            subject=s,
                            relation=r,
                            object=o,
                            subject_type=st,
                            object_type=ot,
                            source=source,
                            chunk_id=chunk_id,
                            meta={"title": meta.get("title", ""), "question_id": meta.get("question_id", "")},
                        )
                    )

            if triples:
                return TripletExtractionResult(triples=triples, raw=raw, parsed=parsed)

        return TripletExtractionResult(triples=[], raw=last_raw, parsed=last_parsed)

    def extract(self, doc: Document, chunk_id: str, source: str) -> TripletExtractionResult:
        llm = self.llm_client.get_llm()
        prompt = self._prompt(doc, chunk_id=chunk_id, source=source)

        # 1) Structured output first (with retries)
        for _ in range(self.max_retries + 1):
            try:
                structured = llm.with_structured_output(TriplesOut, include_raw=True)
                out = structured.invoke(prompt)

                # LangChain variants: out may be dict with parsed/raw, or parsed model directly
                parsed_obj = None
                raw_msg = None

                if isinstance(out, dict):
                    parsed_obj = out.get("parsed")
                    raw_msg = out.get("raw")
                elif isinstance(out, TriplesOut):
                    parsed_obj = out
                else:
                    # Try attribute access
                    parsed_obj = getattr(out, "parsed", None)
                    raw_msg = getattr(out, "raw", None)

                if parsed_obj is None:
                    continue

                # If parsed_obj is dict, coerce into TriplesOut
                if isinstance(parsed_obj, dict):
                    parsed_obj = TriplesOut.model_validate(parsed_obj)

                triples = self._to_kg_triples(parsed_obj, doc, chunk_id, source)

                raw_text = ""
                if raw_msg is not None:
                    raw_text = (getattr(raw_msg, "content", str(raw_msg)) or "").strip()

                return TripletExtractionResult(
                    triples=triples,
                    raw=raw_text,
                    parsed=parsed_obj.model_dump(),
                )

            except NotImplementedError:
                break
            except Exception:
                continue

        # 2) Fallback to prompt+JSON parsing
        return self._extract_via_prompt_and_json(doc, chunk_id, source)
