import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_WS = re.compile(r"\s+")
_BAD = re.compile(r"[^A-Z0-9_]+")


def canon_relation(name: str) -> str:
    s = (name or "").strip().upper()
    s = s.replace(" ", "_")
    s = _BAD.sub("", s)
    s = _WS.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def simple_similarity(a: str, b: str) -> float:
    """
    Sehr leichte Ähnlichkeitsfunktion ohne externe Dependencies.
    Reicht aus, um triviale Dubletten wie RELATEDTO vs RELATED_TO zu mergen.
    """
    a = canon_relation(a)
    b = canon_relation(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Token Overlap
    ta = set(a.split("_"))
    tb = set(b.split("_"))
    if not ta or not tb:
        return 0.0
    j = len(ta & tb) / max(1, len(ta | tb))
    # Prefix Bonus
    pref = 0.15 if a.startswith(b) or b.startswith(a) else 0.0
    return min(1.0, j + pref)


@dataclass
class ProposedRelation:
    name: str
    description: str = ""
    subject_labels: Optional[List[str]] = None
    object_labels: Optional[List[str]] = None
    symmetric: bool = False
    evidence_examples: Optional[List[str]] = None


class RelationRegistry:
    """
    Dynamisches Relations-Lexikon.

    - allowed: kanonische Relationsnamen, die dem LLM als Startset gegeben werden
    - aliases: Mapping alias -> canonical
    - pending: neue Vorschläge, die noch nicht final akzeptiert wurden
    """

    def __init__(
        self,
        allowed: Optional[Set[str]] = None,
        aliases: Optional[Dict[str, str]] = None,
        pending: Optional[Dict[str, ProposedRelation]] = None,
    ):
        self.allowed: Set[str] = set(allowed or set())
        self.aliases: Dict[str, str] = dict(aliases or {})
        self.pending: Dict[str, ProposedRelation] = dict(pending or {})

    def resolve(self, name: str) -> str:
        c = canon_relation(name)
        return self.aliases.get(c, c)

    def is_allowed(self, name: str) -> bool:
        c = self.resolve(name)
        return c in self.allowed

    def maybe_merge_as_alias(self, proposed_name: str, threshold: float = 0.88) -> Tuple[bool, str]:
        """
        Versucht eine neue Relation als Alias einer bestehenden allowed Relation zu mergen.
        Gibt (merged, canonical_target) zurück.
        """
        p = canon_relation(proposed_name)
        if not p:
            return False, p
        if p in self.allowed:
            return True, p

        best = ("", 0.0)
        for a in self.allowed:
            sc = simple_similarity(p, a)
            if sc > best[1]:
                best = (a, sc)

        if best[0] and best[1] >= threshold:
            self.aliases[p] = best[0]
            return True, best[0]

        return False, p

    def propose(self, pr: ProposedRelation, alias_threshold: float = 0.88) -> str:
        """
        Nimmt einen Vorschlag entgegen, normalisiert und legt ihn als pending ab,
        falls er nicht gemerged werden kann und noch nicht allowed ist.
        """
        c = canon_relation(pr.name)
        if not c:
            return ""

        pr.name = c

        merged, target = self.maybe_merge_as_alias(c, threshold=alias_threshold)
        if merged:
            return target

        if c in self.allowed:
            return c

        if c not in self.pending:
            self.pending[c] = pr
        else:
            # leichte Anreicherung
            existing = self.pending[c]
            if pr.description and not existing.description:
                existing.description = pr.description
            if pr.subject_labels and not existing.subject_labels:
                existing.subject_labels = pr.subject_labels
            if pr.object_labels and not existing.object_labels:
                existing.object_labels = pr.object_labels
            if pr.evidence_examples:
                ex = list(existing.evidence_examples or [])
                for e in pr.evidence_examples:
                    if e and e not in ex:
                        ex.append(e)
                existing.evidence_examples = ex[:10]

        return c

    def accept(self, name: str, aliases: Optional[List[str]] = None) -> str:
        """
        Akzeptiert eine Relation in allowed und optional ihre Aliase.
        """
        c = canon_relation(name)
        if not c:
            return ""
        self.allowed.add(c)
        if aliases:
            for a in aliases:
                aa = canon_relation(a)
                if aa:
                    self.aliases[aa] = c
        self.pending.pop(c, None)
        return c

    def reject(self, name: str) -> None:
        c = canon_relation(name)
        self.pending.pop(c, None)

    def snapshot_allowed_sorted(self) -> List[str]:
        return sorted(self.allowed)

    def save(self, path: Path) -> None:
        rec = {
            "allowed": sorted(self.allowed),
            "aliases": dict(self.aliases),
            "pending": {k: asdict(v) for k, v in self.pending.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RelationRegistry":
        if not path.exists():
            return cls()
        rec = json.loads(path.read_text(encoding="utf-8"))
        allowed = set(rec.get("allowed") or [])
        aliases = dict(rec.get("aliases") or {})
        pending_raw = rec.get("pending") or {}
        pending: Dict[str, ProposedRelation] = {}
        for k, v in pending_raw.items():
            try:
                pending[k] = ProposedRelation(**(v or {}))
            except Exception:
                pending[k] = ProposedRelation(name=k, description=str(v or ""))
        return cls(allowed=allowed, aliases=aliases, pending=pending)
