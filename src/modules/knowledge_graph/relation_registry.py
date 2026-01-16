import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_WS = re.compile(r"\s+")
_BAD = re.compile(r"[^A-Z0-9_]+")


def canon_relation(name: str) -> str:
    # Canonicalize relation names to UPPER_SNAKE_CASE, stripping unsupported characters.
    s = (name or "").strip().upper()
    s = s.replace(" ", "_")
    s = _BAD.sub("", s)
    s = _WS.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def simple_similarity(a: str, b: str) -> float:
    # Lightweight similarity heuristic for relation name de-duplication and aliasing.
    a = canon_relation(a)
    b = canon_relation(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta = set(a.split("_"))
    tb = set(b.split("_"))
    if not ta or not tb:
        return 0.0
    j = len(ta & tb) / max(1, len(ta | tb))
    pref = 0.15 if a.startswith(b) or b.startswith(a) else 0.0
    return min(1.0, j + pref)


@dataclass
class ProposedRelation:
    # Relation proposal payload used for pending review or auto-acceptance into the registry.
    name: str
    description: str = ""
    subject_labels: Optional[List[str]] = None
    object_labels: Optional[List[str]] = None
    symmetric: bool = False
    evidence_examples: Optional[List[str]] = None


class RelationRegistry:
    def __init__(
        self,
        allowed: Optional[Set[str]] = None,
        aliases: Optional[Dict[str, str]] = None,
        pending: Optional[Dict[str, ProposedRelation]] = None,
    ):
        # Normalize inputs to canonical form for stable comparisons and persistence.
        self.allowed: Set[str] = set(canon_relation(x) for x in (allowed or set()) if x)
        self.aliases: Dict[str, str] = {}
        for k, v in (aliases or {}).items():
            kk = canon_relation(k)
            vv = canon_relation(v)
            if kk and vv:
                self.aliases[kk] = vv

        self.pending: Dict[str, ProposedRelation] = dict(pending or {})

    def resolve(self, name: str) -> str:
        # Resolve an alias to its canonical relation name.
        c = canon_relation(name)
        return self.aliases.get(c, c)

    def is_allowed(self, name: str) -> bool:
        # Check whether a relation (after alias resolution) is currently allowed.
        c = self.resolve(name)
        return c in self.allowed

    def maybe_merge_as_alias(self, proposed_name: str, threshold: float = 0.88) -> Tuple[bool, str]:
        # Attempt to merge a proposed relation into an existing allowed relation via similarity.
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
        # Normalize the proposal, merge as alias when appropriate, otherwise store/update in pending.
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
        # Accept a relation into the allowed set, optionally registering additional aliases.
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
        # Remove a relation proposal from pending.
        c = canon_relation(name)
        self.pending.pop(c, None)

    def snapshot_allowed_sorted(self) -> List[str]:
        # Return allowed relations in a stable order for prompting and persistence.
        return sorted(self.allowed)

    def save(self, path: Path) -> None:
        # Persist registry state to disk as JSON.
        rec = {
            "allowed": sorted(self.allowed),
            "aliases": dict(self.aliases),
            "pending": {k: asdict(v) for k, v in self.pending.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RelationRegistry":
        # Load registry state from disk, falling back to canonical defaults when no file exists.
        if not path.exists():
            # Startet sauber mit Defaults in kanonischer Form
            allowed = set(canon_relation(x) for x in _DEFAULT_RELATIONS() if x)
            return cls(allowed=allowed, aliases={}, pending={})

        rec = json.loads(path.read_text(encoding="utf-8"))
        allowed = set(rec.get("allowed") or [])
        aliases = dict(rec.get("aliases") or {})
        pending_raw = rec.get("pending") or {}
        pending: Dict[str, ProposedRelation] = {}
        for k, v in pending_raw.items():
            try:
                pending[k] = ProposedRelation(**(v or {}))
            except Exception:
                # Defensive fallback when persisted pending payloads are malformed.
                pending[k] = ProposedRelation(name=k, description=str(v or ""))
        return cls(allowed=allowed, aliases=aliases, pending=pending)


def _DEFAULT_RELATIONS() -> List[str]:
    # Baseline relation set used when no registry file is present.
    return [
        "has_symptom",
        "has_risk_factor",
        "diagnosed_by",
        "treated_with",
        "managed_by",
        "uses_medication",
        "caused_by",
        "associated_with",
        "comorbid_with",
        "occurs_in",
        "affects",
        "can_affect",
        "part_of",
        "indicates",
        "measures",
        "detects",
        "contraindicated_for",
        "has_side_effect",
        "increases_risk_of",
        "related_to",
    ]

