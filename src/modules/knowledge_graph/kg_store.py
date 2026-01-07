
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Any, Optional

import networkx as nx

from src.modules.knowledge_graph.kg_schema import KGTriple


_WS = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u00a0", " ")
    s = _WS.sub(" ", s)
    return s


def _canon_node(s: str) -> str:
    return _norm_text(s).casefold()


def _dedup_preserve_order(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


class KGStore:
    """
    KG Store als DiGraph.

    - Eine Kante pro (u, v). Relation Mixing wird verhindert.
    - Observations sind die kanonische Quelle fuer alles.
    """

    def __init__(self):
        self.g = nx.DiGraph()
        self._alias_by_canon: Dict[str, str] = {}

    def _resolve_node_id(self, name: str) -> str:
        raw = _norm_text(name)
        if not raw:
            return ""
        c = _canon_node(raw)
        if c in self._alias_by_canon:
            return self._alias_by_canon[c]
        self._alias_by_canon[c] = raw
        return raw

    def _ensure_node(self, node: str, node_type: str) -> str:
        nid = self._resolve_node_id(node)
        if not nid:
            return ""

        ntype = str(node_type or "Unknown")

        if nid not in self.g:
            self.g.add_node(nid, type=ntype, type_conflicts=[])
            return nid

        existing = str(self.g.nodes[nid].get("type", "Unknown") or "Unknown")
        if existing in ("", "Unknown") and ntype not in ("", "Unknown"):
            self.g.nodes[nid]["type"] = ntype
        elif existing not in ("", "Unknown") and ntype not in ("", "Unknown") and existing != ntype:
            conflicts = self.g.nodes[nid].get("type_conflicts") or []
            entry = {"seen": ntype}
            if entry not in conflicts:
                conflicts.append(entry)
            self.g.nodes[nid]["type_conflicts"] = conflicts

        return nid

    def _pick_primary_relation(self, observations: List[Dict[str, Any]]) -> str:
        for obs in observations:
            meta = obs.get("meta") if isinstance(obs.get("meta"), dict) else {}
            if not bool(meta.get("synthetic", False)):
                return str(obs.get("relation", "") or "")
        return str(observations[0].get("relation", "") or "") if observations else ""
    
    def node_name(self, node: str) -> str:
        """
        Stabiler Display Name. Da wir node ids als kanonisierte Raw Strings halten,
        ist der Knotenname identisch zur Node ID.
        """
        return str(node or "")

    def edge_attr(self, u: str, v: str) -> Optional[dict]:
        """
        Gibt die Edge Attribute für (u,v) zurück, oder None falls keine Kante existiert.
        """
        if not u or not v:
            return None
        if not self.g.has_edge(u, v):
            return None
        data = self.g.get_edge_data(u, v) or {}
        return dict(data)


    def add_triples(self, triples: Iterable[KGTriple]) -> None:
        for t in triples:
            u = self._ensure_node(t.subject, t.subject_type)
            v = self._ensure_node(t.object, t.object_type)
            if not u or not v:
                continue

            rel = _norm_text(t.relation)
            if not rel:
                continue

            src = _norm_text(t.source)
            cid = _norm_text(t.chunk_id)
            meta = t.meta if isinstance(t.meta, dict) else {}

            obs = {"relation": rel, "source": src, "chunk_id": cid, "meta": meta}

            if self.g.has_edge(u, v):
                data = dict(self.g.get_edge_data(u, v) or {})
                existing_rel = str(data.get("relation", "") or "")

                if existing_rel and existing_rel != rel:
                    conflicts = data.get("relation_conflicts") or []
                    conflicts.append(obs)
                    self.g[u][v].update({"relation_conflicts": conflicts})
                    continue

                observations = list(data.get("observations") or [])
                observations.append(obs)

                sources = list(data.get("sources") or [])
                chunk_ids = list(data.get("chunk_ids") or [])
                if src:
                    sources.append(src)
                if cid:
                    chunk_ids.append(cid)

                sources = _dedup_preserve_order(sources)
                chunk_ids = _dedup_preserve_order(chunk_ids)

                primary = self._pick_primary_relation(observations)

                self.g[u][v].update(
                    {
                        "relation": primary,
                        "observations": observations,
                        "sources": sources,
                        "chunk_ids": chunk_ids,
                    }
                )
            else:
                observations = [obs]
                sources = [src] if src else []
                chunk_ids = [cid] if cid else []
                primary = self._pick_primary_relation(observations)

                self.g.add_edge(
                    u,
                    v,
                    relation=primary,
                    observations=observations,
                    sources=sources,
                    chunk_ids=chunk_ids,
                    relation_conflicts=[],
                )

    def node_type(self, node: str) -> str:
        return str(self.g.nodes[node].get("type", "Unknown"))

    def stats(self) -> Dict[str, int]:
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()}

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for u, v, data in self.g.edges(data=True):
                observations = list(data.get("observations") or [])

                relations = [str(o.get("relation", "") or "") for o in observations if o.get("relation")]
                relations = _dedup_preserve_order([r for r in relations if r])

                metas = [o.get("meta", {}) for o in observations if isinstance(o.get("meta"), dict)]

                rec = {
                    "u": u,
                    "v": v,
                    "u_type": self.g.nodes[u].get("type", "Unknown"),
                    "v_type": self.g.nodes[v].get("type", "Unknown"),
                    "relation": str(data.get("relation", "") or ""),
                    "relations": relations,
                    "sources": list(data.get("sources") or []),
                    "chunk_ids": list(data.get("chunk_ids") or []),
                    "metas": metas,
                    # diagnostics and lossless payload
                    "relation_conflicts": list(data.get("relation_conflicts") or []),
                    "observations": observations,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_jsonl(self, path: Path) -> None:
        self.g.clear()
        self._alias_by_canon.clear()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)

                u = self._ensure_node(str(rec.get("u", "")), str(rec.get("u_type", "Unknown")))
                v = self._ensure_node(str(rec.get("v", "")), str(rec.get("v_type", "Unknown")))
                if not u or not v:
                    continue

                relation = str(rec.get("relation", "") or "")
                sources = list(rec.get("sources") or [])
                chunk_ids = list(rec.get("chunk_ids") or [])

                # Prefer lossless observations if present
                observations = list(rec.get("observations") or [])
                if not observations:
                    relations = list(rec.get("relations") or [])
                    metas = list(rec.get("metas") or [])

                    # coarse fallback: attach first source/chunk_id if available
                    src0 = sources[0] if sources else ""
                    cid0 = chunk_ids[0] if chunk_ids else ""

                    observations = []
                    if relations and metas and len(relations) == len(metas):
                        for r, m in zip(relations, metas):
                            observations.append(
                                {"relation": str(r), "source": src0, "chunk_id": cid0, "meta": m if isinstance(m, dict) else {}}
                            )
                    else:
                        for m in metas:
                            observations.append(
                                {"relation": relation, "source": src0, "chunk_id": cid0, "meta": m if isinstance(m, dict) else {}}
                            )

                if not relation:
                    relation = self._pick_primary_relation(observations)

                self.g.add_edge(
                    u,
                    v,
                    relation=relation,
                    observations=observations,
                    sources=sources,
                    chunk_ids=chunk_ids,
                    relation_conflicts=list(rec.get("relation_conflicts") or []),
                )
