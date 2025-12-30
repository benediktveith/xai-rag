from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import networkx as nx

from src.modules.kg_schema import KGTriple


class KGStore:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def add_triples(self, triples: Iterable[KGTriple]) -> None:
        for t in triples:
            self.g.add_node(t.subject, type=t.subject_type)
            self.g.add_node(t.object, type=t.object_type)
            self.g.add_edge(
                t.subject,
                t.object,
                relation=t.relation,
                source=t.source,
                chunk_id=t.chunk_id,
                meta=t.meta or {},
            )


    def node_type(self, node: str) -> str:
        return str(self.g.nodes[node].get("type", "Unknown"))

    def edge_data(self, u: str, v: str) -> Optional[Dict]:
        return self.g.get_edge_data(u, v)

    def stats(self) -> Dict[str, int]:
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()}

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for u, v, data in self.g.edges(data=True):
                rec = {
                    "u": u,
                    "v": v,
                    "u_type": self.g.nodes[u].get("type", "Unknown"),
                    "v_type": self.g.nodes[v].get("type", "Unknown"),
                    "relation": data.get("relation"),
                    "source": data.get("source"),
                    "chunk_id": data.get("chunk_id"),
                    "meta": data.get("meta", {}),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_jsonl(self, path: Path) -> None:
        self.g.clear()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.g.add_node(rec["u"], type=rec.get("u_type", "Unknown"))
                self.g.add_node(rec["v"], type=rec.get("v_type", "Unknown"))
                self.g.add_edge(
                    rec["u"],
                    rec["v"],
                    relation=rec.get("relation"),
                    source=rec.get("source"),
                    chunk_id=rec.get("chunk_id"),
                    meta=rec.get("meta", {}),
                )
