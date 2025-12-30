import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


class MedMCQADataLoader:
    """
    Loader für lokal abgelegte MedMCQA Dateien.
    Unterstützt JSON und JSONL.

    Erwartete Pfade:
    data/medmcqa/raw/train.json  oder train.jsonl
    data/medmcqa/raw/dev.json    oder dev.jsonl
    data/medmcqa/raw/test.json   oder test.jsonl
    """

    def __init__(self):
        try:
            src_dir = Path(__file__).resolve().parent
            self.PROJECT_ROOT = src_dir.parent.parent
        except NameError:
            self.PROJECT_ROOT = Path.cwd()

        self.DATA_DIR = self.PROJECT_ROOT / "data" / "medmcqa"
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.RAW_DIR.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def setup(
        self,
        split: str = "train",
        as_documents: bool = True,
        limit: Optional[int] = None,
    ) -> List[Any]:
        split = split.lower().strip()
        if split == "val":
            split = "dev"

        path = self._find_split_file(split)
        data = self._load(path, cache_key=split)

        if limit is not None:
            data = data[:limit]

        if as_documents:
            return self._to_documents(data, split)
        return data

    def _find_split_file(self, split: str) -> Path:
        candidates = [
            self.RAW_DIR / f"{split}.json",
            self.RAW_DIR / f"{split}.jsonl",
        ]
        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(
            f"Keine Datei für split='{split}' gefunden. Erwartet z.B. {candidates[0]} oder {candidates[1]}."
        )

    def _load(self, path: Path, cache_key: str) -> List[Dict[str, Any]]:
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Erst als JSONL versuchen, weil dein Fehlerbild exakt danach aussieht
        out: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
            # Wenn wir mindestens 1 Objekt parsed haben und nicht alles leer war, ist es JSONL
            if out:
                self._cache[cache_key] = out
                return out
        except Exception:
            out = []

        # Fallback: klassisches JSON (Liste oder Objekt)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data = data["data"]

        if not isinstance(data, list):
            raise ValueError(
                f"Unerwartetes Format in {path}. Erwartet JSONL oder JSON Liste oder Objekt mit Feld 'data' als Liste."
            )

        self._cache[cache_key] = data
        return data


    def _to_documents(self, data: List[Dict[str, Any]], split: str) -> List[Document]:
        docs: List[Document] = []
        for idx, item in enumerate(data):
            question = str(item.get("question", "")).strip()

            options = item.get("options")
            if isinstance(options, dict) and options:
                options_text = " ".join(str(v) for v in options.values() if v)
            else:
                options_text = ""

            answer = item.get("answer", item.get("correct", None))

            doc = Document(
                page_content=options_text,
                metadata={
                    "question_id": item.get("id", idx),
                    "question": question,
                    "answer": answer,
                    "split": split,
                    "source": "medmcqa",
                },
            )
            docs.append(doc)
        return docs
