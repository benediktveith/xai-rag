import json
import random
import zipfile
import gdown
import urllib.request
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document

import tomllib

class MedMCQADataLoader:
    """
    Dateien sind hier: https://github.com/medmcqa/medmcqa?tab=readme-ov-file
    
    Loader für lokal abgelegte MedMCQA Dateien.
    Unterstützt JSON und JSONL.

    Erwartete Pfade:
    data/medmcqa/raw/train.json  oder train.jsonl
    data/medmcqa/raw/dev.json    oder dev.jsonl
    data/medmcqa/raw/test.json   oder test.jsonl
    """

    def __init__(self):
        try:
            src_dir = Path(__file__).resolve().parent.parent
            self.PROJECT_ROOT = src_dir.parent.parent
        except NameError:
            self.PROJECT_ROOT = Path.cwd()

        self.DATA_DIR = self.PROJECT_ROOT / "data" / "medmcqa"
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky&export=download&authuser=0"
        self.FILE_ID = "15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky"

        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def setup(
        self,
        split: str = "train",
        as_documents: bool = True,
        limit: Optional[int] = None,
        ids: List[str] = None,
    ) -> List[Any]:
        split = split.lower().strip()
        if split == "val":
            split = "dev"

        config = self._load_config()
        config_medmcqa = config.get("medmcqa", {}) if isinstance(config.get("medmcqa"), dict) else {}
        if ids is not None:
            self.allowed_ids = ids
        else:
            self.allowed_ids = config_medmcqa.get("question_ids", [])

        seed = self._coerce_int(
            config_medmcqa.get("seed")
        )

        if limit is None:
            limit = self._coerce_int(
                config_medmcqa.get("n_qa_questions")
            )

        path = self._find_split_file(split)
        data = self._load(path, cache_key=split)

        if limit is not None:
            limit = max(0, int(limit))
            data = self._select_with_seed(data, limit, int(seed))

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
        
        # If not found, try to download and extract
        print(f"File for split='{split}' not found. Attempting download...")
        self._download_and_extract()

        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(
            f"Keine Datei für split='{split}' gefunden (auch nach Download). "
            f"Erwartet z.B. {candidates[0]} oder {candidates[1]}."
        )

    def _download_and_extract(self) -> None:
        """Downloads the zip file and extracts files from the 'data/' folder to RAW_DIR."""
        self.RAW_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = self.RAW_DIR / "temp_data.zip"

        try:
            # Download the ZIP file
            if not zip_path.exists():
                print(f"Downloading ID {self.FILE_ID} from Google Drive...")
                url = f'https://drive.google.com/uc?id={self.FILE_ID}'
                gdown.download(url, str(zip_path), quiet=False)

            print("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                targets = {"train.json", "test.json", "dev.json"}
                for member in zip_ref.namelist():
                    # Check if the file is inside the 'data/' folder in the zip
                    print(Path(member).name)
                    
                    if Path(member).name in targets:
                        # Remove 'data/' prefix to extract directly to RAW_DIR
                        filename = Path(member).name
                        target_path = self.RAW_DIR / filename
                        
                        # Open the file from zip and write it to the target path
                        with zip_ref.open(member) as source, open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
            
            print("Download and extraction complete.")

        except Exception as e:
            # Clean up partial downloads if necessary
            raise RuntimeError(f"Failed to download/extract data: {e}")
        finally:
            # Remove the zip file after extraction to save space
            if zip_path.exists():
                zip_path.unlink()

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

    def _load_config(self) -> Dict[str, Any]:
        config_path = self.PROJECT_ROOT / "config.toml"
        
        if not config_path.exists():
            return {}
        if tomllib is None:
            raise RuntimeError("tomllib is required to load config.toml (Python 3.11+).")
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    def _coerce_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _select_with_seed(self, data: List[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
        """
        Randomly (with seed) selects a subset of data according to the allowed indices and the limit.
        """
        data = [data[i] for i in range(len(data)) if data[i].get("id") in self.allowed_ids]

        indices = list(range(len(data)))
        random.Random(seed).shuffle(indices)
        indices = indices[:limit]

        if limit >= len(data):
            return data
        
        return [data[i] for i in indices]
    

    def _to_documents(self, data, split: str):
        from langchain_core.documents import Document

        def cop_to_letter(cop):
            if isinstance(cop, int):
                return {1: "A", 2: "B", 3: "C", 4: "D"}.get(cop)
            if isinstance(cop, str):
                c = cop.strip().upper()
                if c in {"A", "B", "C", "D"}:
                    return c
            return None

        docs = []
        for idx, item in enumerate(data):
            question = str(item.get("question", "")).strip()
            exp = str(item.get("exp", "")).strip()

            opa = str(item.get("opa", "")).strip()
            opb = str(item.get("opb", "")).strip()
            opc = str(item.get("opc", "")).strip()
            opd = str(item.get("opd", "")).strip()

            options_lines = []
            if opa: options_lines.append(f"A: {opa}")
            if opb: options_lines.append(f"B: {opb}")
            if opc: options_lines.append(f"C: {opc}")
            if opd: options_lines.append(f"D: {opd}")

            options_block = "\n".join(options_lines).strip()

            content_parts = []
            if question:
                content_parts.append(question)
            if options_block:
                content_parts.append(options_block)

            if exp:
                content_parts.append(f"Explanation: {exp}")

            page_content = "\n\n".join(content_parts).strip()
            if not page_content:
                continue

            cop = item.get("cop", None)
            gold = cop_to_letter(cop)

            doc = Document(
                page_content=page_content,
                metadata={
                    "question_id": item.get("id", idx),
                    "question": question,
                    "answer": gold,         # A/B/C/D
                    "cop_raw": cop,         # original
                    "split": split,
                    "source": "medmcqa",
                    "subject_name": item.get("subject_name", None),
                    "topic_name": item.get("topic_name", None),
                    "choice_type": item.get("choice_type", None),
                },
            )
            docs.append(doc)

        return docs
    
def format_medmcqa_question(item):
    question = str(item.get("question", "")).strip()
    options = []
    for label, key in [("A", "opa"), ("B", "opb"), ("C", "opc"), ("D", "opd")]:
        opt = str(item.get(key, "")).strip()
        if opt:
            options.append(f"{label}: {opt}")
    if options:
        question = f"{question}\n\nOptions:\n" + "\n".join(options)
    return question.strip()
