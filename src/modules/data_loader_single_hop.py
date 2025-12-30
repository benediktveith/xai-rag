import json
import requests
import pyarrow
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document


class BoolQDataLoader:
    """
    Loader for the BoolQ dataset (single-hop yes/no QA).
    Supports both legacy JSONL (GCS) and Parquet (HF) sources.
    Converts each example into a LangChain Document with the passage as page_content
    and question/answer metadata.
    """

    # Candidate sources per split; loader picks the first that downloads successfully.
    BOOLQ_SOURCES: Dict[str, List[Dict[str, str]]] = {
        "train": [
            {"format": "jsonl", "url": "https://storage.googleapis.com/boolq/train.jsonl"},
            {
                "format": "parquet",
                "url": "https://huggingface.co/datasets/google/boolq/resolve/main/data/train-00000-of-00001.parquet?download=1",
            },
        ],
        "validation": [
            {"format": "jsonl", "url": "https://storage.googleapis.com/boolq/val.jsonl"},
            {
                "format": "parquet",
                "url": "https://huggingface.co/datasets/google/boolq/resolve/main/data/validation-00000-of-00001.parquet?download=1",
            },
        ],
    }

    def __init__(self):
        try:
            src_dir = Path(__file__).resolve().parent
            self.PROJECT_ROOT = src_dir.parent.parent
        except NameError:
            # Fallback when __file__ is not set (e.g., in some notebook contexts)
            self.PROJECT_ROOT = Path.cwd()

        self.DATA_DIR = self.PROJECT_ROOT / "data" / "boolq"
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.RAW_DIR.mkdir(parents=True, exist_ok=True)

        # Will hold cached split data after first load
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def setup(
        self,
        split: str = "train",
        as_documents: bool = True,
        limit: Optional[int] = None,
    ) -> List[Any]:
        """
        Ensure the requested BoolQ split is downloaded, load it, and return either
        LangChain Documents or the raw list of examples.

        Args:
            split: "train" or "validation" (also accepts "val").
            as_documents: If True, returns List[Document]; otherwise raw dicts.
            limit: Optional cap on the number of examples (useful for quick tests).
        """
        split = split.lower()
        if split == "val":
            split = "validation"
        if split not in self.BOOLQ_SOURCES:
            raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'validation'.")

        path, fmt = self._ensure_split_file(split)
        data = self._load_split(path, split, fmt)

        if limit is not None:
            data = data[:limit]

        if as_documents:
            return self._to_documents(data, split=split)
        return data

    def _existing_local(self, split: str) -> Optional[Tuple[Path, str]]:
        """Return existing local file if present (prefers parquet over jsonl if both)."""
        parquet_path = self.RAW_DIR / f"{split}.parquet"
        jsonl_path = self.RAW_DIR / f"{split}.jsonl"
        if parquet_path.exists():
            return parquet_path, "parquet"
        if jsonl_path.exists():
            return jsonl_path, "jsonl"
        return None

    def _ensure_split_file(self, split: str) -> Tuple[Path, str]:
        """Download the requested split if it is not present yet."""
        existing = self._existing_local(split)
        if existing:
            return existing

        errors = []
        for source in self.BOOLQ_SOURCES[split]:
            fmt = source["format"]
            url = source["url"]
            target = self.RAW_DIR / f"{split}.{fmt}"
            try:
                resp = requests.get(url, stream=True)
                resp.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return target, fmt
            except Exception as e:
                errors.append(f"{url}: {e}")
                if target.exists():
                    target.unlink()

        raise RuntimeError(
            f"Failed to download BoolQ split '{split}'. Tried: " + "; ".join(errors)
        )

    def _load_split(self, path: Path, split: str, fmt: str) -> List[Dict[str, Any]]:
        """Load a split into memory, with simple caching."""
        if split in self._cache:
            return self._cache[split]

        if fmt == "jsonl":
            examples: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    examples.append(json.loads(line))
        elif fmt == "parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                try:
                    import pandas as pd
                except ImportError as e:
                    raise RuntimeError(
                        "Parquet loading requires either 'pyarrow' or 'pandas' to be installed."
                    ) from e
                df = pd.read_parquet(path)
                examples = df.to_dict(orient="records")
            else:
                table = pq.read_table(path)
                cols = table.to_pydict()
                keys = list(cols.keys())
                length = len(next(iter(cols.values()), []))
                examples = [{k: cols[k][i] for k in keys} for i in range(length)]
        else:
            raise ValueError(f"Unsupported format '{fmt}' for split '{split}'.")

        self._cache[split] = examples
        return examples

    def _to_documents(self, data: List[Dict[str, Any]], split: str) -> List[Document]:
        """Convert BoolQ examples to LangChain Documents."""
        docs: List[Document] = []
        for idx, item in enumerate(data):
            question = item.get("question", "")
            passage = item.get("passage", "")
            answer = item.get("answer", None)

            doc = Document(
                page_content=passage,
                metadata={
                    "question": question,
                    "answer": answer,
                    "split": split,
                    "source": "boolq",
                    "id": idx,
                },
            )
            docs.append(doc)
        return docs
