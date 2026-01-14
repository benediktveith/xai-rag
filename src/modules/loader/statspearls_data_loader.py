from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from lxml import etree
except Exception:
    etree = None


@dataclass(frozen=True)
class StatPearlsBuildStats:
    tarball_downloaded: bool
    extracted: bool
    nxml_files_found: int
    jsonl_files_created: int
    articles_loaded: int
    chunks_emitted: int


class StatPearlsDataLoader:
    """
    Repo Stil, aber als Loader Datei.

    Quelle, wie im Repo:
    https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz

    Output Ordnerstruktur:
    <root_dir>/statpearls_NBK430685/               , entpackter Inhalt
    <root_dir>/statpearls_NBK430685/chunks/        , JSONL pro Artikel
    """

    DEFAULT_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"

    def __init__(
        self,
        root_dir: str = "../data",
        dataset_dirname: str = "statpearls_NBK430685",
        tarball_name: str = "statpearls_NBK430685.tar.gz",
        url: str = DEFAULT_URL,
        chunk_size: int = 600,
        overlap: int = 100,
    ):
        self.root_dir = Path(root_dir)
        self.dataset_dir = self.root_dir / dataset_dirname
        self.chunks_dir = self.dataset_dir / "chunks"
        self.tarball_path = self.root_dir / tarball_name
        self.url = url

        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

    def setup(
        self,
        limit_articles: Optional[int] = 300,
        as_documents: bool = True,
        force_download: bool = False,
        force_extract: bool = False,
        force_rebuild_jsonl: bool = False,
    ) -> Tuple[List[Document], StatPearlsBuildStats]:
        downloaded = self._ensure_tarball(force_download=force_download)
        extracted, nxml_found = self._ensure_extracted(force_extract=force_extract)
        jsonl_created = self._ensure_jsonl(force_rebuild=force_rebuild_jsonl)

        docs = self._load_chunks_as_documents(limit_articles=limit_articles)
        stats = StatPearlsBuildStats(
            tarball_downloaded=downloaded,
            extracted=extracted,
            nxml_files_found=nxml_found,
            jsonl_files_created=jsonl_created,
            articles_loaded=self._count_jsonl_files(limit_articles=limit_articles),
            chunks_emitted=len(docs),
        )
        return docs, stats

    def _ensure_tarball(self, force_download: bool) -> bool:
        if self.tarball_path.exists() and not force_download:
            return False

        import urllib.request

        self.tarball_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(self.url) as resp:
            data = resp.read()
        self.tarball_path.write_bytes(data)
        return True

    def _ensure_extracted(self, force_extract: bool) -> Tuple[bool, int]:
        marker = self.dataset_dir / ".extracted_ok"
        if marker.exists() and not force_extract:
            return False, self._count_nxml_files()

        if self.dataset_dir.exists() and force_extract:
            marker.unlink(missing_ok=True)

        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(self.tarball_path, "r:gz") as tf:
            tf.extractall(path=self.dataset_dir)

        nxml_count = self._count_nxml_files()
        marker.write_text("ok", encoding="utf-8")
        return True, nxml_count

    def _ensure_jsonl(self, force_rebuild: bool) -> int:
        existing = list(self.chunks_dir.glob("*.jsonl"))
        if existing and not force_rebuild:
            return 0

        if force_rebuild:
            for fp in existing:
                try:
                    fp.unlink()
                except Exception:
                    pass

        nxml_files = self._list_nxml_files()
        created = 0

        for nxml_path in nxml_files:
            base = nxml_path.stem
            out_path = self.chunks_dir / f"{base}.jsonl"
            if out_path.exists() and not force_rebuild:
                continue

            article = self._parse_nxml_to_article(nxml_path)
            if not article:
                continue

            out_path.write_text(json.dumps(article, ensure_ascii=False) + "\n", encoding="utf-8")
            created += 1

        return created

    def _load_chunks_as_documents(self, limit_articles: Optional[int]) -> List[Document]:
        jsonl_files = sorted(self.chunks_dir.glob("*.jsonl"))
        if limit_articles is not None:
            jsonl_files = jsonl_files[: max(1, int(limit_articles))]

        docs: List[Document] = []

        for fp in jsonl_files:
            rec = self._read_one_jsonl(fp)
            if not rec:
                continue

            title = str(rec.get("title", "") or "").strip()
            abstract = str(rec.get("abstract", "") or "").strip()
            body = str(rec.get("body", "") or "").strip()
            filename = str(rec.get("filename", fp.name) or fp.name)

            text_parts = [p for p in [title, abstract, body] if p.strip()]
            text = "\n\n".join(text_parts).strip()
            if not text:
                continue

            chunks = self.splitter.split_text(text)
            for idx, chunk in enumerate(chunks):
                chunk_id = self._chunk_hash(chunk)

                meta: Dict[str, Any] = {
                    "source": "statpearls",
                    "split": "repo",
                    "title": title or "StatPearls",
                    "topic_name": title or "StatPearls",
                    "source_filename": filename,
                    "chunk_index": idx,
                    "chunk_id": chunk_id,
                }

                docs.append(Document(page_content=chunk, metadata=meta))

        return docs

    def _chunk_hash(self, text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def _read_one_jsonl(self, fp: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                line = f.readline()
                if not line:
                    return None
                return json.loads(line)
        except Exception:
            return None

    def _count_nxml_files(self) -> int:
        return len(self._list_nxml_files())

    def _list_nxml_files(self) -> List[Path]:
        return sorted(self.dataset_dir.rglob("*.nxml"))

    def _count_jsonl_files(self, limit_articles: Optional[int]) -> int:
        files = sorted(self.chunks_dir.glob("*.jsonl"))
        if limit_articles is not None:
            files = files[: max(1, int(limit_articles))]
        return len(files)

    def _parse_nxml_to_article(self, file_path: Path) -> Optional[Dict[str, Any]]:
        if etree is None:
            return self._parse_nxml_fallback(file_path)

        try:
            tree = etree.parse(str(file_path))
            root = tree.getroot()
            ns = {"ns": root.nsmap.get(None, "")}

            def extract_text(elem) -> str:
                if elem is None:
                    return ""
                return "".join(elem.itertext()).strip()

            title_elem = root.find(".//ns:article-title", namespaces=ns)
            abstract_elem = root.find(".//ns:abstract", namespaces=ns)
            body_elem = root.find(".//ns:body", namespaces=ns)

            title = self._cleanup_text(extract_text(title_elem))
            abstract = self._cleanup_text(extract_text(abstract_elem))
            body = self._cleanup_text(extract_text(body_elem))

            if not (title or abstract or body):
                return None

            return {
                "filename": file_path.name,
                "title": title,
                "abstract": abstract,
                "body": body,
            }
        except Exception:
            return None

    def _parse_nxml_fallback(self, file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            import xml.etree.ElementTree as ET

            raw = file_path.read_bytes()
            root = ET.fromstring(raw)

            def all_text(elem) -> str:
                if elem is None:
                    return ""
                parts: List[str] = []
                for e in elem.iter():
                    if e.text and e.text.strip():
                        parts.append(e.text.strip())
                    if e.tail and e.tail.strip():
                        parts.append(e.tail.strip())
                return " ".join(parts)

            title = self._cleanup_text(all_text(root.find(".//article-title")))
            abstract = self._cleanup_text(all_text(root.find(".//abstract")))
            body = self._cleanup_text(all_text(root.find(".//body")))

            if not (title or abstract or body):
                return None

            return {
                "filename": file_path.name,
                "title": title,
                "abstract": abstract,
                "body": body,
            }
        except Exception:
            return None

    def _cleanup_text(self, s: str) -> str:
        s = (s or "").replace("\u00a0", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s