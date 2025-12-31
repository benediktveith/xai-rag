# src/modules/rag_engine.py
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class RAGEngine:
    def __init__(self, persist_dir: str = "../data/vector_db"):
        self.persist_dir = persist_dir

        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._vectorstore = None
        self._retriever = None

    def setup(self, documents: Optional[List[Document]] = None, reset: bool = False, k_documents: int = 4):
        """
        Initializes the vector store.
        If documents are provided and DB is empty or reset is True, it ingests them.
        Otherwise, it loads the existing DB.
        """

        db_path = Path(self.persist_dir)

        if reset and db_path.exists():
            print(f"Clearing existing vector store at {self.persist_dir}...")
            shutil.rmtree(self.persist_dir)

        if db_path.exists() and any(db_path.iterdir()):
            print(f"Loading existing vector store from {self.persist_dir}...")
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_model,
            )
        else:
            if not documents:
                raise ValueError("No existing vector store found. You must provide documents to create one.")
            print(f"Creating new vector store with {len(documents)} documents...")
            self._create_vector_store(documents)

        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_documents},
        )
        print("RagEngine ready.")

    def _create_vector_store(self, documents: List[Document]):
        """
        Ingest Documents direkt, ohne erneut zu splitten,
        damit chunk_id aus metadata stabil bleibt.
        """

        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir,
        )

    def retrieve_documents(self, query: str) -> List[Document]:
        if not self._retriever:
            raise RuntimeError("RagEngine not setup. Call setup() first.")
        return self._retriever.invoke(query)

    def retrieve_with_scores(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if not self._vectorstore:
            raise RuntimeError("RagEngine not setup. Call setup() first.")

        results = self._vectorstore.similarity_search_with_relevance_scores(query, k=k)
        trace: List[Dict[str, Any]] = []

        for idx, (doc, score) in enumerate(results):
            s = float(score)
            if s < 0.0:
                s = 0.0
            if s > 1.0:
                s = 1.0

            meta = (doc.metadata or {}).copy()
            chunk_id = str(meta.get("chunk_id") or f"chunk-{idx+1}")

            trace.append(
                {
                    "id": chunk_id,
                    "score": s,
                    "title": meta.get("title", ""),
                    "content": doc.page_content,
                    "metadata": meta,
                }
            )
        return trace

    def get_retriever(self):
        return self._retriever
