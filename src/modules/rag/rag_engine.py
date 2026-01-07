import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGEngine:
    def __init__(self, persist_dir: str = "../data/vector_db"):
        self.persist_dir = persist_dir
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._vectorstore = None
        self._retriever = None
        self._default_k = 4 # Default k value

    def setup(self, documents: Optional[List[Document]] = None, reset: bool = False, k: int = 4):
        """
        Initializes the vector store.
        If 'documents' are provided and DB is empty (or reset=True), it ingests them.
        Otherwise, it loads the existing DB.
        """
        self._default_k = k
        db_path = Path(self.persist_dir)

        if reset and db_path.exists():
            print(f"Clearing existing vector store at {self.persist_dir}...")
            shutil.rmtree(self.persist_dir)

        if db_path.exists() and any(db_path.iterdir()):
            print(f"Loading existing vector store from {self.persist_dir}...")
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_model
            )
        else:
            if not documents:
                raise ValueError("No existing vector store found. You must provide 'documents' to create one.")
            
            print(f"Creating new vector store with {len(documents)} documents...")
            self._create_vector_store(documents)

        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._default_k}
        )
        print("RagEngine ready.")

    def _create_vector_store(self, documents: List[Document]):
        """Internal: Splits text and saves to Chroma"""
        text_splitter = RecursiveCharacterTextSplitter()
        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks.")

        self._vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir
        )

    def retrieve_documents(self, query: str, k_documents: Optional[int] = None) -> List[Document]:
        """
        Retrieves documents. If k_documents is specified, it overrides the default k for this call.
        """
        if not self._retriever:
            raise RuntimeError("RagEngine not setup. Call setup() first.")
        
        # If a specific k is provided for this call, use it
        if k_documents is not None:
            return self._vectorstore.similarity_search(query, k=k_documents)
        
        # Otherwise, use the default retriever settings
        return self._retriever.invoke(query)

    def retrieve_with_scores(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Returns retrieval trace with scores and metadata for explainability.
        """
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

            text = doc.page_content
            trace.append(
                {
                    "id": f"chunk-{idx+1}",
                    "score": s,
                    "title": doc.metadata.get("title", ""),
                    "content": text,
                    "metadata": doc.metadata,
                }
            )
        return trace

    def get_retriever(self):
        """Returns the raw retriever object for use in LangChain chains"""
        return self._retriever