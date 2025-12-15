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
        
        # Best local embedding model for general RAG (fast & standard)
        # You can swap this for OllamaEmbeddings if you want 100% Ollama
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._vectorstore = None
        self._retriever = None

    def setup(self, documents: Optional[List[Document]] = None, reset: bool = False):
        """
        Initializes the vector store.
        If 'documents' are provided and DB is empty (or reset=True), it ingests them.
        Otherwise, it loads the existing DB.
        """
        db_path = Path(self.persist_dir)

        # A. Reset Logic: Delete existing DB if requested
        if reset and db_path.exists():
            print(f"Clearing existing vector store at {self.persist_dir}...")
            shutil.rmtree(self.persist_dir)

        # B. Check if DB exists
        if db_path.exists() and any(db_path.iterdir()):
            print(f"Loading existing vector store from {self.persist_dir}...")
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_model
            )
        else:
            # C. Create new DB
            if not documents:
                raise ValueError("No existing vector store found. You must provide 'documents' to create one.")
            
            print(f"Creating new vector store with {len(documents)} documents...")
            self._create_vector_store(documents)

        # Initialize the retriever interface
        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4} # Adjust 'k' based on HotpotQA needs
        )
        print("RagEngine ready.")

    def _create_vector_store(self, documents: List[Document]):
        """Internal: Splits text and saves to Chroma"""
        
        # 1. Split Documents (Crucial for RAG)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Size of each chunk
            chunk_overlap=50 # Overlap to maintain context
        )
        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks.")

        # 2. Ingest into Chroma
        self._vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir
        )

    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Raw retrieval: Get documents without sending to LLM.
        Useful for inspection or custom pipelines.
        """
        if not self._retriever:
            raise RuntimeError("RagEngine not setup. Call setup() first.")
        
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
            text = doc.page_content
            trace.append(
                {
                    "id": f"chunk-{idx+1}",
                    "score": float(score),
                    "title": doc.metadata.get("title", ""),
                    "content": text,
                    "metadata": doc.metadata,
                }
            )
        return trace

    def get_retriever(self):
        """Returns the raw retriever object for use in LangChain chains"""
        return self._retriever
