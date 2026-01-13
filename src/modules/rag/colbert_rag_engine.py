import torch
import torch.nn as nn
import os
import torch
import pickle
import torch.nn as nn
from typing import List, Any, Optional
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModel
from .colbert_wrapper import NativeColbertWrapper

class NativeColbertRAGEngine:
    """
    A PyTorch-native RAG Engine compatible with LangChain Documents.
    Supports saving/loading the index to/from disk.
    """
    def __init__(self, persist_dir: str = "../data/colbert_index", model_name="colbert-ir/colbertv2.0"):
        print("Initializing Native ColBERT Engine...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = NativeColbertWrapper(model_name)
        self.model.eval()
        self.k = None
        
        self.persist_dir = persist_dir
        self.documents: Optional[List[Document]] = None
        self.doc_embeddings: Optional[List[torch.Tensor]] = None
        
        # Ensure the directory exists
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
            
        # Try to load existing index immediately
        self.load_index()

    def setup(self, documents: List[Document], k: int=4):
        """
        Indexes new documents and saves the index to disk.
        """
        # set k for retrieval
        self.k = k

        if not documents:
            raise ValueError("No documents provided, could not setup ColbertRAGEngine.")

        self.documents = documents
        doc_texts = [doc.page_content for doc in documents]
        
        print(f"Indexing {len(documents)} documents...")
        self.doc_embeddings = self._index_documents(doc_texts)
        
        # Save the newly created index
        self.save_index()

    def _index_documents(self, texts: List[str]):
        """Helper to turn text strings into cached embeddings."""
        embeddings = []
        total = len(texts)
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                # Simple progress log
                if i % 10 == 0 or i == total - 1:
                    print(f"Encoding doc {i+1}/{total}", end='\r')

                # Truncate to 256 tokens for speed/memory on CPU
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                emb = self.model(inputs["input_ids"], inputs["attention_mask"])
                
                # Keep embeddings on CPU RAM to avoid GPU OOM
                embeddings.append(emb.cpu())
        print("\nIndexing complete.")
        return embeddings

    def save_index(self):
        """Saves documents and embeddings to the persist_dir."""
        if self.documents is None or self.doc_embeddings is None:
            print("Nothing to save.")
            return

        emb_path = os.path.join(self.persist_dir, "embeddings.pt")
        doc_path = os.path.join(self.persist_dir, "documents.pkl")

        print(f"Saving index to {self.persist_dir}...")
        
        # Save Embeddings (efficient torch format)
        torch.save(self.doc_embeddings, emb_path)
        
        # Save Documents (pickle for objects)
        with open(doc_path, "wb") as f:
            pickle.dump(self.documents, f)
            
        print("Index saved successfully.")

    def load_index(self):
        """Loads documents and embeddings from the persist_dir if they exist."""
        emb_path = os.path.join(self.persist_dir, "embeddings.pt")
        doc_path = os.path.join(self.persist_dir, "documents.pkl")

        if os.path.exists(emb_path) and os.path.exists(doc_path):
            print(f"Found existing index at {self.persist_dir}. Loading...")
            
            try:
                # Load Embeddings
                self.doc_embeddings = torch.load(emb_path, map_location="cpu")
                
                # Load Documents
                with open(doc_path, "rb") as f:
                    self.documents = pickle.load(f)
                
                print(f"Loaded {len(self.documents)} documents and embeddings.")
            except Exception as e:
                print(f"Error loading index: {e}")
        else:
            print(f"No existing index found at {self.persist_dir}. Please call setup() to index documents.")

    def retrieve_documents(self, query: str) -> List[Document]:
        if not self.doc_embeddings or not self.documents:
            raise RuntimeError("Index not loaded. Call setup() to index or check persist_dir.")

        # 1. Encode Query
        q_inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=64)
        
        # Move query to model device (GPU/CPU) for inference
        with torch.no_grad():
            q_emb = self.model(q_inputs["input_ids"], q_inputs["attention_mask"])
        
        # 2. Linear Scan Score
        scores = []
        for i, d_emb in enumerate(self.doc_embeddings):
            # Move doc embedding to model device temporarily for calculation
            d_emb_device = d_emb.to(self.model.device)
            
            score = self.model.score(q_emb, d_emb_device).item()
            scores.append((score, i))
        
        # 3. Sort and Return
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:self.k]
        
        results = []
        for score, idx in top_k:
            original_doc = self.documents[idx]
            new_metadata = original_doc.metadata.copy()
            new_metadata["score"] = score
            
            results.append(Document(
                page_content=original_doc.page_content, 
                metadata=new_metadata
            ))
            
        return results