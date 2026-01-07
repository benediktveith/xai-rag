import torch
import torch.nn as nn
from typing import List, Any
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModel
from .colbert_wrapper import NativeColbertWrapper

class NativeColbertRAGEngine:
    """
    A PyTorch-native RAG Engine compatible with LangChain Documents.
    """
    def __init__(self, documents: List[Document], model_name="colbert-ir/colbertv2.0"):
        print("Initializing Native ColBERT Engine...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = NativeColbertWrapper(model_name)
        self.model.eval()
        self._retriever = True 
        
        # Store the original Document objects (so we keep metadata!)
        self.documents = documents
        
        # Extract text content for indexing
        # Your dataloader puts the text in 'page_content'
        doc_texts = [doc.page_content for doc in documents]
        
        print(f"Indexing {len(documents)} documents...")
        self.doc_embeddings = self._index_documents(doc_texts)

    def _index_documents(self, texts: List[str]):
        """Helper to turn text strings into cached embeddings."""
        embeddings = []
        with torch.no_grad():
            for text in texts:
                # Truncate to 256 tokens for speed/memory on CPU
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                emb = self.model(inputs["input_ids"], inputs["attention_mask"])
                # Keep embeddings on CPU RAM to avoid GPU OOM if you have many docs
                embeddings.append(emb.cpu())
        return embeddings

    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
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
        top_k = scores[:k]
        
        results = []
        for score, idx in top_k:
            original_doc = self.documents[idx]
            
            # Create a new Document to attach the score without mutating the original dataset
            # We copy the original rich metadata (gold_indices, title, etc.)
            new_metadata = original_doc.metadata.copy()
            new_metadata["score"] = score
            
            results.append(Document(
                page_content=original_doc.page_content, 
                metadata=new_metadata
            ))
            
        return results
