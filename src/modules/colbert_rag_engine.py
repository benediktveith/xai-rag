from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from ragatouille import RAGPretrainedModel
import os
import shutil

class ColbertRAGEngine:
    """
    A RAG engine that uses ColBERT for dense retrieval, powered by Ragatouille.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        """
        Initializes the ColbertRAGEngine.

        :param model_name: The name of the ColBERT model to use.
        """
        self.model_name = model_name
        self.rag_model = RAGPretrainedModel.from_pretrained(self.model_name)
        self.collection_name = None # Will be set during indexing

    def index(self, documents: List[Document], collection_name: str, force_reindex: bool = False):
        """
        Creates or loads a ColBERT index.

        :param documents: A list of LangChain Document objects to index.
        :param collection_name: A unique name for this document collection/index.
        :param force_reindex: If True, deletes any existing index with the same name and recreates it.
        """
        self.collection_name = collection_name
        
        # Ragatouille stores indices in a default location, typically ~/.ragatouille/
        # We need to manage reindexing externally if a user wants to reset.
        
        # Check if index exists and force_reindex is True, then delete
        # NOTE: This assumes ragatouille's default index location logic
        # For a more robust solution, check rag_model.list_indexes() and delete via model.delete_index()
        # For simplicity here, we'll rely on Ragatouille's internal handling and a prompt from user.
        
        doc_contents = [doc.page_content for doc in documents]
        doc_metadatas = [doc.metadata for doc in documents]

        print(f"Indexing {len(doc_contents)} documents with ColBERT (collection: {collection_name})...")
        self.rag_model.index(
            collection=doc_contents,
            collection_as_dict=doc_metadatas, # Store metadata
            index_name=self.collection_name,
            overwrite=force_reindex
        )
        print(f"ColBERT index '{collection_name}' ready.")

    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieves documents from the ColBERT index for a given query.

        :param query: The search query.
        :param k: The number of top documents to retrieve.
        :return: A list of LangChain Document objects.
        """
        if not self.collection_name:
            raise RuntimeError("ColBERT index not set up. Call index() first.")
        
        results = self.rag_model.search(query=query, k=k, index_name=self.collection_name)
        
        retrieved_docs = []
        for res in results:
            # Ragatouille results are dicts with 'content' and 'score' and other metadata
            doc = Document(page_content=res['content'], metadata={**res['metadata'], 'score': res['score']})
            retrieved_docs.append(doc)
        
        return retrieved_docs

    def rerank_documents(self, query: str, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Reranks a given list of documents against a query using the ColBERT model
        and returns the documents with their ColBERT scores. This is suitable for
        explaining the ranking.

        :param query: The query string.
        :param documents: A list of LangChain Document objects to rerank.
        :return: A list of dictionaries, each containing 'document', 'score', and other metadata.
        """
        if not self.rag_model:
            raise RuntimeError("ColBERT model not loaded. Call __init__() first.")

        document_contents = [doc.page_content for doc in documents]
        
        # Ragatouille's rerank method directly gives us scores
        # It expects a query and a list of passages
        scores = self.rag_model.rerank(query=query, passages=document_contents)
        
        reranked_results = []
        for i, doc in enumerate(documents):
            reranked_results.append({
                "document": doc,
                "score": scores[i], # The scalar ColBERT score
                "original_index": i # Keep track of original order if needed
            })
        
        # Sort by score for consistency, although LIME/SHAP doesn't strictly need it
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        return reranked_results

    def get_underlying_model(self):
        """
        Returns the underlying RAGPretrainedModel instance for direct access,
        e.g., for tokenizer or model internals needed by DeepExplainer.
        """
        return self.rag_model

