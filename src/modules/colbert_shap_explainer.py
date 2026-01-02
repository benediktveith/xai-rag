import shap
import numpy as np
import torch
from transformers import AutoTokenizer
from typing import Any, Dict, List, Tuple
from langchain_core.documents import Document

from .colbert_rag_engine import ColbertRAGEngine

class ColbertShapExplainer:
    """
    Explains the ColBERT reranking score using shap.DeepExplainer.
    It identifies the contribution of each token in a document to its ColBERT
    similarity score with a given query.
    """

    def __init__(self, colbert_rag_engine: ColbertRAGEngine):
        """
        Initializes the ColbertShapExplainer.

        :param colbert_rag_engine: An initialized instance of ColbertRAGEngine.
        """
        self.colbert_engine = colbert_rag_engine
        self.model = self.colbert_engine.get_underlying_model().model # The underlying ColBERT model (PyTorch)
        self.tokenizer = self.colbert_engine.get_underlying_model().tokenizer # ColBERT's tokenizer
        
        # Ensure model is in evaluation mode and on the correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _get_colbert_score_function(self, model: torch.nn.Module):
        """
        Returns a function that takes query_ids and doc_ids (as PyTorch tensors)
        and computes the ColBERT similarity score. This is the function DeepExplainer
        will wrap.
        """
        def colbert_score_fn(query_ids_tensor: torch.Tensor, doc_ids_tensor: torch.Tensor) -> torch.Tensor:
            """
            Computes the ColBERT similarity score.
            Expects token IDs as input.
            """
            # Ensure inputs are on the correct device
            query_ids_tensor = query_ids_tensor.to(self.device)
            doc_ids_tensor = doc_ids_tensor.to(self.device)

            # Generate query and document embeddings
            query_embeddings = model.query_encoder(query_ids_tensor).to(self.device)
            doc_embeddings = model.doc_encoder(doc_ids_tensor).to(self.device)

            # Compute ColBERT score (max-sim over token embeddings)
            # Ensure query_embeddings and doc_embeddings are properly shaped for comparison
            # Assuming query_embeddings: (batch_size, query_len, embedding_dim)
            # Assuming doc_embeddings: (batch_size, doc_len, embedding_dim)
            
            # Simplified interaction for a single query-doc pair
            # Expand dimensions if batch_size is missing (e.g., if input is (seq_len, dim))
            if query_embeddings.dim() == 2:
                query_embeddings = query_embeddings.unsqueeze(0)
            if doc_embeddings.dim() == 2:
                doc_embeddings = doc_embeddings.unsqueeze(0)

            # Perform the ColBERT "max-sim" operation
            # Calculate dot products between query tokens and document tokens
            # (batch_size, query_len, embedding_dim) x (batch_size, embedding_dim, doc_len) -> (batch_size, query_len, doc_len)
            scores_per_token = torch.bmm(query_embeddings, doc_embeddings.transpose(1, 2))
            
            # Max pooling over document tokens (for each query token)
            max_scores_per_query_token = scores_per_token.max(dim=2).values
            
            # Sum up the max scores for query tokens to get final score
            final_score = max_scores_per_query_token.sum(dim=1)
            
            return final_score

        return colbert_score_fn

    def explain(self, query: str, document: Document) -> Dict[str, Any]:
        """
        Generates SHAP explanations for the tokens of a single document
        in the context of a given query using ColBERT's scoring.

        :param query: The query string.
        :param document: The LangChain Document object to explain.
        :return: A dictionary with the token-level explanations.
        """
        print(f"--- Explaining Document for Query: '{query}' ---")
        
        doc_text = document.page_content
        
        # Tokenize query and document
        query_encoded = self.tokenizer(query, return_tensors="pt", add_special_tokens=True)
        doc_encoded = self.tokenizer(doc_text, return_tensors="pt", add_special_tokens=True)
        
        # We need to detach query from the model as DeepExplainer expects
        # the model to only take one input at a time, or separate inputs for explanation.
        # For ColBERT, explanation is usually done on document tokens, with query as context.
        # DeepExplainer can explain w.r.t. specific inputs.
        # Here, we'll fix the query and explain document tokens.
        
        query_ids = query_encoded['input_ids'].to(self.device)
        doc_ids = doc_encoded['input_ids'].to(self.device)

        # The model function for DeepExplainer expects a single input (the document token IDs)
        # and will use the query_ids as a fixed context.
        def model_output_for_deepshap(doc_token_ids: torch.Tensor) -> torch.Tensor:
            return self._get_colbert_score_function(self.model)(query_ids, doc_token_ids)
        
        # DeepExplainer requires a background. For text, it's typically
        # a set of tokenized (masked) sequences.
        # Here, a single "masked" document.
        # For ColBERT, the special [MASK] token has specific embedding semantics.
        # We'll use a single background sample where document tokens are replaced by tokenizer.mask_token_id
        
        # Create a masked version of the document token IDs for the background
        masked_doc_ids = doc_ids.clone()
        if self.tokenizer.mask_token_id is not None:
             masked_doc_ids[:, 1:-1] = self.tokenizer.mask_token_id # Mask all tokens except CLS/SEP
        else:
             # Fallback if no mask_token_id, though ColBERT models usually have it
             masked_doc_ids[:, 1:-1] = self.tokenizer.pad_token_id # Use pad_token_id as a proxy

        # DeepExplainer expects the model and an input for background
        explainer = shap.DeepExplainer(model_output_for_deepshap, masked_doc_ids)

        # Calculate SHAP values for the actual document token IDs
        shap_values = explainer.shap_values(doc_ids)

        # shap_values will be a list of arrays for multi-output models.
        # For our single-output scalar score, it will be one array.
        # Typically, it's (batch_size, sequence_length, embedding_dim) or (batch_size, sequence_length)
        # We want the SHAP values for each token ID.
        
        # The output of DeepExplainer for token classification usually has shape
        # (1, sequence_length, num_classes) or (1, sequence_length).
        # Since our model_output_for_deepshap returns a scalar, shap_values will be (1, sequence_length).
        # We take the first (and only) sample's values.
        token_shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values

        # Convert token IDs back to tokens for display
        doc_tokens = self.tokenizer.convert_ids_to_tokens(doc_ids[0].cpu().numpy())
        
        # DeepExplainer often includes special tokens in its SHAP values.
        # We might want to remove SHAP values for [CLS] and [SEP] or other special tokens
        # if they are not relevant to the content explanation.
        # Here, we keep them for now, but be aware.
        
        explanation = {
            "query": query,
            "document_text": doc_text,
            "tokens": doc_tokens,
            "shap_values": token_shap_values,
        }

        return explanation
