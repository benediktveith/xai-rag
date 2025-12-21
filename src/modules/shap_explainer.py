import shap
import numpy as np
import torch
from transformers import AutoTokenizer
from typing import Any, Dict, List
from langchain_core.documents import Document
import torch.nn.functional as F

from .rag_engine import RAGEngine

class ShapExplainer:
    """
    Explains why a document was ranked highly by a RAGEngine by showing the contribution
    of each token within that document to its similarity score with the query.
    """

    def __init__(
        self,
        rag_engine: RAGEngine,
        tokenizer_name: str = "bert-base-uncased",
    ):
        """
        Initializes the ShapExplainer.

        :param rag_engine: An initialized instance of the RAGEngine.
        :param tokenizer_name: The name of the tokenizer to use for token-level explanations.
        """
        self.rag_engine = rag_engine
        # The embedding model from the RAGEngine is our "model" to explain
        self.embedding_model = rag_engine.embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token = self.tokenizer.mask_token if self.tokenizer.mask_token else "[MASK]"

    def explain(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates SHAP explanations for the top document at each hop in a trace.

        :param trace: The trace object, which must contain 'hops' with 'query_for_this_hop'
                      and 'retrieved_document_for_this_hop'.
        :return: A dictionary with the token-level explanations for the top document at each hop.
        """
        explanations = {}

        for i, hop in enumerate(trace["hops"]):
            hop_number = hop["hop_number"]
            query = hop["query_for_this_hop"]
            top_document = hop["retrieved_document_for_this_hop"]
            
            print(f"--- Explaining Tokens in Top Document for Hop {hop_number} ---")
            
            doc_text = top_document.page_content
            tokens = self.tokenizer.tokenize(doc_text)
            num_tokens = len(tokens)
            
            # Get the embedding for the original query once
            query_embedding = self._embed_text(query)

            # Create the prediction function for this specific hop and document
            prediction_fn = self._create_similarity_prediction_fn(
                query_embedding, tokens
            )

            # Background is a version of the document with all tokens masked
            # TODO: He we can test different backround images!
            background = np.zeros((1, num_tokens))
            explainer = shap.KernelExplainer(prediction_fn, background)

            # The instance to explain is the original document with all tokens present
            instance_to_explain = np.ones((1, num_tokens))
            
            # Adjust nsamples based on token count to manage computation time
            nsamples = 2 ** num_tokens if num_tokens < 8 else 256

            shap_values = explainer.shap_values(instance_to_explain, nsamples=nsamples)

            explanations[f"hop_{hop_number}"] = {
                "shap_values": shap_values[0],
                "tokens": tokens,
                "document_text": doc_text,
                "query": query
            }

        return explanations

    def _embed_text(self, text: str) -> torch.Tensor:
        """Helper to generate a normalized embedding for a single string."""
        # The embedding model client expects a list of texts
        embedding_vector = self.embedding_model.embed_documents([text])[0]
        # Convert to tensor and normalize for cosine similarity
        embedding_tensor = torch.tensor(embedding_vector).unsqueeze(0)
        return F.normalize(embedding_tensor, p=2, dim=1)

    def _create_similarity_prediction_fn(
        self, query_embedding: torch.Tensor, original_tokens: List[str]
    ):
        """
        Creates a prediction function that calculates the cosine similarity between
        a query and a perturbed document.
        """
        
        def prediction_fn(x: np.ndarray) -> np.ndarray:
            """
            This function takes a binary mask `x` representing subsets of tokens
            and returns the cosine similarity score with the query.
            """
            scores = []
            texts_to_embed = []
            for sample in x:  # For each perturbation (mask)
                # Reconstruct the document text based on the token mask
                perturbed_tokens = [token if included else self.mask_token for token, included in zip(original_tokens, sample)]
                perturbed_text = self.tokenizer.convert_tokens_to_string(perturbed_tokens)
                texts_to_embed.append(perturbed_text)

            # Embed all perturbed texts in a single batch
            if not texts_to_embed:
                return np.array([])
                
            doc_embeddings_list = self.embedding_model.embed_documents(texts_to_embed)
            doc_embeddings = torch.tensor(doc_embeddings_list)
            doc_embeddings_normalized = F.normalize(doc_embeddings, p=2, dim=1)
            
            # Calculate cosine similarity for the whole batch
            # (query_embedding is (1, dim), doc_embeddings_normalized is (batch, dim))
            similarity_scores = torch.mm(query_embedding, doc_embeddings_normalized.transpose(0, 1))
            
            return similarity_scores.flatten().numpy()

        return prediction_fn
