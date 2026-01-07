import shap
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Any, Dict, List

class ShapExplainer:
    """
    Explains retrieval rankings by attributing cosine similarity scores to individual tokens.
    Uses 'True Deletion' to find negative contributors (tokens that dilute the vector).
    """

    def __init__(
        self,
        rag_engine,
        tokenizer_name: str = "bert-base-uncased",
    ):
        self.rag_engine = rag_engine
        self.embedding_model = rag_engine.embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def explain(self, trace: Dict[str, Any], explained_doc_key: str = "retrieved_document_for_this_hop") -> Dict[str, Any]:
        explanations = {}

        for hop in trace["hops"]:
            hop_number = hop["hop_number"]
            query = hop["query_for_this_hop"]
            document = hop.get(explained_doc_key)
            
            if not document:
                continue
                
            print(f"--- Explaining Hop {hop_number} ({explained_doc_key}) ---")
            
            doc_text = document.page_content
            # Tokenize
            tokens = self.tokenizer.tokenize(doc_text)
            num_tokens = len(tokens)
            
            # 1. Embed Query
            query_embedding = self._embed_text(query)

            # 2. Define Prediction Function (Uses Deletion)
            prediction_fn = self._create_similarity_prediction_fn(
                query_embedding, tokens
            )

            # 3. Setup SHAP
            # Background: zeros (representing deleted tokens)
            # TODO: He we can test different backround images!
                # We use all Zeros which is the "OOV" method discussed in the paper, 
                # however it could also be the average of all emebedding vectors in the vocabulary

            background = np.zeros((1, num_tokens))
            explainer = shap.KernelExplainer(prediction_fn, background)

            # Instance: ones (all tokens present)
            instance_to_explain = np.ones((1, num_tokens))
            
            # Adaptive sampling for speed
            nsamples = "auto" if num_tokens < 20 else 512

            shap_values = explainer.shap_values(instance_to_explain, nsamples=nsamples)[0]
            
            # 4. Post-Processing for Comparability
            # Mean Centering: Subtract the average contribution.
            # Positive = helps more than the average token.
            # Negative = drags the document score down (dilutes the embedding).
            mean_shap = np.mean(shap_values)
            centered_shap_values = shap_values - mean_shap

            explanations[f"hop_{hop_number}"] = {
                "shap_values": shap_values,            # Raw SHAP (Sum = Cosine Score)
                "centered_shap_values": centered_shap_values, # Relativized (Force negatives)
                "tokens": tokens,
                "document_text": doc_text,
                "query": query,
                "score": prediction_fn(instance_to_explain)[0] # The final cosine score
            }

        return explanations

    def _embed_text(self, text: str) -> torch.Tensor:
        """Embeds text handling empty strings gracefully."""
        if not text or not text.strip():
            # Return a zero vector for empty text
            # This ensures 'nothing' has 0 similarity to 'something'
            # We need the output dim of the model to make the zero vector
            # We'll do a dummy embed to get dim if needed, or assume standard dims
            dummy = self.embedding_model.embed_documents(["a"])[0]
            return torch.zeros(1, len(dummy))
            
        embedding_vector = self.embedding_model.embed_documents([text])[0]
        embedding_tensor = torch.tensor(embedding_vector).unsqueeze(0)
        return F.normalize(embedding_tensor, p=2, dim=1)

    def _create_similarity_prediction_fn(
        self, query_embedding: torch.Tensor, original_tokens: List[str]
    ):
        def prediction_fn(x: np.ndarray) -> np.ndarray:
            texts_to_embed = []
            
            # Optimize: Reconstruct batch of texts
            for sample in x:
                # True Deletion Logic:
                # Only include tokens where mask is 1. 
                # Where mask is 0, the token is gone (string becomes shorter).
                active_tokens = [t for t, mask in zip(original_tokens, sample) if mask == 1]
                
                # Handle the "Empty String" baseline case
                if not active_tokens:
                    texts_to_embed.append("") 
                else:
                    texts_to_embed.append(self.tokenizer.convert_tokens_to_string(active_tokens))

            # Batch embedding
            if not texts_to_embed:
                return np.array([])

            # Get embeddings for the whole batch
            # Note: We need to handle the empty string case manually if the model crashes on it
            # But most langchain wrappers handle empty strings by returning a vector or erroring.
            # We map empty strings to a placeholder or catch it.
            try:
                doc_embeddings_list = self.embedding_model.embed_documents(texts_to_embed)
            except Exception:
                # Fallback for models that crash on empty string
                # We embed a dummy and zero it out, or embed a space
                sanitized = [t if t.strip() else " " for t in texts_to_embed]
                doc_embeddings_list = self.embedding_model.embed_documents(sanitized)
            
            doc_embeddings = torch.tensor(doc_embeddings_list)
            
            # Normalize doc embeddings
            # Note: Avoid div by zero for the empty string vector (if it's zero)
            doc_norms = doc_embeddings.norm(p=2, dim=1, keepdim=True)
            doc_embeddings_normalized = doc_embeddings.div(doc_norms + 1e-10) # epsilon for stability
            
            # Cosine similarity: (Batch, Dim) @ (Dim, 1) -> (Batch, 1)
            scores = torch.mm(doc_embeddings_normalized, query_embedding.transpose(0, 1))
            
            return scores.flatten().numpy()

        return prediction_fn