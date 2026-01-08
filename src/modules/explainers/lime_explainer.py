import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

from ..rag.rag_engine import RAGEngine

class LimeExplainer:
    """
    Explains why a document was ranked highly by a RAGEngine using the LIME (Local
    Interpretable Model-agnostic Explanations) framework. It identifies the contribution
    of each token to the document's similarity score.
    """

    def __init__(self, rag_engine: RAGEngine):
        """
        Initializes the LimeExplainer.

        :param rag_engine: An initialized instance of the RAGEngine, used to access
                           the embedding model.
        """
        self.rag_engine = rag_engine
        self.embedding_model = rag_engine.embedding_model

    def explain(self, trace: Dict[str, Any], explained_doc_key: str = "retrieved_document_for_this_hop") -> Dict[str, Any]:
        """
        Generates LIME explanations for the top document at each hop in a trace.

        :param trace: The trace object from a multi-hop run.
        :param explained_doc_key: The key in the hop dictionary to find the document to explain.
        :return: A dictionary containing the token-level explanations for each hop.
        """
        explanations = {}
        # We now have two classes: "opposite_score" and "similarity_score"
        explainer = LimeTextExplainer(class_names=["opposite_score", "similarity_score"])

        for hop in trace["hops"]:
            hop_number = hop["hop_number"]
            query = hop["query_for_this_hop"]
            document = hop.get(explained_doc_key)
            
            if not document:
                continue
                
            print(f"--- LIME Explaining Hop {hop_number} ({explained_doc_key}) ---")
            
            doc_text = document.page_content
            
            prediction_fn = self._create_similarity_prediction_fn(query)

            true_probs = prediction_fn([doc_text])
            true_score = true_probs[0, 1]
            
            explanation = explainer.explain_instance(
                doc_text,
                prediction_fn,
                labels=(1,), # We only want the explanation for our "similarity_score" class (index 1)
                num_features=len(doc_text.split()),
                num_samples=1000
            )

            lime_intercept = explanation.intercept[1]
            # The weighted sum for class 1 (what you were calculating)
            lime_prediction = explanation.local_pred[0]
            
            # as_map will now return a map for each class; we want the one for our score (index 1)
            token_weights = explanation.as_map()[1]

            explanations[f"hop_{hop_number}"] = {
                "explanation_tuples": explanation.as_list(label=1), # Specify the label here too
                "token_weights": token_weights,
                "document_text": doc_text,
                "query": query,
                "true_score": true_score,
                "lime_intercept": lime_intercept,
                "lime_prediction": lime_prediction
            }

        return explanations

    def _embed_text(self, text: List[str]) -> torch.Tensor:
        """Helper to generate normalized embeddings for a list of strings."""
        embedding_vectors = self.embedding_model.embed_documents(text)
        embedding_tensor = torch.tensor(embedding_vectors)
        return F.normalize(embedding_tensor, p=2, dim=1)

    def _create_similarity_prediction_fn(self, query: str):
        """
        Creates a prediction function that LIME can use. This function takes a list
        of perturbed text strings and returns their cosine similarity to the query.
        """
        query_embedding = self._embed_text([query])

        def prediction_fn(perturbed_texts: List[str]) -> np.ndarray:
            """
            LIME expects a numpy array of shape (num_texts, num_classes).
            We'll return (num_texts, 2) where the columns are (1-score, score).
            """
            if not perturbed_texts:
                return np.array([]).reshape(-1, 2)
            
            doc_embeddings = self._embed_text(perturbed_texts)
            
            similarity_scores = torch.mm(query_embedding, doc_embeddings.transpose(0, 1)).flatten()
            
            # Stack the scores to create the (n_samples, 2) shape
            # Column 0: 1 - score
            # Column 1: score
            return np.vstack((1 - similarity_scores.numpy(), similarity_scores.numpy())).T

        return prediction_fn
    
    def plot_bar(self, explanation_data: Dict[str, Any], top_k: int = 10):
        """
        Plots the LIME weights.
        
        Args:
            explanation_data: The dictionary output from explain() for a single hop.
                              Must contain 'explanation_tuples', 'true_score', and 'lime_intercept'.
        """
        tuples = explanation_data["explanation_tuples"]
        true_score = explanation_data["true_score"]
        intercept = explanation_data["lime_intercept"]
        
        labels = [t[0] for t in tuples]
        values = [t[1] for t in tuples]

        sum_topk_to_last = np.sum(values[top_k::])
        len_labels = len(labels)

        # strip labels and values down to top_k
        labels = labels[:top_k]
        values = values[:top_k]
        labels.append(f"Sum of {len_labels} other features")
        values.append(sum_topk_to_last)

        # Calculate the sum for display purposes
        weight_sum = np.sum(values)
        lime_total = intercept + weight_sum

        colors = ['tab:red' if v < 0 else 'tab:blue' for v in values]

        plt.figure(figsize=(10, 6))
        
        # Plot bars
        # Reverse slices [::-1] so the most important features are at the top
        y_pos = np.arange(len(labels[:top_k])+1)
        plt.barh(y_pos, values, color=colors)
        plt.yticks(y_pos, labels)

        # Draw the centerline at 0
        plt.axvline(0, color='grey', lw=0.5)
        
        plt.xlabel("Impact on Similarity Score")
        plt.title(
            f"True Similarity: {true_score:.3f}\n"
            f"LIME Approx: {intercept:.3f} (Base) + {weight_sum:.3f} (Sum) = {lime_total:.3f}"
        )
        
        plt.figtext(0.5, -0.05, 
                    f"Note: The Base Intercept is {intercept:.3f}. These tokens shift the score from that baseline.", 
                    ha="center", fontsize=9, color="gray")

        plt.tight_layout()
        plt.show()