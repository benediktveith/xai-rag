import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Literal
import re
from IPython.display import display, HTML


class LimeExplainer:

    _DOC_TYPES = Literal["highest_ranked_document", "lowest_ranked_documents", "context_documents"]

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.embedding_model = rag_engine.embedding_model

    def explain(self, trace: Dict[str, Any], explained_doc_key: _DOC_TYPES = "highest_ranked_document") -> Dict[str, Any]:
        explanations = {}
        
        explainer = LimeTextExplainer(class_names=["irrelevant", "relevant"])

        for hop in trace["hops"]:
            hop_number = hop["hop_number"]
            query = hop["query_for_this_hop"]
            documents = hop.get(explained_doc_key)
            
            if not documents:
                continue

            print(f"--- LIME Explaining Hop {hop_number} ---")

            explanations_hop = []

            for i, doc in enumerate(documents):
                print(f"- Explaining Doc {i+1}/{len(documents)} -")

                if not isinstance(documents, list):
                    documents = [documents]
                    
                
                doc_text = doc.page_content
                
                # setup the prediction function wrapper
                prediction_fn = self._create_similarity_prediction_fn(query)
                
                true_probs = prediction_fn([doc_text])
                true_score = true_probs[0, 1] 

                # run LIME
                # labels=(1,) tells LIME we only care about explaining Class 1 ("relevant")
                explanation = explainer.explain_instance(
                    doc_text,
                    prediction_fn,
                    labels=(1,), 
                    num_features=10000, 
                    num_samples=3000 
                )

                # extract data for Class 1
                # LIME returns a map of {label: list_of_tuples}
                explanation_tuples = explanation.as_list(label=1)
                
                # get the intercept (baseline probability) for Class 1
                # explanation.intercept is a dict {label: value}
                lime_intercept = explanation.intercept[1]
                lime_prediction = explanation.local_pred[0] # Usually a float for the specific label

                explanations_hop.append({
                    "explanation_tuples": explanation_tuples,
                    "document_text": doc_text,
                    "query": query,
                    "true_score": true_score,
                    "lime_intercept": lime_intercept,
                    "lime_prediction": lime_prediction
                })

            explanations[f"hop_{hop_number}"] = explanations_hop

        return explanations


    def _embed_text(self, text: List[str]) -> torch.Tensor:
        """Helper to generate normalized embeddings for a list of strings."""
        embeddings = self.rag_engine.embedding_model.embed_documents(text)
        
        if isinstance(embeddings, list):
            tensor = torch.tensor(embeddings)
        else:
            tensor = torch.as_tensor(embeddings)
            
        return F.normalize(tensor, p=2, dim=1)


    def _create_similarity_prediction_fn(self, query: str):
        query_embedding = self._embed_text([query])

        def prediction_fn(perturbed_texts: List[str]) -> np.ndarray:
            if not perturbed_texts:
                return np.array([]).reshape(0, 2)
            
            cleaned_texts = [t if t.strip() else " " for t in perturbed_texts]
            doc_embeddings = self._embed_text(cleaned_texts)
            
            # calculate Cosine Similarity
            scores = torch.mm(query_embedding, doc_embeddings.transpose(0, 1)).flatten()
            scores_np = scores.numpy()
            
            # force 0.0 for empty strings
            for i, text in enumerate(perturbed_texts):
                if not text.strip():
                    scores_np[i] = 0.0
            
            # clip to ensure valid probabilities [0, 1]
            scores_np = np.clip(scores_np, 0.0, 1.0)
            
            # Column 0: Irrelevant (1 - score)
            # Column 1: Relevant (score)
            return np.vstack((1 - scores_np, scores_np)).T

        return prediction_fn
    

    def plot_bar(self, explanation_data: Dict[str, Any], top_k: int = 10):
        """
        Plots the LIME weights for regression (Impact on Similarity Score).
        """
        tuples = explanation_data["explanation_tuples"]
        true_score = explanation_data["true_score"]
        intercept = explanation_data["lime_intercept"]
        
        labels = [t[0] for t in tuples]
        values = [t[1] for t in tuples]

        if len(values) > top_k:
            sum_topk_to_last = np.sum(values[top_k:])
            labels = labels[:top_k]
            values = values[:top_k]
            labels.append(f"Sum of others")
            values.append(sum_topk_to_last)

        weight_sum = np.sum(values) # sum of displayed bars (including "others")
        lime_total = intercept + np.sum([t[1] for t in tuples]) # total LIME prediction

        colors = ['tab:red' if v > 0 else 'tab:blue' for v in values]

        plt.figure(figsize=(10, 6))
        
        y_pos = np.arange(len(labels))
        plt.barh(y_pos, values[::-1], color=colors[::-1])
        plt.yticks(y_pos, labels[::-1])

        plt.axvline(0, color='black', lw=0.8, linestyle='--')
        plt.xlabel("Contribution to Cosine Similarity")
       
        plt.title(
            f"Why this document? (LIME Regression)\n"
            f"Base Score (Intercept): {intercept:.3f} | Predicted Score: {lime_total:.3f} | Actual Score: {true_score:.3f}"
        )
        
        plt.tight_layout()
        plt.show()


    def plot_text_heatmap(self, explanation_data: Dict[str, Any]):
        """
        Renders the document text as HTML, highlighting tokens based on LIME weights.
        Red = Positive (Increases similarity)
        Blue = Negative (Dilutes/decreases similarity)
        """
        text = explanation_data["document_text"]
        weights = dict(explanation_data["explanation_tuples"])
        
        # normalize weights for color intensity
        if not weights:
            print("No weights to plot.")
            return

        max_weight = max(abs(val) for val in weights.values())
        if max_weight == 0:
            max_weight = 1e-9

        def get_color(word_token):
            clean_word = word_token.strip(".,!?;:()[]\"'")
            weight = weights.get(clean_word, 0)
            
            if weight == 0:
                return "transparent", "black"
            
            alpha = (abs(weight) / max_weight)
            
            if weight > 0:
                return f"rgba(255, 0, 0, {alpha:.2f})", "black"
            else:
                return f"rgba(0, 0, 255, {alpha:.2f})", "black"

        # Split text loosely to wrap spans
        words = text.split()
        html_parts = []
        
        html_parts.append(f"<div style='border:1px solid #ddd; padding:15px; font-family:sans-serif; line-height:1.6;'>")
        html_parts.append(f"<h4>Similarity Heatmap (Query: <i>{explanation_data['query']}</i>)</h4>")
        html_parts.append(f"<p style='font-size:12px; color:gray'><b>Red</b> = Increases Similarity | <b>Blue</b> = Decreases Similarity</p>")
        
        for word in words:
            bg_color, text_color = get_color(word)
            html_parts.append(
                f"<span style='background-color:{bg_color}; color:{text_color}; padding: 0 2px; border-radius: 3px;'>{word}</span>"
            )
            
        html_parts.append("</div>")
        
        display(HTML(" ".join(html_parts)))