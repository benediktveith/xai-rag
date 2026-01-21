import shap
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Any, Dict, List, Literal, Optional
from langchain_core.documents import Document
from IPython.display import display, HTML
from sklearn.feature_extraction.text import TfidfVectorizer

class ShapExplainer:
    """
    Explains retrieval rankings using SHAP.
    Supports two background strategies:
    1. 'Zero': Replaces tokens with nothing (Deletion).
    2. 'Low-IDF': Replaces tokens with common 'background' words from the corpus.
    """

    # Options for explain function
    _DOC_TYPES = Literal["highest_ranked_document", "lowest_ranked_documents", "context_documents"]
    _BACKGROUND_TYPES = Literal["Low-IDF", "Zero"]

    def __init__(
        self,
        rag_engine,
        documents,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        idf_top_k: int = 150
    ):
        self.rag_engine = rag_engine
        self.embedding_model = rag_engine.embedding_model
        
        model_name = getattr(self.embedding_model, 'model_name', tokenizer_name)
        
        print(f"Loading tokenizer for: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # pre-calculate Low-IDF tokens if documents are provided
        self.low_idf_ids = []
        if documents:
            print(f"Calculating IDF for {len(documents)} documents...")
            self.low_idf_ids = self._get_low_idf_tokens(documents, top_k=idf_top_k)
        else:
            print("Note: No documents provided. 'Low-IDF' mode will need manual setup or will fail if selected.")


    def explain(
        self, 
        trace: Dict[str, Any], 
        explained_doc_key: _DOC_TYPES = "highest_ranked_document",
        background: _BACKGROUND_TYPES = "Zero"
    ) -> Dict[str, Any]:
        
        if background == "Low-IDF" and not self.low_idf_ids:
            raise ValueError("Background 'Low-IDF' selected but no background tokens calculated. Pass 'documents' to __init__.")

        explanations = {}

        for hop in trace["hops"]:
            hop_number = hop["hop_number"]
            query = hop["query_for_this_hop"]
            documents = hop.get(explained_doc_key)
            
            if not documents:
                continue

            if not isinstance(documents, list):
                documents = [documents]
                
            print(f"--- Shap Explaining Hop {hop_number} (Background: {background}) ---")

            explanations_hop = []

            for i,doc in enumerate(documents):
                print(f"- Explaining Doc {i+1}/{len(documents)} -")

                if isinstance(doc, tuple):
                    doc = doc[0]

                doc_text = doc.page_content
                
                # tokenize to IDs for safe reconstruction
                encoded = self.tokenizer(doc_text, add_special_tokens=False)
                token_ids = np.array(encoded['input_ids'])
                display_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                num_tokens = len(token_ids)
                
                query_embedding = self._embed_text([query])

                # create prediction function based on selected background strategy
                prediction_fn = self._create_similarity_prediction_fn(
                    query_embedding=query_embedding, 
                    original_token_ids=token_ids,
                    mode=background
                )

                # SHAP Setup
                # background is always zeros (representing "masked state")
                # the prediction_fn interprets what "masked" means (delete vs replace)
                # hence, the background is not always "0" in the model interpretation,
                # it depends on the background type selected (Zero or Low_IDF)!
                background_data = np.zeros((1, num_tokens))
                explainer = shap.KernelExplainer(prediction_fn, background_data)

                instance_to_explain = np.ones((1, num_tokens))
                
                nsamples = 150*num_tokens

                shap_values = explainer.shap_values(instance_to_explain, nsamples=nsamples, silent=True)[0]

                # aggregate Tokens back to full words
                words, shap_values = self.aggregate_shap_to_words(shap_values, display_tokens)

                explanations_hop.append ({
                    "shap_values": shap_values,
                    "base_value": explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                    "tokens": words, # display_tokens
                    "base_text": doc_text,
                    "query": query,
                    "score": prediction_fn(instance_to_explain)[0],
                    "background_mode": background
                })

            explanations[f"hop_{hop_number}"] = explanations_hop

        # Dict(List(Dict))
        return explanations
    
    def explain_v2(
        self, 
        trace: Dict[str, Any], 
        explained_doc_key: _DOC_TYPES = "highest_ranked_document"
    ) -> Dict[str, Any]:
        """
        Uses standard shap.Explainer (PartitionExplainer).
        Note: This forces the 'Zero' (Deletion) background strategy.
        """
        explanations = {}

        for hop in trace["hops"]:
            hop_number = hop["hop_number"]
            query = hop["query_for_this_hop"]
            document = hop.get(explained_doc_key)
            
            if not document:
                continue
                
            print(f"--- Explaining Hop {hop_number} (Standard Explainer) ---")
            
            doc_text = document.page_content
            query_embedding = self._embed_text([query])

            # 1. Define Predict Function
            # Returns 1D array of scores: [0.8, 0.5, 0.9]
            def predict(texts: List[str]) -> np.ndarray:
                valid_indices = [i for i, t in enumerate(texts) if t.strip()]
                valid_texts = [texts[i] for i in valid_indices]
                
                scores = np.zeros(len(texts))
                
                if valid_texts:
                    doc_embeddings = self._embed_text(valid_texts)
                    sims = torch.mm(doc_embeddings, query_embedding.transpose(0, 1))
                    np.put(scores, valid_indices, sims.flatten().numpy())
                
                return scores

            # 2. Create Masker
            masker = shap.maskers.Text(self.tokenizer)

            # 3. Create Explainer
            explainer = shap.Explainer(predict, masker)

            # 4. Run Explanation
            shap_values = explainer([doc_text])

            # 5. Extract results
            # shap_values is an Explanation object. 
            # .values gives the numbers.
            values = shap_values.values[0] 

            explanations[f"hop_{hop_number}"] = {
                "shap_values": values,
                # .data holds the tokens
                "tokens": shap_values.data[0], 
                "base_text": doc_text,
                "query": query,
                "score": np.sum(shap_values),   # since all shap values sum up to the outcome
                # .base_values is the starting score (expected value)
                "base_value": shap_values.base_values[0] 
            }

        return explanations
    

    def _get_low_idf_tokens(self, documents: List[Document], top_k: int) -> List[int]:
        """Calculates top_k tokens with lowest IDF (most common)."""

        if not documents:
            return []
            
        try:
            # compute IDF including stop words (to avoid shift in meaning)
            document_text = [doc.page_content for doc in documents]

            vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
            vectorizer.fit_transform(document_text)
            
            feature_names = vectorizer.get_feature_names_out()
            idf_scores = vectorizer.idf_
            
            # sort idf_score to get lowest ones
            sorted_items = sorted(zip(feature_names, idf_scores), key=lambda x: x[1])
            low_idf_words = [word for word, score in sorted_items[:top_k]]
            
            print(f"Background Vocabulary (Lowest IDF): {low_idf_words}")
            
            low_idf_ids = []
            for word in low_idf_words:
                encoded = self.tokenizer.encode(word, add_special_tokens=False)
                if encoded:
                    low_idf_ids.append(encoded[0])
            
            return low_idf_ids
            
        except Exception as e:
            print(f"Error calculating IDF: {e}")
            return []

    def _embed_text(self, texts: List[str]) -> torch.Tensor:
        """Embeds text and normalizes."""
        if not texts:
            return torch.tensor([])
            
        embeddings = self.rag_engine.embedding_model.embed_documents(texts)
        if isinstance(embeddings, list):
            tensor = torch.tensor(embeddings)
        else:
            tensor = torch.as_tensor(embeddings)
        return F.normalize(tensor, p=2, dim=1)

    def _create_similarity_prediction_fn(
        self, 
        query_embedding: torch.Tensor, 
        original_token_ids: np.ndarray,
        mode: _BACKGROUND_TYPES
    ):
        """
        Creates the SHAP prediction function.
        mode='Zero': Deletes masked tokens (reconstructs shorter string).
        mode='Low-IDF': Replaces masked tokens with random idf filler words.
        """
        def prediction_fn(mask_batch: np.ndarray) -> np.ndarray:
            reconstructed_texts = []
            rng = np.random.default_rng()

            for mask in mask_batch:
                # identify which tokens are KEPT (1) and which are MASKED (0)
                # then apply the selected Background image type
                kept_indices = np.where(mask == 1.0)[0]
                masked_indices = np.where(mask == 0.0)[0]

                if mode == "Zero":
                    # --- ZERO MODE: DELETION ---
                    # nnly keep the IDs where mask is 1
                    active_ids = original_token_ids[kept_indices].astype(int)
                    
                    if len(active_ids) == 0:
                        reconstructed_texts.append("")
                    else:
                        reconstructed_texts.append(self.tokenizer.decode(active_ids))

                elif mode == "Low-IDF":
                    # --- IDF MODE: REPLACEMENT ---
                    current_ids = original_token_ids.copy()
                    
                    # if masked tokens, replace them with random fillers
                    if len(masked_indices) > 0:
                        replacements = rng.choice(self.low_idf_ids, size=len(masked_indices))
                        current_ids[masked_indices] = replacements
                    
                    # decode the FULL length sequence
                    reconstructed_texts.append(self.tokenizer.decode(current_ids))

            # --- BATCH EMBEDDING ---
            if not reconstructed_texts:
                return np.array([])
            
            valid_indices = [i for i, t in enumerate(reconstructed_texts) if t.strip()]
            valid_texts = [reconstructed_texts[i] for i in valid_indices]
            
            scores = np.zeros(len(reconstructed_texts))
            
            if valid_texts:
                doc_embeddings = self._embed_text(valid_texts)
                sims = torch.mm(doc_embeddings, query_embedding.transpose(0, 1))
                sims_np = sims.flatten().numpy()
                
                # map back to original positions
                np.put(scores, valid_indices, sims_np)
            
            return scores

        return prediction_fn

    def plot_bar(self, explanations: Dict[str, Any]):
        """Helper to plot the shap bar plot."""

        predicted_score = round(explanations['base_value'] + np.sum(explanations["shap_values"]), 4)

        print(f"Base Score (Intercept): {explanations['base_value']:.4f} | Predicted Score: {predicted_score} | Actual Score: {explanations['score']:.4f}")
        
        return shap.plots.bar(shap.Explanation(
            values=explanations["shap_values"],
            base_values=explanations["base_value"], # approx baseline
            data=explanations["tokens"]
        ))

    def plot_text_heatmap(self, explanations: Dict[str, Any]):
        """
        Renders the document text as HTML, highlighting tokens based on SHAP Values.
        Red = Positive (Increases similarity)
        Blue = Negative (Dilutes/decreases similarity)
        """

        tokens = explanations["tokens"]
        weights = explanations["shap_values"]

        max_weight = max(abs(val) for val in weights)
        if max_weight == 0:
            max_weight = 1e-9

        def get_color(index):
            weight = weights[index]
            
            if weight == 0:
                return "transparent", "black"
            
            alpha = (abs(weight) / max_weight)
            
            if weight > 0:
                # RED for Positive
                return f"rgba(255, 0, 0, {alpha:.2f})", "black"
            else:
                # BLUE for Negative
                return f"rgba(0, 0, 255, {alpha:.2f})", "black"

        html_parts = []
        
        html_parts.append(f"<div style='border:1px solid #ddd; padding:15px; font-family:sans-serif; line-height:1.6;'>")
        html_parts.append(f"<h4>Similarity Heatmap (Query: <i>{explanations['query']}</i>)</h4>")
        html_parts.append(f"<p style='font-size:14px; color:gray'><b>HINT</b>: Some words are split by the Tokenizer. There were put back together and shap values aggregated.</p>")
        html_parts.append(f"<p style='font-size:12px; color:gray'><b>Red</b> = Increases Similarity | <b>Blue</b> = Decreases Similarity</p>")
        
        for index, token in enumerate(tokens):
            bg_color, text_color = get_color(index)
            html_parts.append(
                f"<span style='background-color:{bg_color}; color:{text_color}; padding: 0 2px; border-radius: 3px;'>{token}</span>"
            )
            
        html_parts.append("</div>")
        
        display(HTML(" ".join(html_parts)))

    def aggregate_shap_to_words(self, shap_values, tokens):
        """
        Aggregates BERT-style subword tokens (e.g., "##ing") into whole words
        and sums their SHAP values.
        """
        word_map = []
        word_scores = []
        
        current_word = ""
        current_score = 0.0
        
        for token, score in zip(tokens, shap_values):
            # for BERT (MiniLM), subwords start with ##
            
            if token.startswith("##"):
                current_word += token.replace("##", "")
                current_score += score
            else:
                if current_word != "":
                    word_map.append(current_word)
                    word_scores.append(current_score)
                
                current_word = token
                current_score = score
                
        if current_word:
            word_map.append(current_word)
            word_scores.append(current_score)
            
        return np.array(word_map), np.array(word_scores)