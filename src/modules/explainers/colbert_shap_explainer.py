import shap
import torch

class ColbertExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def explain(self, query, document_text):
        # Tokenize query and document_text
        q_enc = self.tokenizer.encode(query, return_tensors="pt")
        d_enc = self.tokenizer.encode(document_text, return_tensors="pt")

        # Define the scoring function for SHAP
        def score_func(d_ids):
            with torch.no_grad():
                # ColBERT math: MaxSim interaction
                Q = self.model.query(q_enc)
                D = self.model.doc(torch.as_tensor(d_ids).long())
                # Resulting score for each doc in batch
                return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1).cpu().numpy()

        # Create a "Background" (a masked version of the document)
        mask_id = self.tokenizer.mask_token_id
        background = d_enc.clone()
        background[0, 1:-1] = mask_id # Mask everything except CLS/SEP tokens

        # Initialize DeepExplainer
        explainer = shap.DeepExplainer(score_func, background)
        shap_values = explainer.shap_values(d_enc)
        
        return shap_values, self.tokenizer.convert_ids_to_tokens(d_enc[0])
