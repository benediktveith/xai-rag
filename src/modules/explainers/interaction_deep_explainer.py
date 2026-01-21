import torch
import torch.nn as nn
import numpy as np
import shap

class InteractionDeepExplainer:
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.model = rag_engine.model
        self.embedding_layer = rag_engine.embedding
        self.device = rag_engine.device
        self.vocab = rag_engine.vocab 

    def explain(self, query, document):
        q_ids = self.model.text_to_ids(query) if hasattr(self.model, 'text_to_ids') else self.rag_engine.text_to_ids(query)
        d_ids = self.model.text_to_ids(document) if hasattr(self.model, 'text_to_ids') else self.rag_engine.text_to_ids(document)
        
        if hasattr(self, 'model') and not hasattr(self.model, 'text_to_ids'):
             q_ids = self.rag_engine.text_to_ids(query)
             d_ids = self.rag_engine.text_to_ids(document)

        d_embeddings = self.embedding_layer(d_ids).detach() 

        class Proxy(nn.Module):
            def __init__(self, original_model, fixed_q_ids):
                super().__init__()
                self.model = original_model
                self.q_ids = fixed_q_ids
                
            def forward(self, doc_embs):
                current_batch_size = doc_embs.shape[0]
                q_emb = self.model.embedding(self.q_ids)
                q_emb_expanded = q_emb.expand(current_batch_size, -1, -1)
                
                model_type = str(type(self.model))
                if "MatchPyramidWrapper" in model_type:
                    interaction = torch.bmm(q_emb_expanded, doc_embs.transpose(1, 2))
                    x = interaction.unsqueeze(1)
                    x = self.model.pool(self.model.relu(self.model.conv1(x)))
                    return self.model.linear(x.view(x.size(0), -1))
                elif "KNRMWrapper" in model_type:
                    interaction = torch.bmm(q_emb_expanded, doc_embs.transpose(1, 2))
                    features = self.model.kernel_pooling(interaction)
                    doc_features = torch.sum(features, dim=1)
                    return self.model.dense(doc_features)
                return None

        # run DeepExplainer
        proxy = Proxy(self.model, q_ids)
        background = torch.zeros_like(d_embeddings)
        explainer = shap.DeepExplainer(proxy, background)
        shap_values_list = explainer.shap_values(d_embeddings, check_additivity=False)
        
        raw_scores = np.sum(shap_values_list[0], axis=2)[0]
        
        id_to_word = {v: k for k, v in self.vocab.items()}

        d_id_list = d_ids[0].cpu().numpy()
        
        aligned_words = []
        for token_id in d_id_list:
            word = id_to_word.get(token_id, "[UNK]")
            aligned_words.append(word)
            
        final_scores = []
        final_words = []
        
        for word, score, token_id in zip(aligned_words, raw_scores, d_id_list):
            if token_id != 0:
                final_words.append(word)
                final_scores.append(score)
                
        return np.array(final_scores), final_words