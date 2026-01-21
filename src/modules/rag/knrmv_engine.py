import torch
from torch import nn
import numpy as np
from .base_interaction_engine import BaseInteractionEngine

class KNRMWrapper(nn.Module):
    def __init__(self, embedding_layer, mu=None, sigma=None):
        super().__init__()
        self.embedding = embedding_layer
        
        # 1.0 = Exact match, 0.9 = Synonym, 0.7 = Related, etc.
        if mu is None:
            self.mu = torch.tensor([1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1], device="cuda")
            self.sigma = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], device="cuda")
        else:
            self.mu = mu
            self.sigma = sigma
            
        self.dense = nn.Linear(len(self.mu), 1)

    def kernel_pooling(self, interaction_matrix):
        """
        Applies Gaussian kernels to the interaction matrix.
        Input: [Batch, Q, D]
        Output: [Batch, Q, K] (K = number of kernels)
        """
        # Broadcast kernels: [1, 1, 1, K]
        mu = self.mu.view(1, 1, 1, -1)
        sigma = self.sigma.view(1, 1, 1, -1)
        
        # Interaction: [Batch, Q, D, 1]
        matrix = interaction_matrix.unsqueeze(-1)
        
        # Gaussian RBF: exp(- (x - mu)^2 / (2*sigma^2))
        raw_kernel_scores = torch.exp(- (matrix - mu)**2 / (2 * sigma**2))
        
        # Sum over Document dimension (Soft Word Count)
        # "How many words in Doc match Query_word_i at level mu_k?"
        pooling = torch.sum(raw_kernel_scores, dim=2) # [Batch, Q, K]
        
        # Log normalization (standard in K-NRM)
        return torch.log(torch.clamp(pooling, min=1e-10))

    def forward(self, q_ids, d_ids):
        q_emb = self.embedding(q_ids)
        d_emb = self.embedding(d_ids)
        
        # 1. Interaction Matrix
        interaction = torch.bmm(q_emb, d_emb.transpose(1, 2))
        
        # 2. Kernel Pooling (The "Histogram" replacement)
        # Features: [Batch, Q, K]
        features = self.kernel_pooling(interaction)
        
        # 3. Sum over Query words (Aggregate evidence)
        # [Batch, K]
        doc_features = torch.sum(features, dim=1)
        
        # 4. Learning to Rank (Linear combination of kernels)
        score = self.dense(doc_features)
        return score

class KNRMRAGEngine(BaseInteractionEngine):
    def __init__(self):
        super().__init__()
        # Ensure mu/sigma are on correct device
        mu = torch.tensor([1.0, 0.9, 0.7], device=self.device)
        sigma = torch.tensor([0.001, 0.1, 0.1], device=self.device)
        self.model = KNRMWrapper(self.embedding, mu, sigma).to(self.device)

    def search(self, query: str):
        q_ids = self.text_to_ids(query)
        scores = []
        with torch.no_grad():
            for doc in self.documents:
                d_ids = self.text_to_ids(doc)
                score = self.model(q_ids, d_ids)
                scores.append(score.item())
        best_idx = np.argmax(scores)
        return self.documents[best_idx], scores[best_idx]