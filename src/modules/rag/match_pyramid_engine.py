import torch
from torch import nn
import numpy as np
from .base_interaction_engine import BaseInteractionEngine


class MatchPyramidWrapper(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding = embedding_layer
        # 1 channel (grayscale), 8 filters, 3x3 kernel (looking for tri-grams)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool2d((4, 4)) # Resize everything to 4x4
        self.linear = nn.Linear(8 * 4 * 4, 1)

    def forward(self, q_ids, d_ids):
        # 1. Get Embeddings
        q_emb = self.embedding(q_ids) # [Batch, Q_len, Dim]
        d_emb = self.embedding(d_ids) # [Batch, D_len, Dim]

        # 2. Build Interaction Matrix (The "Image")
        # [Batch, Q, Dim] @ [Batch, Dim, D] -> [Batch, Q, D]
        interaction = torch.bmm(q_emb, d_emb.transpose(1, 2))
        
        # 3. Add Channel Dimension for CNN: [Batch, 1, Q, D]
        x = interaction.unsqueeze(1)
        
        # 4. CNN Layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 5. Flatten and Score
        x = x.view(x.size(0), -1)
        score = self.linear(x)
        return score

class MatchPyramidRAGEngine(BaseInteractionEngine):
    def __init__(self):
        super().__init__()
        self.model = MatchPyramidWrapper(self.embedding).to(self.device)

    def search(self, query: str):
        q_ids = self.text_to_ids(query)
        scores = []
        with torch.no_grad():
            for doc in self.documents:
                d_ids = self.text_to_ids(doc)
                score = self.model(q_ids, d_ids)
                scores.append(score.item())
        
        # Return top result
        best_idx = np.argmax(scores)
        return self.documents[best_idx], scores[best_idx]