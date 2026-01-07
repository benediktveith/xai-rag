import torch
import torch.nn as nn
from transformers import AutoModel

class NativeColbertWrapper(nn.Module):
    def __init__(self, model_name="colbert-ir/colbertv2.0"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 128, bias=False) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        return torch.nn.functional.normalize(embeddings, p=2, dim=2)

    def score(self, query_emb, doc_emb):
        # query_emb: [batch, q_len, dim]
        # doc_emb:   [batch, d_len, dim]
        interaction = torch.bmm(query_emb, doc_emb.transpose(1, 2))
        return interaction.max(dim=2).values.sum(dim=1)