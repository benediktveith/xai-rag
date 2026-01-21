import torch
import torch.nn as nn
import numpy as np
import re
from typing import List

# FIX RANDOMNESS, NOTE: These are still untrained weights!
torch.manual_seed(150)
np.random.seed(150)

class BaseInteractionEngine:
    def __init__(self, embedding_path="../data/glove.6B.50d.txt", embedding_dim=50):
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load GloVe dictionary temporarily
        print("Loading GloVe embeddings... (this might take a moment)")
        self.glove_vectors = {}
        try:
            with open(embedding_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    self.glove_vectors[word] = vector
            print(f"Loaded {len(self.glove_vectors)} GloVe vectors.")
        except FileNotFoundError:
            print("GloVe file not found! Using random embeddings (Results will be bad).")

        # Initialize Embedding layer (Placeholder)
        self.embedding = nn.Embedding(2, embedding_dim, padding_idx=0)
        self.to(self.device)

    def to(self, device):
        self.embedding.to(device)
        self.device = device

    def _clean_tokenize(self, text):
        """
        Splits text, removes punctuation, and forces lowercase.
        "The patient, John." -> ["the", "patient", "john"]
        """
        # Regex: Keep only alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.lower().split()

    def index_documents(self, documents: List[str]):
        self.documents = documents
        
        print("Building vocabulary...")
        for doc in documents:
            for word in self._clean_tokenize(doc):
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        
        vocab_size = len(self.vocab)
        print(f"Vocab size: {vocab_size}")

        embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, self.embedding_dim))
        embedding_matrix[0] = np.zeros(self.embedding_dim) # PAD

        found = 0
        missing_samples = []
        
        for word, idx in self.vocab.items():
            # Check exact match first
            if word in self.glove_vectors:
                embedding_matrix[idx] = self.glove_vectors[word]
                found += 1
            else:
                if len(missing_samples) < 5: missing_samples.append(word)

        print(f"Found GloVe vectors for {found}/{vocab_size} words.")
        if found == 0:
            print("CRITICAL WARNING: Still found 0 words. Check your GloVe file format.")
            print(f"Sample words from your vocab: {missing_samples}")
            print(f"Sample keys from GloVe: {list(self.glove_vectors.keys())[:5]}")

        # Load into PyTorch
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), 
            freeze=False, 
            padding_idx=0
        )
        self.embedding.to(self.device)
        
        if hasattr(self, 'model'):
            self.model.embedding = self.embedding

    def text_to_ids(self, text: str, max_len=300):
        clean_words = self._clean_tokenize(text)
        ids = [self.vocab.get(w, 1) for w in clean_words]
        
        if len(ids) > max_len: 
            ids = ids[:max_len]
        else: 
            ids += [0] * (max_len - len(ids))
            
        return torch.tensor([ids], dtype=torch.long).to(self.device)