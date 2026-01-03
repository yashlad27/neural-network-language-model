#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Exported from: Problem8_PyTorch_20NG_M4_MPS_20251110_200929.ipynb
# Export time: 2025-11-10T20:12:38
# Notes:
# - Markdown cells are preserved as comments.
# - Device selection is generalized: MPS (Apple Silicon) -> CUDA -> CPU.


# === Cell 1 (markdown) ===
# # Problem 8: Word Vectors - PyTorch NN (20 Newsgroups)
# 
# **Dataset:** 20 Newsgroups (5-8 categories)  
# **Method:** Neural network trained on PPMI matrix (PyTorch + SGD)

# === Cell 2 (code) ===
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import re
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups

# === Cell 3 (code) ===
# ==== Device setup: Apple Silicon (M1–M4) GPU via MPS, CUDA if available, else CPU ====
import torch, platform

def get_device(prefer_mps=True):
    # Prefer Apple MPS on Apple Silicon
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    # Then CUDA if present
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Finally CPU
    return torch.device("cpu")

device = get_device()
print("Using device:", device)

# (Optional) On Apple Silicon, many ops only support float32; set default dtype accordingly.
torch.set_default_dtype(torch.float32)

# Helper movers
def to_device(x, non_blocking=False):
    try:
        return x.to(device, non_blocking=non_blocking)
    except Exception:
        return x

def move_batch(batch):
    # Move common batch structures to device
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(b) for b in batch)
    if isinstance(batch, dict):
        return {k: to_device(v) for k, v in batch.items()}
    return to_device(batch)

# If you previously hard-coded "cuda", use the 'device' variable or torch.set_default_device(device.type)
try:
    torch.set_default_device(device.type)
except Exception:
    pass

# === Cell 4 (markdown) ===
# ## 1. Preprocessing

# === Cell 5 (code) ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.split()

def build_vocabulary(words, vocab_size=10000, min_count=3):
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
                  'is', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'this', 'that', 'it', 'as',
                  'will', 'would', 'could', 'can', 'do', 'does', 'did', 'not', 'no', 'if', 'so'}
    
    word_counts = Counter(words)
    filtered = [(w, c) for w, c in word_counts.items()
                if c >= min_count and len(w) >= 3 and w not in stop_words]
    
    most_common = sorted(filtered, key=lambda x: x[1], reverse=True)[:vocab_size]
    word_to_id = {word: idx for idx, (word, _) in enumerate(most_common)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    corpus = [word_to_id[word] for word in words if word in word_to_id]
    
    return word_to_id, id_to_word, corpus

# === Cell 6 (markdown) ===
# ## 2. Co-occurrence Matrix

# === Cell 7 (code) ===
def build_cooccurrence_matrix(corpus, vocab_size, window_size=5, device='cpu'):
    cooccur = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=device)
    
    for i in tqdm(range(len(corpus)), desc="Building co-occurrence"):
        center = corpus[i]
        start = max(0, i - window_size)
        end = min(len(corpus), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:
                context = corpus[j]
                distance = abs(i - j)
                weight = 1.0 / distance
                cooccur[center, context] += weight
    
    return cooccur

# === Cell 8 (markdown) ===
# ## 3. PPMI Computation

# === Cell 9 (code) ===
def compute_ppmi(cooccur_matrix):
    print("Computing PPMI (vectorized)...")
    
    # Move to device if not already
    device = cooccur_matrix.device
    
    total = cooccur_matrix.sum()
    word_counts = cooccur_matrix.sum(dim=1, keepdim=True)  # (vocab_size, 1)
    context_counts = cooccur_matrix.sum(dim=0, keepdim=True)  # (1, vocab_size)
    
    # Vectorized PPMI computation
    # P(i,j) = cooccur[i,j] / total
    # P(i) = word_counts[i] / total
    # P(j) = context_counts[j] / total
    # PMI = log(P(i,j) / (P(i) * P(j)))
    
    p_ij = cooccur_matrix / total
    p_i_p_j = (word_counts @ context_counts) / (total * total)  # Outer product
    
    # Avoid log(0) by masking
    mask = cooccur_matrix > 0
    pmi = torch.zeros_like(cooccur_matrix)
    pmi[mask] = torch.log(p_ij[mask] / (p_i_p_j[mask] + 1e-10))
    
    # PPMI: max(0, PMI)
    ppmi = torch.clamp(pmi, min=0)
    
    # Zero out diagonal
    ppmi.fill_diagonal_(0)
    
    # Normalize rows
    row_sums = ppmi.sum(dim=1, keepdim=True)
    ppmi_normalized = ppmi / (row_sums + 1e-10)
    
    return ppmi_normalized

# === Cell 10 (markdown) ===
# ## 4. Neural Network

# === Cell 11 (code) ===
class WordVectorNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # No bias
        self.W1 = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.W2 = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Initialize with different std
        nn.init.normal_(self.W1.weight, mean=0, std=0.1)
        nn.init.normal_(self.W2.weight, mean=0, std=0.1)
    
    def forward(self, one_hot):
        hidden = self.W1(one_hot)
        output = self.W2(hidden)
        probs = torch.softmax(output, dim=-1)
        return probs
    
    def get_embeddings(self):
        return self.W1.weight.t().detach().cpu().numpy()

# === Cell 12 (markdown) ===
# ## 5. Training

# === Cell 13 (code) ===
def manual_ce_loss(probs, target):
    return -torch.sum(target * torch.log(probs + 1e-10))

def train_model(model, ppmi_matrix, optimizer, epochs=10, device='cpu'):
    model = model.to(device)
    ppmi_matrix = ppmi_matrix.to(device)
    vocab_size = ppmi_matrix.shape[0]
    
    for epoch in range(epochs):
        total_loss = 0
        indices = torch.randperm(vocab_size)
        
        for idx in tqdm(indices, desc=f"Epoch {epoch+1}/{epochs}"):
            one_hot = torch.zeros(vocab_size, device=device)
            one_hot[idx] = 1
            target = ppmi_matrix[idx]
            
            optimizer.zero_grad()
            probs = model(one_hot)
            loss = manual_ce_loss(probs, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / vocab_size
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

# === Cell 14 (markdown) ===
# ## 6. Evaluation

# === Cell 15 (code) ===
def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(v1, v2) / (norm1 * norm2)

def find_similar_words(word, word_to_id, id_to_word, embeddings, top_k=15):
    if word not in word_to_id:
        return None
    
    word_idx = word_to_id[word]
    word_emb = embeddings[word_idx]
    
    similarities = []
    for idx in range(len(embeddings)):
        if idx != word_idx:
            other_emb = embeddings[idx]
            sim = cosine_similarity(word_emb, other_emb)
            similarities.append((id_to_word[idx], sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def evaluate_model(embeddings, test_words, word_to_id, id_to_word):
    for word in test_words:
        similar = find_similar_words(word, word_to_id, id_to_word, embeddings)
        if similar:
            print(f"\n{word.upper()}:")
            for w, sim in similar:
                print(f"  {w:20s} {sim:.4f}")

# === Cell 16 (markdown) ===
# ## 7. Run Pipeline

# === Cell 17 (code) ===
# Use the device detected at the top (MPS, CUDA, or CPU)
print(f"Device: {device}")

print("\nLoading 20 Newsgroups...")
categories = [
    'comp.graphics', 'comp.sys.ibm.pc.hardware',
    'sci.crypt', 'sci.space',
    'soc.religion.christian',
    'talk.politics.guns', 'talk.politics.mideast',
    'alt.atheism'
]
newsgroups = fetch_20newsgroups(subset='train', categories=categories,
                                remove=('headers', 'footers', 'quotes'))

text = ' '.join(newsgroups.data)
words = preprocess_text(text)
print(f"Total words: {len(words):,}")

print("\nBuilding vocabulary...")
word_to_id, id_to_word, corpus = build_vocabulary(words, vocab_size=10000)
print(f"Vocab: {len(word_to_id):,}, Corpus: {len(corpus):,}")

print("\nBuilding co-occurrence matrix...")
cooccur = build_cooccurrence_matrix(corpus, len(word_to_id), window_size=5, device=device.type if isinstance(device, torch.device) else device)

print("\nComputing PPMI...")
ppmi = compute_ppmi(cooccur)

print("\nInitializing model...")
model = WordVectorNN(vocab_size=len(word_to_id), embedding_dim=200)
optimizer = optim.SGD(model.parameters(), lr=0.8)

print("\nTraining...")
train_model(model, ppmi, optimizer, epochs=10, device=device.type if isinstance(device, torch.device) else device)

print("\nExtracting embeddings...")
embeddings = model.get_embeddings()
print(f"Embeddings shape: {embeddings.shape}")

print("\n" + "="*60)
print("EVALUATION")
print("="*60)
test_words = ["china", "computer", "phone", "napoleon", "god", "catholic"]
evaluate_model(embeddings, test_words, word_to_id, id_to_word)
