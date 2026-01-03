#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Problem 8: Word Vectors - PyTorch NN (Text8 with MPS Support)
# Dataset: Text8 (750K words)
# Method: Neural network trained on PPMI matrix (PyTorch + SGD)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import re
from tqdm import tqdm
import time
from datetime import datetime

# ==============================================================================
# DEVICE SETUP: Apple Silicon (M1–M4) GPU via MPS, CUDA if available, else CPU
# ==============================================================================

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

# On Apple Silicon, many ops only support float32
torch.set_default_dtype(torch.float32)

# Helper movers
def to_device(x, non_blocking=False):
    try:
        return x.to(device, non_blocking=non_blocking)
    except Exception:
        return x

def move_batch(batch):
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(b) for b in batch)
    if isinstance(batch, dict):
        return {k: to_device(v) for k, v in batch.items()}
    return to_device(batch)

# Set default device
try:
    torch.set_default_device(device.type)
except Exception:
    pass

# ==============================================================================
# PREPROCESSING
# ==============================================================================

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

# ==============================================================================
# CO-OCCURRENCE MATRIX
# ==============================================================================

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

# ==============================================================================
# PPMI COMPUTATION
# ==============================================================================

def compute_ppmi(cooccur_matrix):
    print("Computing PPMI (vectorized)...")
    
    total = cooccur_matrix.sum()
    word_counts = cooccur_matrix.sum(dim=1, keepdim=True)
    context_counts = cooccur_matrix.sum(dim=0, keepdim=True)
    
    p_ij = cooccur_matrix / total
    p_i_p_j = (word_counts @ context_counts) / (total * total)
    
    mask = cooccur_matrix > 0
    pmi = torch.zeros_like(cooccur_matrix)
    pmi[mask] = torch.log(p_ij[mask] / (p_i_p_j[mask] + 1e-10))
    
    ppmi = torch.clamp(pmi, min=0)
    ppmi.fill_diagonal_(0)
    
    row_sums = ppmi.sum(dim=1, keepdim=True)
    ppmi_normalized = ppmi / (row_sums + 1e-10)
    
    return ppmi_normalized

# ==============================================================================
# NEURAL NETWORK
# ==============================================================================

class WordVectorNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.W1 = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.W2 = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        nn.init.normal_(self.W1.weight, mean=0, std=0.1)
        nn.init.normal_(self.W2.weight, mean=0, std=0.1)
    
    def forward(self, one_hot):
        hidden = self.W1(one_hot)
        output = self.W2(hidden)
        probs = torch.softmax(output, dim=-1)
        return probs
    
    def get_embeddings(self):
        return self.W1.weight.t().detach().cpu().numpy()

# ==============================================================================
# TRAINING
# ==============================================================================

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

# ==============================================================================
# EVALUATION
# ==============================================================================

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

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("WORD VECTORS - PYTORCH NEURAL NETWORK (TEXT8 750K)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Dataset: text8_750K.txt")
    print(f"Vocab Size: 10,000")
    print(f"Embedding Dim: 200")
    print(f"Window Size: 5")
    print(f"Learning Rate: 0.789")
    print(f"Epochs: 10")
    print("="*70)
    
    start_time = time.time()
    
    # Load dataset
    print("\nLoading Text8 (750K words)...")
    with open('/Users/yashlad/Development/CS6140/HW5/text8_750K.txt', 'r') as f:
        text = f.read()
    
    words = preprocess_text(text)
    print(f"Total words: {len(words):,}")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    word_to_id, id_to_word, corpus = build_vocabulary(words, vocab_size=10000)
    print(f"Vocab: {len(word_to_id):,}, Corpus: {len(corpus):,}")
    
    # Build co-occurrence matrix
    print("\nBuilding co-occurrence matrix...")
    cooccur = build_cooccurrence_matrix(corpus, len(word_to_id), window_size=5,
                                       device=device.type if isinstance(device, torch.device) else device)
    
    # Compute PPMI
    print("\nComputing PPMI...")
    ppmi = compute_ppmi(cooccur)
    
    # Initialize model
    print("\nInitializing model...")
    model = WordVectorNN(vocab_size=len(word_to_id), embedding_dim=200)
    optimizer = optim.SGD(model.parameters(), lr=0.789)
    
    # Train
    print("\nTraining...")
    train_model(model, ppmi, optimizer, epochs=10,
               device=device.type if isinstance(device, torch.device) else device)
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = model.get_embeddings()
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    test_words = ["china", "computer", "phone", "napoleon", "god", "catholic"]
    evaluate_model(embeddings, test_words, word_to_id, id_to_word)
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("="*70)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"text8_750k_results_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write("Text8 750K - Word Vector Neural Network Results\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Vocab Size: {len(word_to_id):,}\n")
        f.write(f"Corpus Size: {len(corpus):,}\n")
        f.write(f"Training Time: {total_time:.1f}s\n")
        f.write("="*70 + "\n\n")
        
        f.write("EVALUATION RESULTS\n")
        f.write("="*70 + "\n")
        for word in test_words:
            similar = find_similar_words(word, word_to_id, id_to_word, embeddings)
            if similar:
                f.write(f"\n{word.upper()}:\n")
                for w, sim in similar:
                    f.write(f"  {w:20s} {sim:.4f}\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
