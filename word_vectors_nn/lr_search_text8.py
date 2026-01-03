#!/usr/bin/env python3
"""
Learning Rate Search for Text8 Dataset
Tests learning rates from 0.5 to 0.8 to find the optimal value
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import re
from tqdm import tqdm
import time
from datetime import datetime

# Device setup
def get_device(prefer_mps=True):
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")
torch.set_default_dtype(torch.float32)

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
# CO-OCCURRENCE & PPMI
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
    
    epoch_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        indices = torch.randperm(vocab_size)
        
        for idx in tqdm(indices, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
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
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    return epoch_losses

# ==============================================================================
# EVALUATION
# ==============================================================================

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(v1, v2) / (norm1 * norm2)

def find_similar_words(word, word_to_id, id_to_word, embeddings, top_k=10):
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

def evaluate_embeddings(embeddings, test_words, word_to_id, id_to_word):
    """Evaluate embedding quality with test words"""
    results = []
    for word in test_words:
        similar = find_similar_words(word, word_to_id, id_to_word, embeddings, top_k=5)
        if similar:
            results.append(f"{word}: {', '.join([w for w, _ in similar])}")
    return "; ".join(results)

# ==============================================================================
# LEARNING RATE SEARCH
# ==============================================================================

def test_learning_rate(lr, ppmi, word_to_id, id_to_word, epochs=10, device='cpu'):
    """Test a single learning rate and return results"""
    print(f"\n{'='*70}")
    print(f"Testing Learning Rate: {lr}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Initialize model
    model = WordVectorNN(vocab_size=len(word_to_id), embedding_dim=200)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Train
    epoch_losses = train_model(model, ppmi, optimizer, epochs=epochs, device=device)
    
    # Extract embeddings
    embeddings = model.get_embeddings()
    
    # Evaluate
    test_words = ["china", "computer", "phone", "king", "queen"]
    eval_summary = evaluate_embeddings(embeddings, test_words, word_to_id, id_to_word)
    
    training_time = time.time() - start_time
    
    result = {
        'lr': lr,
        'final_loss': epoch_losses[-1],
        'min_loss': min(epoch_losses),
        'epoch_losses': epoch_losses,
        'training_time': training_time,
        'eval_summary': eval_summary
    }
    
    print(f"\nResults for LR={lr}:")
    print(f"  Final Loss: {result['final_loss']:.4f}")
    print(f"  Min Loss: {result['min_loss']:.4f}")
    print(f"  Training Time: {training_time:.1f}s")
    print(f"  Eval: {eval_summary}")
    
    return result

def main():
    print("\n" + "="*70)
    print("LEARNING RATE SEARCH FOR TEXT8 DATASET")
    print("="*70)
    print(f"Device: {device}")
    print(f"Vocab Size: 10,000")
    print(f"Epochs: 10")
    print(f"LR Range: 0.5 - 0.8")
    print("="*70)
    
    # Load dataset
    print("\nLoading Text8 dataset...")
    with open('/Users/yashlad/Development/CS6140/HW5/text8_1M.txt', 'r') as f:
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
    
    # Test different learning rates
    learning_rates = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    results = []
    
    for lr in learning_rates:
        result = test_learning_rate(lr, ppmi, word_to_id, id_to_word, epochs=10, 
                                   device=device.type if isinstance(device, torch.device) else device)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("LEARNING RATE SEARCH SUMMARY")
    print("="*70)
    print(f"{'LR':<8} {'Final Loss':<12} {'Min Loss':<12} {'Time (s)':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['lr']:<8.2f} {r['final_loss']:<12.4f} {r['min_loss']:<12.4f} {r['training_time']:<10.1f}")
    
    # Find best LR
    best_result = min(results, key=lambda x: x['final_loss'])
    print("\n" + "="*70)
    print(f"BEST LEARNING RATE: {best_result['lr']}")
    print(f"Final Loss: {best_result['final_loss']:.4f}")
    print(f"Min Loss: {best_result['min_loss']:.4f}")
    print("="*70)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"lr_search_results_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write("Learning Rate Search Results - Text8 Dataset\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Vocab Size: 10,000\n")
        f.write(f"Epochs: 10\n")
        f.write(f"LR Range: 0.5 - 0.8\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'LR':<8} {'Final Loss':<12} {'Min Loss':<12} {'Time (s)':<10}\n")
        f.write("-"*70 + "\n")
        for r in results:
            f.write(f"{r['lr']:<8.2f} {r['final_loss']:<12.4f} {r['min_loss']:<12.4f} {r['training_time']:<10.1f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"BEST LEARNING RATE: {best_result['lr']}\n")
        f.write(f"Final Loss: {best_result['final_loss']:.4f}\n")
        f.write(f"Min Loss: {best_result['min_loss']:.4f}\n")
        f.write("="*70 + "\n\n")
        
        f.write("\nDetailed Results:\n")
        f.write("="*70 + "\n")
        for r in results:
            f.write(f"\nLR = {r['lr']}\n")
            f.write(f"  Final Loss: {r['final_loss']:.4f}\n")
            f.write(f"  Min Loss: {r['min_loss']:.4f}\n")
            f.write(f"  Training Time: {r['training_time']:.1f}s\n")
            f.write(f"  Epoch Losses: {', '.join([f'{l:.4f}' for l in r['epoch_losses']])}\n")
            f.write(f"  Evaluation: {r['eval_summary']}\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
