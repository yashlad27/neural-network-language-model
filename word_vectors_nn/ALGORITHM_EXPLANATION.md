# Word Vector Neural Network Implementation - Algorithm Explanation

## Overview

This document explains the algorithm, mathematical foundations, and learning process used in all four Jupyter notebooks for creating word embeddings using neural networks trained on PPMI matrices.

---

## Table of Contents

1. [Problem 7: Manual Implementation (NumPy)](#problem-7-manual-implementation)
   - [Text8 Dataset](#problem-7-text8)
   - [20 Newsgroups Dataset](#problem-7-20-newsgroups)
2. [Problem 8: PyTorch Implementation](#problem-8-pytorch-implementation)
   - [Text8 Dataset](#problem-8-text8)
   - [20 Newsgroups Dataset](#problem-8-20-newsgroups)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Learning Process](#learning-process)
5. [Key Differences Between Implementations](#key-differences)

---

## High-Level Algorithm Flow

### 1. Text Preprocessing
**Purpose:** Clean and prepare raw text data for vocabulary building.

**Steps:**
- Convert all text to lowercase
- Remove punctuation and special characters (keep only a-z and spaces)
- Tokenize into words
- Apply comprehensive stop word filtering

**Math:** None (text manipulation only)

**Code Pattern:**
```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]+", ' ', text)  # Remove non-alphabetic
    text = re.sub(r'\s+', ' ', text)        # Normalize whitespace
    return text.split()
```

---

### 2. Vocabulary Building
**Purpose:** Create a mapping between words and unique integer IDs.

**Steps:**
1. Count word frequencies
2. Filter by:
   - Minimum count threshold (default: 3 occurrences)
   - Minimum word length (≥3 characters)
   - Not in stop word list
3. Keep top N most frequent words (vocab_size)
4. Create bidirectional mappings: word ↔ ID

**Math:**
```
Filtered words = {w : count(w) ≥ min_count ∧ len(w) ≥ 3 ∧ w ∉ StopWords}
Vocabulary = top_k(Filtered words, k = vocab_size)
```

**Output:**
- `word_to_id`: Dictionary mapping word → integer ID
- `id_to_word`: Dictionary mapping integer ID → word
- `corpus`: List of word IDs representing the text

---

### 3. Co-occurrence Matrix Construction
**Purpose:** Capture how often words appear near each other in text.

**Algorithm:**
For each word position i in the corpus:
1. Define a context window of size W (default: 5)
2. For each word j within the window (excluding i):
   - Weight the co-occurrence by inverse distance: `1 / |i - j|`
   - Add weight to `cooccur[word_i][word_j]`

**Math:**
```
CoOccur[w_i, w_j] = Σ (1 / distance(i, j))
                    for all positions i, j where:
                    - |i - j| ≤ window_size
                    - i ≠ j
```

**Example:**
```
Text:     "the cat sat on the mat"
Positions: 0   1   2  3   4   5

For "sat" (position 2), window_size=2:
- "cat" (pos 1): weight = 1/|2-1| = 1.0
- "the" (pos 0): weight = 1/|2-0| = 0.5
- "on"  (pos 3): weight = 1/|2-3| = 1.0
- "the" (pos 4): weight = 1/|2-4| = 0.5
```

**Matrix Properties:**
- Size: `vocab_size × vocab_size`
- Symmetric (if undirected context)
- Sparse (most word pairs don't co-occur)

---

### 4. PPMI (Positive Pointwise Mutual Information) Computation
**Purpose:** Normalize co-occurrence counts to measure meaningful word associations.

**Intuition:** 
PMI measures how much more likely two words co-occur than we'd expect by chance. PPMI sets negative values to 0 (we only care about positive associations).

**Math:**

**Step 1: Compute Probabilities**
```
Total = Σ CoOccur[i,j]  (sum of all co-occurrences)

P(w_i, w_j) = CoOccur[i,j] / Total        # Joint probability
P(w_i)      = Σ_j CoOccur[i,j] / Total    # Marginal probability of word i
P(w_j)      = Σ_i CoOccur[i,j] / Total    # Marginal probability of word j
```

**Step 2: Compute PMI**
```
PMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) × P(w_j)))
```

**Step 3: Positive PMI**
```
PPMI(w_i, w_j) = max(0, PMI(w_i, w_j))
```

**Step 4: Row Normalization**
```
PPMI_normalized[i,j] = PPMI[i,j] / Σ_j PPMI[i,j]
```

**Interpretation:**
- High PPMI: Words co-occur much more than expected → strong association
- Low/Zero PPMI: Words co-occur as expected or less → weak/no association
- Normalization converts to a probability distribution (sums to 1 per row)

**Example:**
```
If "neural" and "network" appear together often:
- P(neural, network) = 0.001  (they co-occur 1000 times out of 1M pairs)
- P(neural) = 0.01            (neural appears 10K times)
- P(network) = 0.01           (network appears 10K times)

PMI = log(0.001 / (0.01 × 0.01)) = log(10) ≈ 2.3
PPMI = max(0, 2.3) = 2.3  (positive association!)

If "the" and "network" appear together:
- P(the, network) = 0.0001    (fewer co-occurrences)
- P(the) = 0.05               (the is very common)
- P(network) = 0.01

PMI = log(0.0001 / (0.05 × 0.01)) = log(0.2) ≈ -1.6
PPMI = max(0, -1.6) = 0  (no meaningful association)
```

---

### 5. Neural Network Architecture
**Purpose:** Learn word embeddings by predicting PPMI distributions.

**Architecture:**
```
Input Layer:    One-hot encoded word (vocab_size dimensions)
      ↓
First Layer:    W1 (vocab_size × embedding_dim), NO BIAS
      ↓
Hidden Layer:   Linear output (embedding_dim dimensions)
      ↓
Second Layer:   W2 (embedding_dim × vocab_size), NO BIAS
      ↓
Output Layer:   Linear output (vocab_size dimensions)
      ↓
Softmax:        Convert to probability distribution
```

**Math:**

**Forward Pass:**
```
x = one_hot(word)                    # vocab_size dimensional vector
h = W1^T × x                         # embedding_dim dimensional vector (hidden)
o = W2^T × h                         # vocab_size dimensional vector (output)
ŷ = softmax(o)                       # predicted PPMI distribution
```

**Where:**
- `x[i] = 1` if word is i, else 0
- `h` is the word embedding
- `W1` contains embeddings as columns
- `W2` maps embeddings back to vocabulary space

**Weight Initialization:**
```
W1 ~ N(0, 0.1²)  (Gaussian with mean 0, std 0.1)
W2 ~ N(0, 0.1²)
```

**Key Design Choices:**
1. **No bias**: Focuses learning on word relationships, not absolute frequencies
2. **Linear activations**: No ReLU/sigmoid in hidden layer
3. **Softmax output**: Ensures valid probability distribution

---

### 6. Training Process

#### Loss Function: Cross-Entropy
**Purpose:** Measure how well predicted distribution matches PPMI distribution.

**Math:**
```
L = -Σ_j y_j × log(ŷ_j + ε)

Where:
- y_j = PPMI_normalized[word, j]  (true distribution)
- ŷ_j = softmax(output)[j]         (predicted distribution)
- ε = 1e-10                         (for numerical stability)
```

**Intuition:** 
- If model predicts high probability for words with high PPMI, loss is low
- If model predicts wrong words, loss is high

#### Gradient Descent Update
**Manual Implementation (Problem 7):**

**Step 1: Compute Softmax Gradient**
```
For cross-entropy + softmax, the gradient simplifies to:
∂L/∂o = ŷ - y  (predicted - true)
```

**Step 2: Backpropagate to W2**
```
∂L/∂W2 = h × (ŷ - y)^T
```

**Step 3: Backpropagate to Hidden**
```
∂L/∂h = W2 × (ŷ - y)
```

**Step 4: Backpropagate to W1**
```
∂L/∂W1 = x × (∂L/∂h)^T
```

**Step 5: Update Weights**
```
W1 ← W1 - learning_rate × ∂L/∂W1
W2 ← W2 - learning_rate × ∂L/∂W2
```

**PyTorch Implementation (Problem 8):**
```python
optimizer.zero_grad()           # Clear previous gradients
loss = manual_ce_loss(ŷ, y)    # Compute loss
loss.backward()                 # Auto-compute gradients
optimizer.step()                # Update weights
```

#### Training Algorithm
```
For epoch in 1..N:
    Shuffle word indices
    For each word w in vocabulary:
        1. Create one-hot encoding: x
        2. Get target PPMI distribution: y = PPMI[w, :]
        3. Forward pass: ŷ = model(x)
        4. Compute loss: L = CE(ŷ, y)
        5. Backward pass: compute gradients
        6. Update weights: W1, W2 -= lr × gradients
    
    Report average loss
```

**Hyperparameters:**
- **Learning Rate:** 0.789 (after tuning, typically 0.5-0.8)
- **Epochs:** 10
- **Embedding Dimension:** 200
- **Vocab Size:** 5,000 (Text8), 10,000 (20 Newsgroups)
- **Batch Size:** 1 (stochastic gradient descent, one word at a time)

---

### 7. Embedding Extraction
**Purpose:** Get the learned word vectors for downstream tasks.

**Method:**
The embeddings are stored in the first layer weights `W1`.

**Math:**
```
Embedding of word i = W1[:, i]  (i-th column of W1)

Or equivalently:
Embedding = W1^T (transposed, so each row is an embedding)
```

**Output:**
- Matrix of shape `(vocab_size, embedding_dim)`
- Row i contains the 200-dimensional embedding for word i

---

### 8. Evaluation: Finding Similar Words
**Purpose:** Test if learned embeddings capture semantic similarity.

**Algorithm:**
```
Given query word w:
1. Get embedding: e_w = Embeddings[word_to_id[w]]
2. For each other word v:
   a. Get embedding: e_v = Embeddings[word_to_id[v]]
   b. Compute cosine similarity: sim(w,v)
3. Sort by similarity (descending)
4. Return top K words
```

**Math: Cosine Similarity**
```
sim(u, v) = (u · v) / (||u|| × ||v||)

Where:
- u · v = Σ_i u_i × v_i        (dot product)
- ||u|| = √(Σ_i u_i²)          (L2 norm)
```

**Interpretation:**
- `sim = 1`: Identical direction (most similar)
- `sim = 0`: Orthogonal (no similarity)
- `sim = -1`: Opposite direction

**Example:**
```
Query: "china"
Top similar words:
- mongolia (0.64)  ← geographically close
- korea (0.63)     ← geographically close
- india (0.60)     ← geographically close
```

---

## How the Model Learns

### Phase 1: Random Initialization
- Embeddings start as random Gaussian vectors
- No semantic meaning initially

### Phase 2: Training Iterations
For each word:
1. **Forward Pass**: Model predicts which context words should co-occur
2. **Compare**: Prediction vs. actual PPMI distribution
3. **Error Signal**: Difference (predicted - actual) indicates wrong predictions
4. **Backpropagation**: Adjusts embeddings to reduce error

### Phase 3: Embedding Formation
Through repeated updates:
- Words with similar PPMI patterns get pushed to similar embedding vectors
- Words that co-occur with similar contexts become neighbors in embedding space
- Semantic relationships emerge (countries near countries, verbs near verbs)

### Convergence
**Loss Decreases Over Epochs:**
```
Epoch 1:  Loss ≈ 9.2  (poor predictions)
Epoch 5:  Loss ≈ 9.0  (learning patterns)
Epoch 10: Loss ≈ 8.0  (good predictions)
```

**Why Loss Doesn't Reach 0:**
- PPMI is noisy (sparse co-occurrence data)
- Low-dimensional embeddings (200D) can't perfectly capture all relationships
- Regularization effect from model architecture

---

## Key Differences Between Implementations

### Problem 7 (Manual NumPy) vs Problem 8 (PyTorch)

| Aspect | Problem 7 (Manual) | Problem 8 (PyTorch) |
|--------|-------------------|---------------------|
| **Gradients** | Hand-coded backpropagation | Automatic differentiation |
| **Forward Pass** | Manual matrix multiplication | `nn.Linear` layers |
| **Optimizer** | Manual SGD updates | `torch.optim.SGD` |
| **Device** | CPU only (NumPy) | CPU/CUDA/MPS support |
| **Speed** | Slower (Python loops) | Faster (optimized kernels) |
| **Code Clarity** | Explicit, educational | Concise, production-ready |

### Text8 vs 20 Newsgroups

| Aspect | Text8 | 20 Newsgroups |
|--------|-------|---------------|
| **Domain** | Wikipedia articles | Usenet discussions |
| **Size** | 200K words | 1M+ words |
| **Vocab** | 5,000 words | 10,000 words |
| **Topics** | General knowledge | Tech, religion, politics |
| **Preprocessing** | Basic | More aggressive stop words |

---

## Complete Mathematical Formulation

### Notation
- `V`: Vocabulary size
- `D`: Embedding dimension (200)
- `W`: Window size (5)
- `T`: Number of epochs (10)
- `η`: Learning rate (0.789)

### Full Algorithm
```
Input: Corpus of text
Output: Word embeddings E ∈ ℝ^(V×D)

1. Preprocess:
   words = preprocess(corpus)

2. Build Vocabulary:
   V = {top V words by frequency}
   word_to_id = {word: i for i, word in enumerate(V)}

3. Build Co-occurrence Matrix C ∈ ℝ^(V×V):
   For position i in corpus:
       For position j where |i-j| ≤ W and i ≠ j:
           C[corpus[i], corpus[j]] += 1/|i-j|

4. Compute PPMI Matrix P ∈ ℝ^(V×V):
   total = Σ_{i,j} C[i,j]
   p_i = Σ_j C[i,j] / total
   p_j = Σ_i C[i,j] / total
   p_ij = C[i,j] / total
   
   PMI[i,j] = log(p_ij / (p_i × p_j))
   PPMI[i,j] = max(0, PMI[i,j])
   P[i,j] = PPMI[i,j] / Σ_k PPMI[i,k]

5. Initialize Weights:
   W1 ~ N(0, 0.1²) ∈ ℝ^(V×D)
   W2 ~ N(0, 0.1²) ∈ ℝ^(D×V)

6. Train for T epochs:
   For epoch = 1 to T:
       For each word index i ∈ {0, 1, ..., V-1} (shuffled):
           # One-hot encode
           x = one_hot(i) ∈ ℝ^V
           
           # Forward pass
           h = W1^T × x ∈ ℝ^D
           o = W2^T × h ∈ ℝ^V
           ŷ = softmax(o) ∈ ℝ^V
           
           # Loss
           L = -Σ_j P[i,j] × log(ŷ[j] + ε)
           
           # Backward pass
           δ_out = ŷ - P[i,:] ∈ ℝ^V
           δ_W2 = h × δ_out^T ∈ ℝ^(D×V)
           δ_h = W2 × δ_out ∈ ℝ^D
           δ_W1 = x × δ_h^T ∈ ℝ^(V×D)
           
           # Update
           W1 ← W1 - η × δ_W1
           W2 ← W2 - η × δ_W2

7. Extract Embeddings:
   E = W1^T ∈ ℝ^(V×D)
   E[i,:] is the embedding for word i
```

---

## Theoretical Foundation

### Why PPMI?
Traditional co-occurrence counts are biased toward frequent words. PPMI normalizes by:
1. Dividing by marginal probabilities (removes frequency bias)
2. Taking log (compresses dynamic range)
3. Clamping to 0 (focuses on positive associations)

### Why Neural Network?
1. **Dimensionality Reduction**: PPMI matrix is V×V (sparse, high-dimensional). Embeddings are V×D where D << V (dense, low-dimensional)
2. **Generalization**: Neural network learns to compress PPMI patterns into compact representations
3. **Distributed Representations**: Each word is a point in continuous space rather than a discrete symbol

### Connection to Word2Vec
This approach is conceptually similar to Word2Vec's Skip-gram model, but:
- **Word2Vec**: Predicts context words from target word
- **Our Method**: Predicts PPMI distribution (derived from co-occurrence)
- Both learn by aligning similar contexts to similar embeddings

---

## Practical Implementation Notes

### Computational Complexity
1. **Co-occurrence Matrix**: O(N × W) where N = corpus length
2. **PPMI Computation**: O(V²) for vectorized version
3. **Training**: O(T × V × D × V) = O(T × V² × D) per epoch

### Memory Requirements
- Co-occurrence matrix: V² floats (sparse in practice)
- PPMI matrix: V² floats
- Model weights: V×D + D×V = 2VD floats
- Example: V=10K, D=200 → ~400MB

### Optimization Tips
1. **Vectorize PPMI**: Use matrix operations instead of loops (100-1000x speedup)
2. **GPU Acceleration**: PyTorch version can use CUDA/MPS for 10-100x speedup
3. **Batch Processing**: Could process multiple words simultaneously (not done here per TA requirements)

---

## Expected Results

### Training Progress
```
Text8 (5K vocab):
Epoch 1:  8.8 → Epoch 5:  8.3 → Epoch 10: 6.7

20 Newsgroups (10K vocab):
Epoch 1:  9.2 → Epoch 5:  9.0 → Epoch 10: 8.0
```

### Embedding Quality
Good embeddings should show:
- **Geographic clustering**: china → {mongolia, korea, india}
- **Semantic clustering**: computer → {software, systems, hardware}
- **Domain clustering**: god → {prophecy, sacred, church}

---

## Summary

This implementation demonstrates:
1. **Statistical NLP**: Using co-occurrence statistics (PPMI) to capture word relationships
2. **Deep Learning**: Using neural networks for dimensionality reduction
3. **Representation Learning**: Learning continuous vector representations of discrete symbols
4. **Evaluation**: Measuring quality through semantic similarity

The key insight: **Words that appear in similar contexts have similar meanings**, and we can capture this through co-occurrence patterns compressed into low-dimensional embeddings.
