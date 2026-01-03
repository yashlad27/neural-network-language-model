# Notebook Markdown Guide

Quick reference for adding explanatory markdown cells to each notebook.

---

## Problem 7: Manual Implementation (Text8)
**File:** `Problem7_Manual_Text8.ipynb`

### Markdown to Add After Title Cell:
```markdown
## Overview
This notebook implements word embeddings using a **manually coded neural network** (NumPy only) trained on PPMI matrices from the **Text8 dataset** (200K words from Wikipedia).

### Algorithm Flow
1. **Preprocess** → Clean text, tokenize
2. **Build Vocabulary** → Top 5,000 words
3. **Co-occurrence Matrix** → Count word pairs within window of 5
4. **PPMI** → Normalize co-occurrences to measure associations
5. **Neural Network** → Train 2-layer network (no bias, linear activations)
6. **Extract Embeddings** → Get word vectors from first layer weights
7. **Evaluate** → Find similar words using cosine similarity

### Key Parameters
- Vocabulary: 5,000 words
- Embedding Dimension: 200
- Window Size: 5
- Learning Rate: 0.789
- Epochs: 10
- Training: One word at a time (SGD)
```

### Markdown to Add Before Training Section:
```markdown
## Neural Network Architecture

**Structure:**
```
Input (one-hot) → W1 (5000×200, no bias) → Hidden (200D) → W2 (200×5000, no bias) → Output → Softmax
```

**Training Process:**
- **Forward:** Predict PPMI distribution for each word
- **Loss:** Cross-entropy between predicted and actual PPMI
- **Backward:** Manual gradient computation
- **Update:** W1, W2 -= learning_rate × gradients

**Math:**
```
h = W1^T × x           (embedding layer)
o = W2^T × h           (reconstruction layer)
ŷ = softmax(o)         (probability distribution)
L = -Σ y_i × log(ŷ_i)  (cross-entropy loss)
```
```

### Markdown to Add Before Evaluation:
```markdown
## Evaluation Strategy

**Method:** Cosine Similarity
```
sim(u, v) = (u · v) / (||u|| × ||v||)
```

**Test Words:** china, computer, phone, napoleon, god, catholic

**Expected Patterns:**
- Geographic words cluster together (china → mongolia, korea)
- Technical words cluster (computer → software, systems)
- Religious words cluster (god → prophecy, sacred)
```

---

## Problem 7: Manual Implementation (20 Newsgroups)
**File:** `Problem7_Manual_20NG.ipynb`

### Markdown to Add After Title Cell:
```markdown
## Overview
Manual neural network implementation on **20 Newsgroups dataset** (1M+ words from Usenet discussions). Uses same architecture as Text8 but with larger vocabulary to capture domain-specific terms.

### Dataset Details
- **Source:** 8 newsgroup categories (comp.graphics, sci.crypt, talk.religion, etc.)
- **Size:** ~1M words
- **Preprocessing:** Aggressive stop word filtering for cleaner technical/political vocabulary

### Why 20 Newsgroups?
Tests if embeddings can capture:
- **Technical jargon** (computer, software, encryption)
- **Political discourse** (china, invasion, democracy)
- **Religious terminology** (god, catholic, prophecy)

### Key Differences from Text8
- Larger vocabulary: 10,000 words (vs 5,000)
- Domain-specific: Technical + political focus
- More noise: Informal discussion vs. Wikipedia prose
```

---

## Problem 8: PyTorch Implementation (Text8)
**File:** `Problem8_PyTorch_Text8.ipynb`

### Markdown to Add After Title Cell:
```markdown
## Overview
**PyTorch implementation** of the same algorithm as Problem 7, but using automatic differentiation and GPU acceleration. Demonstrates production-ready deep learning practices.

### Advantages of PyTorch Version
1. **Automatic Gradients**: No manual backpropagation
2. **GPU Support**: 10-100× faster with CUDA/MPS
3. **Built-in Optimizer**: `torch.optim.SGD` handles updates
4. **Cleaner Code**: More concise and maintainable

### Architecture (Same as Manual)
```python
class WordVectorNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        self.W1 = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.W2 = nn.Linear(embedding_dim, vocab_size, bias=False)
        # Initialized with N(0, 0.1)
```

### Training Loop (PyTorch Style)
```python
for epoch in range(epochs):
    for word_idx in vocabulary:
        optimizer.zero_grad()           # Clear gradients
        prediction = model(one_hot)     # Forward pass
        loss = CE_loss(prediction, ppmi)  # Compute loss
        loss.backward()                 # Auto-compute gradients
        optimizer.step()                # Update weights
```
```

### Markdown to Add Before Device Setup:
```markdown
## Device Configuration
This implementation auto-detects available compute:
- **MPS** (Apple Silicon M1-M4): Best for Mac
- **CUDA** (NVIDIA GPU): Best for Linux/Windows
- **CPU**: Fallback (slower but works everywhere)
```

---

## Problem 8: PyTorch Implementation (20 Newsgroups)
**File:** `Problem8_PyTorch_20NG.ipynb`

### Markdown to Add After Title Cell:
```markdown
## Overview
PyTorch implementation on **20 Newsgroups** with 10,000 word vocabulary. Combines benefits of:
- Domain-specific dataset (technical + political vocabulary)
- Modern deep learning framework (PyTorch)
- GPU acceleration (MPS/CUDA)

### Expected Training Time
- **MPS (Apple M4)**: ~4-5 minutes
- **CUDA (GPU)**: ~3-4 minutes  
- **CPU**: ~20-30 minutes

### Hyperparameter Tuning
This notebook uses learning rate **0.789** (found through grid search on 0.5-0.8 range). Higher learning rates work well because:
1. PPMI provides smooth target distributions
2. No bias terms reduce parameter space
3. Linear activations avoid gradient issues
```

### Markdown to Add Before PPMI Computation:
```markdown
## PPMI: From Counts to Meaningful Associations

**Why Not Raw Co-occurrence?**
- Biased toward frequent words ("the", "and", "is")
- Doesn't measure "surprise" in co-occurrence

**PPMI Solution:**
```
PMI(w,c) = log(P(w,c) / (P(w) × P(c)))
PPMI(w,c) = max(0, PMI(w,c))
```

**Interpretation:**
- **High PPMI**: Words co-occur more than random → meaningful association
- **Zero PPMI**: Co-occur as expected or less → no special relationship

**Normalization:** Row-normalize to get probability distributions for training.
```

---

## Common Markdown Sections for All Notebooks

### Before Co-occurrence Matrix:
```markdown
## Co-occurrence Matrix Construction

**Window-Based Counting:**
For each word at position `i`, count context words at positions `j` where `|i-j| ≤ window_size`.

**Distance Weighting:**
```
weight = 1 / distance
```
Closer words get higher weight (e.g., adjacent word = 1.0, distance 2 = 0.5).

**Example:**
```
Text: "the cat sat on the mat"
For "sat" with window=2:
  - "cat" (distance 1): weight 1.0
  - "on" (distance 1): weight 1.0  
  - "the" (distance 2): weight 0.5
  - "the" (distance 2): weight 0.5
```
```

### Before Training:
```markdown
## Training Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Learning Rate | 0.789 | Tuned for fast convergence |
| Epochs | 10 | Balance speed vs. quality |
| Batch Size | 1 | One word at a time (SGD) |
| Embedding Dim | 200 | Standard for word embeddings |
| No Bias | True | Focus on relationships, not frequencies |
| Activation | Linear | Simplifies gradient computation |
```

### After Training Results:
```markdown
## Loss Interpretation

**Typical Loss Curve:**
- **Epoch 1-3**: ~9.0-8.5 (initial learning)
- **Epoch 4-7**: ~8.5-7.5 (fast improvement)
- **Epoch 8-10**: ~7.5-6.5 (convergence)

**Why Loss Doesn't Reach 0:**
1. PPMI is noisy (limited data)
2. 200D can't perfectly represent all relationships
3. Model trades off different word pairs

**Lower Loss = Better?** Not always! Focus on embedding quality (semantic similarity).
```

### Before Evaluation Results:
```markdown
## Embedding Quality Checks

**Good Embeddings Should Show:**
1. **Semantic Similarity**: Related words cluster
   - china → {mongolia, korea, thailand}
   - computer → {software, systems, hardware}
   
2. **Analogies**: Word relationships transfer
   - king - man + woman ≈ queen
   - paris - france + germany ≈ berlin

3. **Domain Coherence**: Technical/religious/political terms separate

**Cosine Similarity Range:**
- 0.8-1.0: Very similar (often synonyms)
- 0.5-0.8: Related (same domain)
- 0.0-0.5: Weak/no relation
- <0.0: Opposite meanings (rare)
```

---

## Usage Instructions

1. **Choose relevant sections** for each notebook
2. **Add as markdown cells** at appropriate locations
3. **Customize examples** based on actual results
4. **Keep concise** - each section should be 2-5 sentences + code/math

## Visual Elements to Add

### For Architecture Diagrams:
```
┌─────────────┐
│  One-Hot    │ (5000 dim)
│  Encoding   │
└──────┬──────┘
       │
   ┌───▼───┐
   │  W1   │ (5000×200, no bias)
   └───┬───┘
       │
┌──────▼──────┐
│   Hidden    │ (200 dim)
│  Embedding  │
└──────┬──────┘
       │
   ┌───▼───┐
   │  W2   │ (200×5000, no bias)
   └───┬───┘
       │
┌──────▼──────┐
│   Output    │ (5000 dim)
└──────┬──────┘
       │
┌──────▼──────┐
│  Softmax    │
└─────────────┘
```

### For Loss Visualization:
```
Loss Over Epochs:
9.2 |●                     (Epoch 1)
    |  ●                  
8.5 |    ●                (Epoch 3)
    |      ●             
8.0 |        ●           (Epoch 5)
    |          ●         
7.5 |            ●       (Epoch 7)
    |              ●     
7.0 |                ●   (Epoch 9)
6.7 |                  ● (Epoch 10)
    └─────────────────────
     1  2  3  4  5  6  7  8  9  10
```

---

## Tips for Writing Good Markdown

1. **Start with "Why"**: Explain motivation before details
2. **Use Examples**: Concrete examples > abstract definitions
3. **Break Up Math**: Separate complex equations into steps
4. **Add Intuition**: What does this math mean in plain English?
5. **Link Concepts**: "This PPMI matrix becomes training targets for..."
6. **Highlight Key Points**: Use **bold** for important concepts
7. **Use Code Blocks**: Show pseudo-code for clarity
8. **Keep Consistent**: Use same terminology across notebooks

---

## Example: Complete Markdown Cell

```markdown
## PPMI Computation

### What is PPMI?
**Positive Pointwise Mutual Information** measures how much two words "like" to appear together, beyond random chance.

### The Math
```
PMI(w, c) = log(P(w,c) / (P(w) × P(c)))
PPMI(w, c) = max(0, PMI(w, c))
```

### Intuition
- If "neural" and "network" appear together often: **High PMI** → meaningful association
- If "the" and "network" appear together: **Low/Negative PMI** → no special relationship
- **PPMI removes negative values**: We only care about positive associations

### Implementation
We vectorize the computation for speed:
```python
p_ij = cooccur / total                    # Joint probability
p_i_p_j = (word_counts @ context_counts)  # Marginal product (outer)
pmi = torch.log(p_ij / (p_i_p_j + ε))    # PMI
ppmi = torch.clamp(pmi, min=0)            # Positive only
```

### Result
A normalized probability distribution for each word, used as training targets.
```
```

This gives you all the context, math, and ready-to-use markdown sections for your notebooks!
