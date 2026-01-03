# Word Vectors - Neural Network Implementations

This folder contains implementations following the TA's hints for training word embeddings using neural networks trained on PPMI matrices.

## Structure

### Problem 7: Manual NumPy Implementation
- **Problem7_Manual_Text8.ipynb** - Text8 dataset (200K words)
- **Problem7_Manual_20NG.ipynb** - 20 Newsgroups (5-8 categories)

### Problem 8: PyTorch Implementation
- **Problem8_PyTorch_Text8.ipynb** - Text8 dataset (200K words)
- **Problem8_PyTorch_20NG.ipynb** - 20 Newsgroups (5-8 categories)

## Architecture (Following TA Hints)

```
Input (one-hot)
    ↓
W1 (vocab_size × embedding_dim) - NO BIAS
    ↓
Hidden (embedding_dim)
    ↓
W2 (embedding_dim × vocab_size) - NO BIAS
    ↓
Output (vocab_size)
    ↓
Softmax
```

## Key Implementation Details

### Preprocessing
1. Convert to lowercase
2. Remove punctuation and special symbols
3. Tokenize into words
4. Build vocabulary (5000 words, filter stop words)
5. Create word_to_id and id_to_word mappings
6. Build corpus with vocab words only

### Co-occurrence Matrix
1. Window size: 5
2. Distance weighting: 1/distance
3. Symmetric matrix (vocab_size × vocab_size)

### PPMI Computation
```
PMI = log(P(i,j) / (P(i) * P(j)))
PPMI = max(0, PMI)
```
- Self-pairs set to 0 (diagonal = 0)
- Rows normalized to probability distribution

### Neural Network Training
- **No bias terms** in linear layers
- **No intermediate activations** (hidden layer = raw linear output)
- **Softmax only at output** layer
- **Train one word at a time** (no batching)
- **Learning rate:** 0.1 (range: 0.1-0.8)
- **Epochs:** 10
- **Weight initialization:** Gaussian N(0, 0.1)
- **Manual CE loss:** `-sum(target * log(probs))`
- **Optimizer (PyTorch):** `torch.optim.SGD`

### Forward Pass
1. One-hot encode input word
2. Hidden = one_hot @ W1 (or just W1[word_idx])
3. Output = hidden @ W2
4. Probs = softmax(output)

### Backward Pass
```
d_output = probs - target  (simplified CE + softmax derivative)
d_W2 = outer(hidden, d_output)
d_hidden = d_output @ W2.T
d_W1[word_idx] = d_hidden
```

### Evaluation
1. Extract embeddings from W1 (first layer weights)
2. For similarity: compute cosine similarity between embeddings
3. Return top 15 most similar words

## Running

### Manual (NumPy)
```bash
jupyter notebook Problem7_Manual_Text8.ipynb
jupyter notebook P7_Manual_20NG.ipynb
```

### PyTorch
```bash
jupyter notebook Problem8_PyTorch_Text8.ipynb
jupyter notebook Problem8_PyTorch_20NG.ipynb
```

## Expected Training Time
- Text8 (200K words, 5K vocab): ~5 minutes per epoch
- 20NG (5-8 categories, 5K vocab): ~5 minutes per epoch

## Configuration
- Vocabulary: 5,000 words
- Embedding dimension: 200
- Window size: 5
- Learning rate: 0.1
- Epochs: 10

## Dependencies
```
numpy
torch
sklearn
tqdm
```
