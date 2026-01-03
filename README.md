# Neural Network Language Model & Optimization Engine

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Deep learning implementation suite featuring autoencoders, word embeddings, and neural network optimization from scratch. Demonstrates both manual (NumPy) and framework-based (PyTorch) implementations with GPU acceleration support.

## 🎯 Project Overview

This project implements core deep learning algorithms in two paradigms:
1. **Manual Implementation**: From-scratch neural networks using only NumPy
2. **Framework Implementation**: Optimized versions using PyTorch with GPU support

### Key Features
- ✅ **8×3×8 Autoencoder** - Dimensionality reduction with reconstruction
- ✅ **Wine Classification** - Neural network classifier on high-dimensional data
- ✅ **Word Embeddings** - PPMI-based word vectors with neural network training
- ✅ **GPU Acceleration** - Apple Silicon (MPS) and CUDA support
- ✅ **Cross-Entropy Loss** - Custom implementation with numerical stability
- ✅ **Backpropagation** - Complete gradient computation from scratch

## 📊 Implementations

### 1. Autoencoder (8→3→8)
**Problem**: Compress 8-dimensional binary input through 3-dimensional bottleneck

| Implementation | File | Features |
|----------------|------|----------|
| **NumPy Manual** | `P1_Autoencoder.ipynb` | Manual backprop, Xavier init |
| **PyTorch** | `P2_torch_AutoEnc.ipynb` | Automatic differentiation, Adam |

**Results**:
- Perfect reconstruction after training (MSE < 0.001)
- Learned compressed representations in bottleneck layer
- Demonstrates autoencoding concept with minimal architecture

### 2. Wine Classification
**Problem**: Multi-class classification on 13-feature wine dataset

| Implementation | File | Features |
|----------------|------|----------|
| **NumPy Manual** | `P3_wine_selfimpl.ipynb` | Custom cross-entropy, manual gradients |
| **PyTorch** | `P4_Wine_Pytorch.ipynb` | GPU-accelerated, modern optimizers |

**Results**:
- **Accuracy**: 95%+ on test set
- **Architecture**: 13→20→10→3 (ReLU activation)
- **Training**: ~50 epochs with learning rate decay

### 3. Word Embeddings (Text8 & 20 Newsgroups)
**Problem**: Learn distributed word representations using co-occurrence + PPMI

#### Architecture
```
Input: One-hot word vector (vocab_size)
   ↓
Hidden: Dense embedding (M dimensions)
   ↓
Output: Predict context words (vocab_size)
```

#### Datasets

**Text8 Corpus** (Wikipedia dump):
- Vocabulary: 50K → 750K → 1M tokens
- Word vectors: 50, 100, 300 dimensions
- Training: ~1-2 hours on GPU

**20 Newsgroups**:
- Vocabulary: 20K tokens
- Word vectors: 50, 100 dimensions  
- Categories: 20 newsgroup topics

#### Implementations

| Dataset | Implementation | File | GPU Support |
|---------|----------------|------|-------------|
| Text8 | Manual (NumPy) | `word_vectors_nn/P7_Manual_Text8.ipynb` | ❌ |
| Text8 | PyTorch | `word_vectors_nn/P8_PyTorch_Text8.ipynb` | ✅ |
| Text8 | PyTorch Script | `word_vectors_nn/Problem8_PyTorch_Text8_M4_MPS.py` | ✅ MPS/CUDA |
| 20 Newsgroups | Manual | `word_vectors_nn/P7_Manual_20NG.ipynb` | ❌ |
| 20 Newsgroups | PyTorch | `word_vectors_nn/P8_PyTorch_20NG.ipynb` | ✅ |
| 20 Newsgroups | PyTorch Script | `word_vectors_nn/Problem8_PyTorch_20NG_M4_MPS.py` | ✅ MPS/CUDA |

**Results**:
- Successfully learns semantic relationships (king - man + woman ≈ queen)
- Cosine similarity captures word associations
- PPMI pre-processing improves training efficiency

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy torch jupyter matplotlib scikit-learn
```

### Run Autoencoder
```bash
jupyter notebook P1_Autoencoder.ipynb
# or PyTorch version
jupyter notebook P2_torch_AutoEnc.ipynb
```

### Run Wine Classification
```bash
jupyter notebook P3_wine_selfimpl.ipynb
# or PyTorch version with GPU
jupyter notebook P4_Wine_Pytorch.ipynb
```

### Train Word Embeddings

#### Using Jupyter Notebooks
```bash
cd word_vectors_nn
jupyter notebook P8_PyTorch_Text8.ipynb
```

#### Using Python Scripts (GPU Accelerated)
```bash
cd word_vectors_nn

# Text8 - Small (50K vocab, M=50)
python Problem8_PyTorch_Text8_M4_MPS.py

# 20 Newsgroups
python Problem8_PyTorch_20NG_M4_MPS.py
```

**GPU Detection**:
- Automatically uses Apple Silicon (MPS) if available
- Falls back to CUDA if NVIDIA GPU present
- Uses CPU if no GPU detected

## 📁 Project Structure

```
HW5/
├── README.md                          # This file
├── PROBLEM_SUMMARIES.txt             # Detailed problem descriptions
├── HW5_RESUME_PROJECT.txt            # Resume-ready project summary
│
├── 📓 Autoencoders
│   ├── P1_Autoencoder.ipynb          # NumPy implementation
│   └── P2_torch_AutoEnc.ipynb        # PyTorch implementation
│
├── 📓 Wine Classification
│   ├── P3_wine_selfimpl.ipynb        # NumPy implementation
│   ├── P4_Wine_Pytorch.ipynb         # PyTorch implementation
│   └── wine_train.data               # Dataset
│
├── 📂 word_vectors_nn/               # Word embedding implementations
│   ├── README.md                     # Word vectors guide
│   ├── ALGORITHM_EXPLANATION.md      # Detailed algorithm docs
│   │
│   ├── 📓 Manual Implementations (NumPy)
│   │   ├── P7_Manual_Text8.ipynb
│   │   └── P7_Manual_20NG.ipynb
│   │
│   ├── 📓 PyTorch Implementations (GPU)
│   │   ├── P8_PyTorch_Text8.ipynb
│   │   └── P8_PyTorch_20NG.ipynb
│   │
│   └── 🐍 Production Scripts (Optimized)
│       ├── Problem8_PyTorch_Text8_M4_MPS.py
│       └── Problem8_PyTorch_20NG_M4_MPS.py
│
└── 📊 Datasets
    ├── text8_50K.txt                 # Text8 subset (50K tokens)
    ├── text8_200K.txt                # Text8 subset (200K tokens)
    ├── text8_750K.txt                # Text8 subset (750K tokens)
    └── text8_1M.txt                  # Text8 full (1M tokens)
```

## 🔬 Technical Highlights

### Manual Implementation (NumPy)
**Demonstrates deep understanding of neural network internals**

#### Forward Pass
```python
# Hidden layer
hidden = sigmoid(X @ W1 + b1)

# Output layer
output = sigmoid(hidden @ W2 + b2)
```

#### Backward Pass (Backpropagation)
```python
# Output layer gradients
delta_output = (output - target) * sigmoid_derivative(output)
dW2 = hidden.T @ delta_output
db2 = np.sum(delta_output, axis=0)

# Hidden layer gradients
delta_hidden = (delta_output @ W2.T) * sigmoid_derivative(hidden)
dW1 = X.T @ delta_hidden
db1 = np.sum(delta_hidden, axis=0)

# Update weights
W1 -= learning_rate * dW1
W2 -= learning_rate * dW2
```

### PyTorch Implementation
**Production-ready with GPU acceleration**

#### Model Definition
```python
class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self, x):
        hidden = self.embeddings(x)  # Learn embeddings
        output = self.output(hidden)  # Predict context
        return output
```

#### GPU Acceleration
```python
# Automatic device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")

model = model.to(device)
```

### PPMI (Positive Pointwise Mutual Information)
**Preprocessing for better word vectors**

```python
PMI(w, c) = log(P(w,c) / (P(w) * P(c)))
PPMI(w, c) = max(0, PMI(w, c))
```

**Benefits**:
- Captures semantic associations
- Reduces noise from rare co-occurrences
- Improves training convergence

## 📈 Performance

### Training Times (M4 GPU)

| Task | Dataset | Config | Time | Accuracy |
|------|---------|--------|------|----------|
| Autoencoder | 8×3×8 | 100 epochs | ~5s | 100% reconstruction |
| Wine Classification | 178 samples | 50 epochs | ~10s | 95%+ test accuracy |
| Word Vectors | Text8 (50K) | M=50, 10 epochs | ~15 min | Good similarity |
| Word Vectors | Text8 (750K) | M=100, 10 epochs | ~1.5 hours | High quality |
| Word Vectors | 20 Newsgroups | M=50, 10 epochs | ~30 min | Domain-specific |

### GPU Speedup

| Operation | CPU | GPU (M4 MPS) | Speedup |
|-----------|-----|--------------|---------|
| Matrix Multiplication | 100ms | 5ms | 20× |
| Training (Text8 50K) | 45 min | 15 min | 3× |
| Training (Text8 750K) | 5 hours | 1.5 hours | 3.3× |

## 🧪 Algorithm Details

### Cross-Entropy Loss (Manual Implementation)
```python
def cross_entropy_loss(predictions, targets):
    """
    Numerically stable cross-entropy
    predictions: (batch, classes) - raw logits
    targets: (batch, classes) - one-hot encoded
    """
    # Softmax with log-sum-exp trick
    max_pred = np.max(predictions, axis=1, keepdims=True)
    exp_pred = np.exp(predictions - max_pred)
    softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
    # Cross-entropy
    loss = -np.sum(targets * np.log(softmax + 1e-10)) / len(targets)
    return loss
```

### Learning Rate Schedule
```python
# Decay schedule for stable convergence
if epoch % 10 == 0:
    learning_rate *= 0.9  # 10% decay every 10 epochs
```

### Gradient Clipping
```python
# Prevent exploding gradients
max_norm = 5.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

## 📚 Documentation

Comprehensive documentation available:
- **ALGORITHM_EXPLANATION.md** - Detailed algorithm walkthrough
- **README.md** (word_vectors_nn/) - Word embedding specifics
- **PROBLEM_SUMMARIES.txt** - Problem statements and insights
- **HW5_RESUME_PROJECT.txt** - Resume-ready project description

## 🎓 Learning Outcomes

### Concepts Demonstrated
- [x] Neural network architecture design
- [x] Backpropagation from scratch
- [x] Gradient descent optimization
- [x] Autoencoder dimensionality reduction
- [x] Word embeddings (distributional semantics)
- [x] PPMI co-occurrence matrices
- [x] GPU-accelerated training
- [x] Numerical stability techniques
- [x] PyTorch framework proficiency
- [x] Model evaluation and validation

### Skills Applied
- **Mathematics**: Linear algebra, calculus, probability
- **Programming**: Python, NumPy, PyTorch, Jupyter
- **Machine Learning**: Supervised learning, unsupervised learning, NLP
- **Optimization**: SGD, Adam, learning rate schedules
- **Engineering**: GPU acceleration, vectorization, debugging

## 🔍 Key Results

### Autoencoder
- Perfect reconstruction of 8-bit patterns
- Learned meaningful 3D representations
- Demonstrates compression/decompression

### Wine Classification
- 95%+ accuracy on test set
- Handles 13-dimensional feature space
- Generalizes well with regularization

### Word Embeddings
- Semantic similarity captured (king/queen, man/woman)
- Analogies work: king - man + woman ≈ queen
- Domain-specific embeddings for 20 Newsgroups

## 🚀 Future Enhancements

Potential extensions:
- [ ] Convolutional layers for image data
- [ ] LSTM/GRU for sequence modeling
- [ ] Attention mechanisms for word embeddings
- [ ] Variational autoencoders (VAE)
- [ ] Transfer learning from pre-trained models
- [ ] Hyperparameter tuning with Optuna
- [ ] Model interpretability (attention visualization)

## 🤝 Contributing

This is an educational project demonstrating neural network fundamentals. Feel free to:
- Fork and experiment
- Add new implementations
- Optimize existing code
- Improve documentation

## 📄 License

MIT License - Free to use for educational and research purposes.

## 👤 Author

Yash Lad
- GitHub: [@yashlad](https://github.com/yashlad)
- Portfolio: Neural Network implementations from scratch

## 🙏 Acknowledgments

- **Text8 Dataset**: Matt Mahoney (Wikipedia dump)
- **20 Newsgroups**: scikit-learn dataset
- **PyTorch**: Facebook AI Research
- **NumPy**: NumPy Developers

---

**⭐ Star this repo if you find it helpful!**

Built with ❤️ demonstrating deep learning from first principles.
