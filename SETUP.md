# Setup Guide

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/yashlad/neural-network-language-model.git
cd neural-network-language-model
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install numpy torch jupyter matplotlib scikit-learn
```

Or create a `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 4: Download Datasets
The Text8 datasets are included in the repository. If you need the full Text8:
```bash
# Download from Matt Mahoney's website
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
```

## Running the Code

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Navigate to desired notebook:
# - P1_Autoencoder.ipynb
# - P3_wine_selfimpl.ipynb
# - word_vectors_nn/P8_PyTorch_Text8.ipynb
```

### Python Scripts (GPU Accelerated)
```bash
cd word_vectors_nn

# Text8 word embeddings
python Problem8_PyTorch_Text8_M4_MPS.py

# 20 Newsgroups word embeddings
python Problem8_PyTorch_20NG_M4_MPS.py
```

## GPU Support

### Apple Silicon (M1/M2/M3/M4)
PyTorch automatically detects MPS backend:
```python
device = torch.device("mps")  # Automatically used if available
```

### NVIDIA CUDA
```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### CPU Only
If no GPU detected, falls back to CPU automatically.

## Troubleshooting

### PyTorch MPS Issues
```bash
# If MPS fails, force CPU:
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Memory Issues
Reduce batch size or vocabulary size in scripts:
```python
BATCH_SIZE = 32  # Reduce from 64
VOCAB_SIZE = 10000  # Reduce from 50000
```

### Jupyter Kernel Issues
```bash
# Install ipykernel
pip install ipykernel
python -m ipykernel install --user --name=venv
```

## Project Structure

```
HW5/
├── README.md              # Main documentation
├── SETUP.md              # This file
├── LICENSE               # MIT License
├── .gitignore            # Git ignore patterns
│
├── P1_Autoencoder.ipynb
├── P2_torch_AutoEnc.ipynb
├── P3_wine_selfimpl.ipynb
├── P4_Wine_Pytorch.ipynb
├── wine_train.data
│
├── text8_50K.txt
├── text8_200K.txt
├── text8_750K.txt
├── text8_1M.txt
│
└── word_vectors_nn/
    ├── README.md
    ├── ALGORITHM_EXPLANATION.md
    ├── P7_Manual_Text8.ipynb
    ├── P8_PyTorch_Text8.ipynb
    ├── Problem8_PyTorch_Text8_M4_MPS.py
    └── Problem8_PyTorch_20NG_M4_MPS.py
```

## Next Steps

1. Start with simple examples (Autoencoder, Wine Classification)
2. Move to word embeddings (Text8 50K first)
3. Scale up to larger datasets (750K, 1M)
4. Experiment with hyperparameters
5. Compare manual vs PyTorch implementations

## Resources

- **Documentation**: See README.md and ALGORITHM_EXPLANATION.md
- **Issues**: Report bugs on GitHub Issues
- **Contributing**: Fork and submit pull requests

Happy coding! 🚀
