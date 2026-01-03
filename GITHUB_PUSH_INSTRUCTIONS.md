# GitHub Push Instructions

## Step-by-Step Guide to Push HW5 as Standalone Repository

### 1. Create GitHub Repository

1. Go to https://github.com/new
2. Fill in repository details:
   - **Repository name**: `neural-network-language-model` (or your preferred name)
   - **Description**: "Neural Network Language Model & Optimization Engine - Deep learning implementations featuring autoencoders, word embeddings, and neural network optimization from scratch"
   - **Visibility**: Public or Private (your choice)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"

### 2. Initialize Git Repository Locally

Open terminal in the HW5 directory and run:

```bash
cd /Users/yashlad/Development/CS6140/HW5

# Initialize git repository
git init

# Add all files
git add .

# Check what will be committed (optional)
git status

# Create initial commit
git commit -m "Initial commit: Neural Network Language Model & Optimization Engine

- Autoencoder implementations (NumPy and PyTorch)
- Wine classification neural networks
- Word embedding models (Text8 and 20 Newsgroups)
- Manual backpropagation from scratch
- GPU-accelerated PyTorch versions (MPS/CUDA)
- Comprehensive documentation and algorithm explanations"
```

### 3. Connect to GitHub Remote

Replace `YOUR_USERNAME` with your GitHub username:

```bash
# Add GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/neural-network-language-model.git

# Verify remote
git remote -v
```

### 4. Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

**Alternative if you prefer SSH:**
```bash
# Use SSH URL instead
git remote add origin git@github.com:YOUR_USERNAME/neural-network-language-model.git
git push -u origin main
```

### 5. Verify Upload

1. Go to your repository URL: `https://github.com/YOUR_USERNAME/neural-network-language-model`
2. Verify all files are present
3. Check that README.md displays correctly
4. Confirm .gitignore is working (no `__pycache__/` or `.DS_Store` files)

---

## Troubleshooting

### Authentication Issues

**HTTPS Authentication:**
```bash
# GitHub will prompt for username and Personal Access Token
# Create token at: https://github.com/settings/tokens
```

**SSH Authentication:**
```bash
# Generate SSH key if needed
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key and add to GitHub
cat ~/.ssh/id_ed25519.pub
# Add at: https://github.com/settings/keys
```

### Large File Issues

If you get errors about large files (>100MB):

```bash
# Check file sizes
find . -type f -size +50M

# If text8_1M.txt is too large, remove from tracking
git rm --cached text8_1M.txt
echo "text8_1M.txt" >> .gitignore
git add .gitignore
git commit -m "Remove large dataset file"
```

### Already Initialized Git

If HW5 is already in a git repository (part of CS6140 repo):

**Option A: Keep as separate standalone repo (Recommended)**
```bash
# Remove connection to parent repo
rm -rf .git

# Reinitialize fresh
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/neural-network-language-model.git
git push -u origin main
```

**Option B: Use git subtree (Advanced)**
```bash
# From CS6140 parent directory
cd /Users/yashlad/Development/CS6140
git subtree push --prefix=HW5 https://github.com/YOUR_USERNAME/neural-network-language-model.git main
```

---

## After Pushing

### 1. Add Topics/Tags
On GitHub repository page:
- Click ⚙️ (Settings) next to "About"
- Add topics: `deep-learning`, `neural-networks`, `pytorch`, `numpy`, `word-embeddings`, `autoencoder`, `nlp`, `machine-learning`, `python`

### 2. Enable GitHub Pages (Optional)
If you want to host documentation:
- Settings → Pages
- Source: Deploy from branch `main`
- Folder: `/docs` or root

### 3. Add Badges to README (Optional)
Already included in README.md:
- Python version badge
- PyTorch badge
- NumPy badge
- License badge

### 4. Create Releases (Optional)
Tag versions for major milestones:
```bash
git tag -a v1.0.0 -m "Release v1.0.0: Complete neural network implementations"
git push origin v1.0.0
```

---

## Quick Copy-Paste Commands

**All commands in one block** (replace `YOUR_USERNAME`):

```bash
# Navigate to HW5
cd /Users/yashlad/Development/CS6140/HW5

# Initialize and commit
git init
git add .
git commit -m "Initial commit: Neural Network Language Model & Optimization Engine"

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/neural-network-language-model.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Files Created for GitHub

Your repository now includes:

✅ **README.md** - Comprehensive project documentation with badges
✅ **LICENSE** - MIT License
✅ **SETUP.md** - Installation and setup guide
✅ **requirements.txt** - Python dependencies
✅ **.gitignore** - Comprehensive ignore patterns
✅ **PROBLEM_SUMMARIES.txt** - Detailed problem descriptions
✅ **HW5_RESUME_PROJECT.txt** - Resume-ready project summary
✅ **ALGORITHM_EXPLANATION.md** - In word_vectors_nn/ directory
✅ **All notebooks** - P1-P4 and word_vectors_nn notebooks
✅ **Python scripts** - GPU-accelerated training scripts

---

## Repository Structure Preview

```
neural-network-language-model/
├── 📄 README.md              ⭐ Main landing page
├── 📄 LICENSE                ⭐ MIT License
├── 📄 SETUP.md              ⭐ Setup instructions
├── 📄 requirements.txt       ⭐ Dependencies
├── 📄 PROBLEM_SUMMARIES.txt
├── 📄 HW5_RESUME_PROJECT.txt
│
├── 📓 Notebooks
│   ├── P1_Autoencoder.ipynb
│   ├── P2_torch_AutoEnc.ipynb
│   ├── P3_wine_selfimpl.ipynb
│   └── P4_Wine_Pytorch.ipynb
│
├── 📂 word_vectors_nn/
│   ├── README.md
│   ├── ALGORITHM_EXPLANATION.md
│   ├── P7_Manual_Text8.ipynb
│   ├── P8_PyTorch_Text8.ipynb
│   ├── Problem8_PyTorch_Text8_M4_MPS.py
│   └── Problem8_PyTorch_20NG_M4_MPS.py
│
├── 📊 Datasets
│   ├── text8_50K.txt
│   ├── text8_200K.txt
│   ├── text8_750K.txt
│   ├── text8_1M.txt
│   └── wine_train.data
│
└── 🙈 .gitignore            ⭐ Ignore patterns
```

---

## Success Checklist

- [ ] Created GitHub repository
- [ ] Initialized git in HW5 directory
- [ ] Added and committed all files
- [ ] Connected to GitHub remote
- [ ] Pushed to main branch
- [ ] Verified files on GitHub
- [ ] README displays correctly
- [ ] Added repository topics/tags
- [ ] (Optional) Created release tag
- [ ] (Optional) Updated profile with project link

---

**Ready to push!** 🚀

If you encounter any issues, refer to the troubleshooting section above.
