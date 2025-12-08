# Neural-Network-from-Scratch

Collection of **from-scratch neural network labs** implemented with PyTorch / NumPy **without using autograd**.  
Forward pass, backward pass, BatchNorm and SGD updates are all coded by hand using basic tensor operations.

- **Lab 1 – Manual MLP:** small feed-forward network for synthetic classification & regression.
- **Lab 2 – Autoencoder + Experiments:** fully-connected autoencoder on Fashion-MNIST, plus UMAP and classifier comparisons.

The goal is **educational clarity**: understand every step of training before relying on high-level abstractions.

---

## 0. Installation & Setup

### Requirements

- Python 3.9+
- PyTorch
- NumPy
- matplotlib
- umap-learn  *(only needed for Lab 2 UMAP plots)*

### (Optional) Virtual environment

    python -m venv .venv
    source .venv/bin/activate        # macOS / Linux
    # .venv\Scripts\activate         # Windows PowerShell

### Install dependencies

    pip install torch numpy matplotlib umap-learn

> Tip: run all scripts from the **shell**, not inside the Python REPL.

---

## 1. Lab 1 – Manual MLP (Classification & Regression)

Manual implementation of a small feed-forward neural network (MLP).  
Supports both **classification** (logits + cross-entropy) and **regression** (raw values + MSE).

### Highlights

- 1–2 hidden layers with configurable width.
- He or Xavier weight initialization.
- ReLU or Sigmoid activations.
- Optional Batch Normalization on hidden layers  
  (manual forward + backward, running mean/var).
- Separate subclasses for:
  - **Classification:** logits + manual softmax cross-entropy.
  - **Regression:** raw values + MSE.
- Mini-batch SGD training loop written by hand.
- Simple synthetic demo datasets for sanity checking.
- Saves learning-curve plots (loss + metric) as PNGs.

### Lab 1 Files

    models.py      # Manual MLP core + classification/regression subclasses
    train.py       # Training/evaluation loops + demo runs and plotting
    Hola.py        # Simple hello-world script (not part of the NN code)
    *.png          # Generated plots from previous runs (learning curves)
    README.md      # This documentation (top-level file)

*(If Lab 1 lives in a subfolder, these paths are relative to that folder.)*

### Running the Lab 1 demos

    python train.py

This will:

1. Build small synthetic datasets.
2. Train a classification model and a regression model.
3. Save learning curves:
   - `clf_curves_loss.png`, `clf_curves_acc.png`
   - `reg_curves_loss.png`, `reg_curves_mse.png`

### How Lab 1 works (short version)

`models.py` defines a base class that manually performs:

- **Linear layers:** `h @ W + b`
- **BatchNorm (optional):** per-batch mean/var, plus running estimates.
- **Activations:** ReLU or Sigmoid with custom backward.
- **Outputs:**
  - Classification → logits → stable softmax + cross-entropy.
  - Regression → raw continuous outputs.
- **Backward pass:** explicit gradient propagation and accumulation.
- **Parameter updates:** vanilla SGD, `param -= lr * grad`.

Losses:

- **Classification:** numerically stable softmax cross-entropy (manual gradient).
- **Regression:** Mean Squared Error.

---

## 2. Lab 2 – Autoencoder + Experiments (Fashion-MNIST)

From-scratch implementation of a fully-connected Autoencoder (AE) and a small classifier, with utilities to:

- Train the AE over a grid of hyperparameters (with early stopping + optional L2).
- Process the test set with a pre-trained AE.
- Visualize learned features (first-layer filters and second-layer back-projections).
- Run UMAP on the latent codes (colored by true labels).
- Compare classifiers trained on raw pixels vs AE latent codes.

All figures are written to `graphs_exp/`. Arrays for reuse go to `outputs/`.  
Checkpoints live in `saved_models/`.

### 2.1 Folder Map (essentials)

    data/                     # processed Fashion-MNIST (28x28 -> 784)
    graphs_exp/               # all generated PNGs
    outputs/                  # NPZ artifacts (latent Z, reconstructions, MSE, labels)
    saved_models/             # checkpoints (ae_fashion_best.pt, ae_fashion_best_reg.pt)

    train_exp/
      AE_experiments.py       # train AE, pick best config, save checkpoint + plots
      AE_experiments_regu.py  # (optional) variant with stronger regularization
      AE_process_test.py      # process test set + feature images + recon grid
      AE_UMAP.py              # UMAP on latent codes from NPZ
      AE_run_compare.py       # convenience runner: baseline vs regularized AE
      FT_compare_raw_vs_ae.py # classifier on RAW pixels vs AE latent (verbose curves)

    models_ae.py              # AE core (manual forward/backward, BN, SGD)
    models.py                 # shared MLP/block pieces (init, activations, BN) + classifier
    dataset.py                # dataset loaders / helpers for processed Fashion-MNIST
    README.md                 # this file

---

## 3. Lab 2 – Step-by-Step Usage

### A Train the Autoencoder (select best config)

    python train_exp/AE_experiments.py

**Outputs**

- `saved_models/ae_fashion_best.pt`
- `graphs_exp/ae_fashion_recon_grid.png` – originals vs reconstructions.
- `graphs_exp/ae_fashion_loss.png` – train/val reconstruction curves.

The script prints `best val_recon`, `test_recon`, `RMSE`, and `PSNR` to stdout.

- Uses **early stopping** (keeps the epoch with best validation loss).
- L2 **weight decay** is controlled via a `weight_decay` argument in `run_experiments`.

---

### B Process the test set + visualize features

    python train_exp/AE_process_test.py \
      --ckpt saved_models/ae_fashion_best.pt \
      --dataset fashion

Produces `outputs/ae_fashion_test_processed.npz` with:

- `Z` `[N, K]` – latent codes.  
- `Yhat` `[N, 784]` – reconstructions (clamped to `[0, 1]` for viewing).  
- `mse` `[N]` – per-image reconstruction MSE.  
- `y` – test labels (`0..9`).  
- `dataset` – `"fashion"`.

And the following figures in `graphs_exp/`:

- `ae_fashion_filters_l1.png` — layer-1 filters.  
- `ae_fashion_filters_l2_backproj.png` — layer-2 filters back-projected to input space.  
- `ae_fashion_recon_grid.png` — originals vs reconstruction grid.

---

### C UMAP of the latent space

    python train_exp/AE_UMAP.py

Reads the NPZ from step B and writes:

- `graphs_exp/ae_fashion_umap.png`

UMAP is a 2-D embedding of `Z` colored by Fashion-MNIST labels.

---

### D (Optional) Baseline vs Regularized AE

If you trained a second checkpoint, e.g. `saved_models/ae_fashion_best_reg.pt`:

    python train_exp/AE_run_compare.py

This will:

- Process both checkpoints (NPZs tagged `_base` and `_reg`).
- Produce side-by-side UMAPs, an MSE boxplot, and reconstruction grids in `graphs_exp/`.

---

### E (Optional) Classifier: RAW pixels vs AE latent

    python train_exp/FT_compare_raw_vs_ae.py

Generates:

- `graphs_exp/ft_val_acc_fashion_verbose.png` — validation accuracy curves.  
- `graphs_exp/ft_test_acc_fashion_verbose.png` — final test accuracies (bar chart).

This runs two classifiers with the **same architecture**:

1. On raw pixels `x ∈ ℝ⁷⁸⁴`.  
2. On AE latent codes `z ∈ ℝᴷ`.

Good for checking when encoded representations help.

---

## 4. Reading Lab-2 Plots

### L1 filters – `ae_fashion_filters_l1.png`

To visualize the first-layer weights:

1. Take each column of the first weight matrix  
   \( W_0 \in \mathbb{R}^{784 \times H_1} \).
2. Apply BatchNorm scale: \( \gamma / \sqrt{\text{var} + \varepsilon} \).
3. Reshape each resulting length-784 vector to **28×28**.
4. Render as grayscale:
   - Bright = positive weight (excites neuron).
   - Dark   = negative weight (inhibits neuron).

Number of images = **H₁** (number of layer-1 neurons).

---

### L2 filters (back-projected) – `ae_fashion_filters_l2_backproj.png`

Columns of  
\( W_1 \in \mathbb{R}^{H_1 \times H_2} \)  
live in layer-1 space. To view them in input space:

    W0_eff = W0 * (γ0 / √(var0 + ε))          # column-wise scale
    W1_eff = W1 * (γ1 / √(var1 + ε))
    backproj[:, j] = W0_eff @ W1_eff[:, j]    ∈ ℝ⁷⁸⁴

Each `backproj[:, j]` is reshaped to 28×28 and plotted.  
Number of images = **H₂**.

---

### UMAP – `ae_fashion_umap.png`

2-D embedding of `Z`, colored by Fashion-MNIST labels:

- 0 T-shirt/top  
- 1 Trouser  
- 2 Pullover  
- 3 Dress  
- 4 Coat  
- 5 Sandal  
- 6 Shirt  
- 7 Sneaker  
- 8 Bag  
- 9 Ankle boot  

Tighter, better-separated clusters → more class-aware latent space.

---

### Reconstruction grid

Side-by-side originals and reconstructions.  
Quick sanity check that the autoencoder has learned a sensible mapping.

---

### Loss curves – `*_loss.png`

- `train_recon`: average reconstruction MSE on the **training** split per epoch.
- `val_recon`: average reconstruction MSE on a held-out **validation** split.

Checkpoint selection: epoch with **lowest `val_recon`** (early stopping).  
`test_recon` is computed once on the test split using that best model.

---

## 5. Dimensions Cheat-Sheet (Lab 2)

- **Input image:**  
  \( x \in \mathbb{R}^{784} \) (flattened 28×28).

- **Encoder layer 1:**  
  \( W_0 \in \mathbb{R}^{784 \times H_1},\; b_0 \in \mathbb{R}^{H_1} \)  
  → activations \( h_1 \in \mathbb{R}^{H_1} \).  
  `H1` = number of L1 filter images.

- **Encoder layer 2 (latent):**  
  \( W_1 \in \mathbb{R}^{H_1 \times H_2} \)  
  → latent code \( z \in \mathbb{R}^{H_2} \).  
  `H2` = latent dimension = number of L2 back-projected filters.

- **Decoder:** mirrors the encoder back to 784.

---

## 6. BatchNorm: “Running Stats” in Plain English

**During training**

For each hidden layer:

    running_mean := m * running_mean + (1 - m) * batch_mean
    running_var  := m * running_var  + (1 - m) * batch_var

The batch is normalized, then scaled and shifted using learnable `γ` and `β`.

**During evaluation / test processing**

BatchNorm uses the **running** mean and variance instead of per-batch stats,  
which avoids stochastic shifts in outputs at test time.

---

## 7. Hyperparameters (Lab 2)

- `enc_dims`, `dec_dims` – encoder/decoder hidden sizes; last encoder size = latent **K**  
  (e.g. `enc_dims=[256, 128]`, `dec_dims=[256]`).
- `activation` – `"relu"` or `"sigmoid"` for hidden layers.
- `init` – `"he"` (good for ReLU) or `"xavier"` (good for sigmoid/tanh).
- `use_batchnorm` – whether to apply BatchNorm on hidden layers.
- `lr`, `batch` – SGD learning rate and batch size.
- `weight_decay` – L2 regularization strength (optional).

A “regularized” AE typically uses stronger L2, smaller K, or heavier dropout  
(here we focus on L2 + early stopping).

---

## 8. RAW vs ENCODED – When Does Encoded Win?

Autoencoder latent codes can outperform raw pixels when:

- **Labels are limited or noisy** → AE denoises and compresses structure.
- **Classifier capacity is small** → `z ∈ ℝᴷ` (e.g. K = 128) is easier than `x ∈ ℝ⁷⁸⁴`.
- **AE is well-regularized** → latent space clusters align with labels (see UMAP).

`FT_compare_raw_vs_ae.py` trains **the same classifier architecture** on both inputs  
for a fair comparison.

---

## 9. Commands Recap

    # ---- Lab 1 (MLP) ----
    python train.py

    # ---- Lab 2 (Autoencoder) ----
    # Train AE (selects best by val_recon)
    python train_exp/AE_experiments.py

    # Process test set + features + recon grid
    python train_exp/AE_process_test.py --ckpt saved_models/ae_fashion_best.pt --dataset fashion

    # UMAP of latent codes
    python train_exp/AE_UMAP.py

    # (Optional) Compare two AEs (baseline vs regularized)
    python train_exp/AE_run_compare.py

    # (Optional) Classifier on raw vs encoded
    python train_exp/FT_compare_raw_vs_ae.py