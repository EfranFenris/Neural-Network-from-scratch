# train_exp/AE_process_test.py
import os, sys, argparse, inspect
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- project import path ---
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dataset as ds
from models_ae import Autoencoder

OUT_DIR_GRAPH = os.path.join(ROOT, "graphs_exp")
OUT_DIR_DATA  = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR_GRAPH, exist_ok=True)
os.makedirs(OUT_DIR_DATA,  exist_ok=True)

# ---------------------------
# Utils for feature plotting
# ---------------------------
def _to_in_out(W, D_in):
    """Return weights shaped [D_in, H] (columns are neurons). Handles [D_in,H] or [H,D_in]."""
    return W if W.shape[0] == D_in else W.t()

def _bn_scale(model, layer_idx):
    """Effective per-feature scale for BN at hidden layer 'layer_idx' (encoder side)."""
    if not getattr(model, "gamma", None):
        # No BN anywhere
        H = _to_in_out(model.W[layer_idx], model.sizes[0]).shape[1]
        return torch.ones(H, device=model.W[layer_idx].device, dtype=model.W[layer_idx].dtype)
    g  = model.gamma[layer_idx]
    rv = model.running_var[layer_idx]
    eps = model.eps
    return g / torch.sqrt(rv + eps)  # [H]

def _standardize(img_1d):
    m, s = img_1d.mean(), img_1d.std()
    if s == 0: 
        return np.zeros_like(img_1d)
    z = (img_1d - m) / (3*s)
    return np.clip(z + 0.5, 0, 1)

def _save_grid(patches_DxK, H, W, title, path, cols=16):
    K = patches_DxK.shape[1]
    rows = int(np.ceil(K / cols))
    plt.figure(figsize=(cols*1.0, rows*1.0))
    for j in range(K):
        ax = plt.subplot(rows, cols, j + 1)
        ax.imshow(_standardize(patches_DxK[:, j]).reshape(H, W), cmap="gray", interpolation="nearest")
        ax.axis("off")
    plt.suptitle(title, y=0.99, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

def _save_recon_grid(X, Yhat, out_path, n_cols=12, img_hw=(28, 28), seed=0):
    H, W = img_hw
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=n_cols, replace=False)
    xs, ys = X[idx], Yhat[idx]
    plt.figure(figsize=(1.2 * n_cols, 2.6))
    for i in range(n_cols):
        ax = plt.subplot(2, n_cols, i + 1)
        ax.imshow(xs[i].reshape(H, W), cmap="gray", interpolation="nearest")
        ax.axis("off")
        if i == 0: ax.set_title("Original", fontsize=9)
        ax = plt.subplot(2, n_cols, n_cols + i + 1)
        ax.imshow(ys[i].reshape(H, W), cmap="gray", interpolation="nearest")
        ax.axis("off")
        if i == 0: ax.set_title("Reconstruction", fontsize=9)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved: {out_path}")

# ---------------------------
# Load dataset
# ---------------------------
def load_test_split(dataset_name: str):
    if dataset_name == "fashion":
        Xtr, ytr, Xte, yte = ds.load_fashion_mnist_processed(os.path.join(ROOT, "data/processed"))
    else:
        Xtr, ytr, Xte, yte = ds.load_mnist_processed(os.path.join(ROOT, "data/processed"))
    return Xte.float(), np.array(yte, dtype=np.int64)

# ---------------------------
# Rebuild AE from checkpoint
# ---------------------------
def load_ae(ckpt_path: str) -> tuple[Autoencoder, dict]:
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg, D = ck["meta"]["config"], ck["meta"]["input_dim"]
    # Signature-safe kwargs (only pass what Autoencoder.__init__ accepts)
    cand = dict(
        input_dim=D,
        enc_dims=cfg["enc_dims"],
        dec_dims=cfg["dec_dims"],
        init=cfg.get("init", "he"),
        activation=cfg.get("act", "relu"),
        use_batchnorm=cfg.get("bn", True),
        eps=1e-5,
        momentum=0.9,
        device="cpu",
        dtype=torch.float32,
    )
    sig = inspect.signature(Autoencoder)
    kwargs = {k: v for k, v in cand.items() if k in sig.parameters}
    ae = Autoencoder(**kwargs)
    # Load parameters (incl. BN if present)
    st = ck["state"]
    for i in range(len(ae.W)):
        ae.W[i].copy_(st["W"][i]); ae.b[i].copy_(st["b"][i])
    if getattr(ae, "gamma", None) and "gamma" in st:
        for i in range(len(ae.gamma)):
            ae.gamma[i].copy_(st["gamma"][i]); ae.beta[i].copy_(st["beta"][i])
            ae.running_mean[i].copy_(st["running_mean"][i]); ae.running_var[i].copy_(st["running_var"][i])
    return ae, ck["meta"]

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "saved_models", "ae_fashion_best.pt"))
    ap.add_argument("--dataset", choices=["fashion", "mnist"], default="fashion")
    ap.add_argument("--tag", default="", help="Optional suffix for NPZ filename, e.g. _base or _reg")
    args = ap.parse_args()

    # 1) Load test set
    Xte, yte = load_test_split(args.dataset)
    D = Xte.shape[1]
    print(f"Dataset={args.dataset} | test={tuple(Xte.shape)}  D={D}")

    # 2) Load trained AE
    ae, meta = load_ae(args.ckpt)
    assert meta["input_dim"] == D, f"Checkpoint D={meta['input_dim']} but test D={D}"
    print(f"Loaded AE from {args.ckpt} | sizes={ae.sizes}")

    # 3) Encode + reconstruct (test set)
    with torch.no_grad():
        Z   = ae.encode(Xte)                               # [N, K]
        Yhat, _ = ae.forward(Xte, training=False)          # [N, D]
        mse = ((Yhat - Xte)**2).mean(dim=1).cpu().numpy()  # per-image MSE
    Z_np    = Z.cpu().numpy()
    Yhat_np = Yhat.clamp(0, 1).cpu().numpy()  # clamp for viewing convenience

    # 4) Save compact NPZ (reusable for UMAP or comparisons)
    out_npz = os.path.join(OUT_DIR_DATA, f"ae_{args.dataset}_test_processed{args.tag}.npz")
    np.savez_compressed(out_npz,
        dataset=args.dataset, Z=Z_np, Yhat=Yhat_np, mse=mse, y=yte)
    print(f"Saved: {out_npz}")

    # 5) (Part 1.2) Plot hidden-layer features as images
    H_img, W_img = 28, 28  # Fashion/MNIST
    # Layer 1 effective filters (include BN scaling)
    W0_in  = _to_in_out(ae.W[0], D)              # [D, H1]
    s0     = _bn_scale(ae, 0)                    # [H1]
    W0_eff = (W0_in * s0)                        # broadcast columns -> [D, H1]
    _save_grid(W0_eff.cpu().numpy(), H_img, W_img,
               "Encoder Layer 1 filters (with BN scaling)",
               os.path.join(OUT_DIR_GRAPH, f"ae_{args.dataset}_filters_l1.png"))

    # Layer 2 backprojected to input space: W0_eff @ (W1_in * s1)
    if len(ae.W) >= 2:
        W1_in  = _to_in_out(ae.W[1], W0_in.shape[1])  # [H1, H2]
        s1     = _bn_scale(ae, 1)
        W1_eff = (W1_in * s1)                         # [H1, H2]
        backproj = (W0_eff @ W1_eff).cpu().numpy()    # [D, H2]
        _save_grid(backproj, H_img, W_img,
                   "Encoder Layer 2 filters (backprojected)",
                   os.path.join(OUT_DIR_GRAPH, f"ae_{args.dataset}_filters_l2_backproj.png"), cols=16)

    # 6) (Nice extra) Recon grid for quick visual sanity check
    _save_recon_grid(Xte.cpu().numpy(), Yhat_np,
                     os.path.join(OUT_DIR_GRAPH, f"ae_{args.dataset}_recon_grid.png"),
                     n_cols=12, img_hw=(H_img, W_img))

if __name__ == "__main__":
    main()