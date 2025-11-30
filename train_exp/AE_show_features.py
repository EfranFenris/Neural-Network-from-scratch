# train_exp/AE_features_umap.py
import os, sys, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import dataset as ds
from models_ae import Autoencoder

CKPT = os.path.join(ROOT, "saved_models", "ae_fashion_best.pt")
OUTD = os.path.join(ROOT, "graphs_exp"); os.makedirs(OUTD, exist_ok=True)

# ---------- utils ----------
def _to_in_out(W, D):
    """Return W_in shaped [D, H] (columns are neurons). Handles [D,H] or [H,D]."""
    return W if W.shape[0] == D else W.t()

def _bn_scale(model, layer_idx):
    if not getattr(model, "gamma", None):   # no BN anywhere
        H = _to_in_out(model.W[layer_idx], model.sizes[0]).shape[1]
        return torch.ones(H)
    g = model.gamma[layer_idx]
    rv = model.running_var[layer_idx]
    eps = model.eps
    return g / torch.sqrt(rv + eps)  # [H]

def _standardize(img):
    m, s = img.mean(), img.std()
    if s == 0: return np.zeros_like(img)
    z = (img - m) / (3*s)
    return np.clip((z + 0.5), 0, 1)  # ~[-1.5,1.5] -> [0,1]

def _save_grid(patches, H, W, title, path, cols=16):
    K = patches.shape[1]
    rows = int(np.ceil(K / cols))
    plt.figure(figsize=(cols*1.0, rows*1.0))
    for j in range(K):
        ax = plt.subplot(rows, cols, j+1)
        ax.imshow(_standardize(patches[:, j]).reshape(H, W), cmap="gray", interpolation="nearest")
        ax.axis("off")
    plt.suptitle(title, y=0.99, fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

def load_ckpt(path):
    ck = torch.load(path, map_location="cpu")
    cfg, D = ck["meta"]["config"], ck["meta"]["input_dim"]
    ae = Autoencoder(
        input_dim=D,
        enc_dims=cfg["enc_dims"], dec_dims=cfg["dec_dims"],
        init=cfg["init"], activation=cfg["act"],
        use_batchnorm=cfg["bn"], bn_where=cfg["bn_where"],
        out_activation=cfg["out_act"], loss_mode=cfg["loss"],
    )
    # load all params (incl. BN)
    st = ck["state"]
    for i in range(len(ae.W)):
        ae.W[i].copy_(st["W"][i]); ae.b[i].copy_(st["b"][i])
    if getattr(ae, "gamma", None) and "gamma" in st:
        for i in range(len(ae.gamma)):
            ae.gamma[i].copy_(st["gamma"][i]); ae.beta[i].copy_(st["beta"][i])
            ae.running_mean[i].copy_(st["running_mean"][i]); ae.running_var[i].copy_(st["running_var"][i])
    return ae, ck["meta"]

@torch.no_grad()
def main():
    ae, meta = load_ckpt(CKPT)
    D, H0, H1 = meta["input_dim"], ae.sizes[1], ae.sizes[2]  # sizes: [D, h1, h2, ..., D]
    H, W = 28, 28
    print("sizes:", ae.sizes, "| bn_where:", getattr(ae, "bn_where", "none"))

    # ---------- Part 2: show features as images ----------
    # Layer 1 effective filters (include BN scaling if BN on encoder)
    W0_in = _to_in_out(ae.W[0], D)                # [D, H1]
    s0 = _bn_scale(ae, 0)                         # [H1]
    W0_eff = (W0_in * s0)                         # broadcast on columns, [D, H1]
    _save_grid(W0_eff.cpu().numpy(), H, W,
               "Encoder Layer 1 filters (with BN scaling)",
               os.path.join(OUTD, "ae_fashion_filters_l1.png"))

    # Layer 2 backprojected to input space: (W0_eff) @ (W1_in * s1)
    W1_in = _to_in_out(ae.W[1], W0_in.shape[1])   # [H1, H2]
    s1 = _bn_scale(ae, 1) if len(ae.W) > 1 else torch.ones(W1_in.shape[1])
    W1_eff = (W1_in * s1)                         # [H1, H2]
    backproj = (W0_eff @ W1_eff).cpu().numpy()    # [D, H2]
    _save_grid(backproj, H, W,
               "Encoder Layer 2 filters (backprojected)",
               os.path.join(OUTD, "ae_fashion_filters_l2_backproj.png"), cols=16)

    # ---------- Part 3: UMAP on latent codes ----------
    # Try to reuse processed outputs; else compute on the fly.
    npz_path = os.path.join(ROOT, "outputs", f"ae_{meta['dataset']}_test_processed.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        Z = data["Z"]; y = data["y"] if "y" in data and data["y"] is not None else None
        print(f"Loaded latent codes from {npz_path}: Z={Z.shape}")
    else:
        print("Processed npz not found; encoding test split now…")
        if meta["dataset"] == "fashion":
            _, _, Xte, y = ds.load_fashion_mnist_processed(os.path.join(ROOT, "data/processed"))
        else:
            _, _, Xte, y = ds.load_mnist_processed(os.path.join(ROOT, "data/processed"))
        Z = ae.encode(Xte.float()).cpu().numpy()

    try:
        import umap
    except Exception:
        raise SystemExit("Install umap-learn: pip install umap-learn")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0)
    Z2 = reducer.fit_transform(Z)  # [N,2]

    plt.figure(figsize=(6,5))
    if y is not None:
        y = np.array(y).astype(int).ravel()
        sc = plt.scatter(Z2[:,0], Z2[:,1], c=y, s=5, cmap="tab10", alpha=0.8)
        cbar = plt.colorbar(sc, ticks=range(10)); cbar.set_label("Fashion-MNIST label")
    else:
        plt.scatter(Z2[:,0], Z2[:,1], s=5, alpha=0.8)
    plt.title("UMAP of AE latent codes (test set)")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.tight_layout()
    out_umap = os.path.join(OUTD, "ae_fashion_umap.png")
    plt.savefig(out_umap, dpi=150); plt.close()
    print(f"Saved: {out_umap}")

if __name__ == "__main__":
    main()