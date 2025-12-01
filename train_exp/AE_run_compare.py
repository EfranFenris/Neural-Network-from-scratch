import os, sys, subprocess, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(ROOT, "graphs_exp")
os.makedirs(OUT_DIR, exist_ok=True)

PY = sys.executable  # use your venv's python

def run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    Z = data["Z"]
    mse = data["mse"]
    y = data["y"] if "y" in data else None
    if y is None or (isinstance(y, np.ndarray) and y.shape == ()):
        y = None
    dsname = data["dataset"].item() if hasattr(data["dataset"], "item") else data["dataset"]
    return Z, mse, y, dsname

def ensure_processed(ckpt_path, tag):
    run([PY, os.path.join(ROOT, "train_exp", "AE_process_test.py"),
         "--ckpt", ckpt_path, "--tag", tag])

def umap2(Z, n_neighbors=15, min_dist=0.1, seed=0):
    try:
        import umap.umap_ as umap
    except Exception:
        import umap  # some installs expose it this way
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        metric="euclidean", random_state=seed)
    return reducer.fit_transform(Z)

def plot_umap_side_by_side(Zb2, yb, Zr2, yr, out_path):
    plt.figure(figsize=(10,4.2))
    ax = plt.subplot(1,2,1)
    if yb is not None:
        sc = ax.scatter(Zb2[:,0], Zb2[:,1], c=yb, s=4, cmap="tab10", alpha=0.8)
        plt.colorbar(sc, ax=ax, ticks=range(10))
    else:
        ax.scatter(Zb2[:,0], Zb2[:,1], s=4, alpha=0.8)
    ax.set_title("Baseline UMAP"); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    ax = plt.subplot(1,2,2)
    if yr is not None:
        sc = ax.scatter(Zr2[:,0], Zr2[:,1], c=yr, s=4, cmap="tab10", alpha=0.8)
        plt.colorbar(sc, ax=ax, ticks=range(10))
    else:
        ax.scatter(Zr2[:,0], Zr2[:,1], s=4, alpha=0.8)
    ax.set_title("Regularized UMAP"); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved: {out_path}")

def plot_mse_box(mse_base, mse_reg, out_path):
    plt.figure(figsize=(5.5,4))
    plt.boxplot([mse_base, mse_reg], labels=["baseline", "regularized"], showfliers=False)
    plt.ylabel("per-image MSE")
    plt.title("Reconstruction error distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved: {out_path}")

def recon_grid(X, Yhat, out_path, n_cols=12, H=28, W=28):
    N = X.shape[0]
    idx = np.random.RandomState(0).choice(N, size=n_cols, replace=False)
    xs = X[idx]; ys = Yhat[idx]
    plt.figure(figsize=(1.2*n_cols, 2.6))
    for i in range(n_cols):
        ax = plt.subplot(2, n_cols, i+1)
        ax.imshow(xs[i].reshape(H,W), cmap="gray", interpolation="nearest")
        ax.axis("off")
        if i == 0: ax.set_title("Original", fontsize=9)
        ax = plt.subplot(2, n_cols, n_cols+i+1)
        ax.imshow(ys[i].reshape(H,W), cmap="gray", interpolation="nearest")
        ax.axis("off")
        if i == 0: ax.set_title("Reconstruction", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved: {out_path}")

def main():
    # 1) Ensure both processed NPZ exist (baseline + regularized)
    base_ckpt = os.path.join(ROOT, "saved_models", "ae_fashion_best.pt")
    reg_ckpt  = os.path.join(ROOT, "saved_models", "ae_fashion_best_reg.pt")

    ensure_processed(base_ckpt, "_base")
    ensure_processed(reg_ckpt,  "_reg")

    out_base = os.path.join(ROOT, "outputs", "ae_fashion_test_processed_base.npz")
    out_reg  = os.path.join(ROOT, "outputs", "ae_fashion_test_processed_reg.npz")

    # 2) Load NPZ
    Zb, mse_b, yb, ds_b = load_npz(out_base)
    Zr, mse_r, yr, ds_r = load_npz(out_reg)
    assert ds_b == ds_r, "Datasets differ between runs"

    # 3) UMAPs
    Zb2 = umap2(Zb, seed=0)
    Zr2 = umap2(Zr, seed=0)
    plot_umap_side_by_side(Zb2, yb, Zr2, yr,
        os.path.join(OUT_DIR, f"{ds_b}_umap_baseline_vs_reg.png"))

    # 4) MSE boxplot
    plot_mse_box(mse_b, mse_r, os.path.join(OUT_DIR, f"{ds_b}_mse_box.png"))

    # 5) Recon grids for both runs
    # reload the test set for originals
    sys.path.insert(0, ROOT)
    import dataset as ds
    if ds_b == "fashion":
        _Xtr, _ytr, Xte, _yte = ds.load_fashion_mnist_processed(os.path.join(ROOT, "data/processed"))
    else:
        _Xtr, _ytr, Xte, _yte = ds.load_mnist_processed(os.path.join(ROOT, "data/processed"))
    Xte = Xte.numpy()

    # Load Yhat from the npz files
    Yhat_b = np.load(out_base, allow_pickle=True)["Yhat"]
    Yhat_r = np.load(out_reg,  allow_pickle=True)["Yhat"]

    recon_grid(Xte, Yhat_b, os.path.join(OUT_DIR, f"{ds_b}_recon_grid_baseline.png"))
    recon_grid(Xte, Yhat_r, os.path.join(OUT_DIR, f"{ds_b}_recon_grid_regularized.png"))

    # 6) Tiny text summary in console
    print("\n=== Summary ===")
    print(f"Baseline  : mean MSE={mse_b.mean():.4f}")
    print(f"Regularized: mean MSE={mse_r.mean():.4f}")

if __name__ == "__main__":
    main()