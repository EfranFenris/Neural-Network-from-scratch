# train_exp/AE_umap.py
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_NPZ = os.path.join(ROOT, "outputs", "ae_fashion_test_processed.npz")
OUT_PNG = os.path.join(ROOT, "graphs_exp", "ae_fashion_umap.png")

def main():
    if not os.path.exists(OUT_NPZ):
        raise FileNotFoundError(f"No existe {OUT_NPZ}. Ejecuta antes AE_process_test.py")

    data = np.load(OUT_NPZ, allow_pickle=True)
    Z = data["Z"]                       # [N, K]
    y = data["y"]
    y = None if y is None or y.shape == () else y

    import umap.umap_ as umap
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0
    )
    Z2 = reducer.fit_transform(Z)       # [N, 2]

    plt.figure(figsize=(7, 6))
    if y is not None:
        scatter = plt.scatter(Z2[:,0], Z2[:,1], c=y, s=3, cmap="tab10", alpha=0.8)
        cbar = plt.colorbar(scatter, ticks=range(10)); cbar.set_label("label")
    else:
        plt.scatter(Z2[:,0], Z2[:,1], s=3, alpha=0.8)
    plt.title("UMAP of AE latent codes (test set)")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150); plt.close()
    print(f"Saved: {OUT_PNG}")

if __name__ == "__main__":
    main()