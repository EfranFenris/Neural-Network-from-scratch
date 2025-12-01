# train_exp/AE_experiments.py
import os, sys, math, inspect
import torch
from typing import List, Dict

# --- project import path ---
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dataset as ds
from models_ae import Autoencoder

# --- headless-safe plotting ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------
# Data loading (X only)
# -----------------------
def get_data(dataset_name: str = "fashion", val_ratio: float = 0.1, seed: int = 0):
    torch.manual_seed(seed)
    if dataset_name == "fashion" and hasattr(ds, "load_fashion_mnist_for_ae"):
        Xtr, Xval, Xte, *_ = ds.load_fashion_mnist_for_ae(
            processed_dir=os.path.join(ROOT, "data/processed"),
            val_ratio=val_ratio, seed=seed,
        )
        return Xtr.float(), Xval.float(), Xte.float()

    if dataset_name == "mnist" and hasattr(ds, "load_mnist_processed"):
        Xtr_full, _ytr, Xte, _yte = ds.load_mnist_processed(
            processed_dir=os.path.join(ROOT, "data/processed")
        )
        N = Xtr_full.shape[0]
        idx = torch.randperm(N)
        n_val = int(val_ratio * N)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        return Xtr_full[tr_idx].float(), Xtr_full[val_idx].float(), Xte.float()

    # default: fashion preprocessed
    Xtr, _ytr, Xte, _yte = ds.load_fashion_mnist_processed(
        processed_dir=os.path.join(ROOT, "data/processed")
    )
    N = Xtr.shape[0]
    idx = torch.randperm(N)
    n_val = int(val_ratio * N)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return Xtr[tr_idx].float(), Xtr[val_idx].float(), Xte.float()

# ---------------------------------
# Batching / train / eval
# ---------------------------------
def batches(X: torch.Tensor, batch_size: int, shuffle=True):
    N = X.shape[0]
    idx = torch.randperm(N) if shuffle else torch.arange(N)
    for i in range(0, N, batch_size):
        yield X[idx[i:i+batch_size]]

def train_ae_one_epoch(model: Autoencoder, X: torch.Tensor,
                       lr: float, batch_size: int, weight_decay: float = 0.0):
    model.zero_grad()
    total_loss, total_count = 0.0, 0
    for xb in batches(X, batch_size, shuffle=True):
        x_hat, cache = model.forward(xb, training=True)
        loss, dY = model.loss_and_grad(x_hat, xb)

        if weight_decay > 0:
            l2 = 0.0
            for W in model.W:
                l2 += (W * W).sum()
            loss = loss + weight_decay * l2 / X.shape[0]

        model.backward(xb, cache, dY)
        model.step(lr=lr)
        total_loss += float(loss.item()) * xb.shape[0]
        total_count += xb.shape[0]
    return total_loss / total_count

@torch.no_grad()
def eval_ae(model: Autoencoder, X: torch.Tensor, batch_size: int = 512):
    model.zero_grad()
    total_loss, total_count = 0.0, 0
    for xb in batches(X, batch_size, shuffle=False):
        x_hat, _ = model.forward(xb, training=False)  # BN uses running stats if present
        loss, _ = model.loss_and_grad(x_hat, xb)
        total_loss += float(loss.item()) * xb.shape[0]
        total_count += xb.shape[0]
    return total_loss / total_count

# ---------------------------------
# Save / load state (parameters)
# ---------------------------------
def ae_state_dict(model: Autoencoder) -> Dict:
    d = {
        "sizes": getattr(model, "sizes", []),
        "activation": getattr(model, "activation_name", ""),
        "use_bn": getattr(model, "use_bn", False),
        "bn_where": getattr(model, "bn_where", "encoder"),
        "out_activation": getattr(model, "out_activation", "linear"),
        "loss_mode": getattr(model, "loss_mode", "mse"),
        "eps": getattr(model, "eps", 1e-5),
        "momentum": getattr(model, "momentum", 0.9),
        "W": [w.detach().cpu().clone() for w in model.W],
        "b": [b.detach().cpu().clone() for b in model.b],
    }
    if getattr(model, "gamma", None):
        d["gamma"] = [g.detach().cpu().clone() for g in model.gamma]
        d["beta"]  = [be.detach().cpu().clone() for be in model.beta]
        d["running_mean"] = [m.detach().cpu().clone() for m in model.running_mean]
        d["running_var"]  = [v.detach().cpu().clone() for v in model.running_var]
    return d

def ae_load_state_dict(model: Autoencoder, state: Dict):
    for i in range(len(model.W)):
        model.W[i].copy_(state["W"][i]); model.b[i].copy_(state["b"][i])
    if getattr(model, "gamma", None) and "gamma" in state:
        for i in range(len(model.gamma)):
            model.gamma[i].copy_(state["gamma"][i])
            model.beta[i].copy_(state["beta"][i])
            model.running_mean[i].copy_(state["running_mean"][i])
            model.running_var[i].copy_(state["running_var"][i])

# ---------------------------------
# Minimal visuals
# ---------------------------------
def _apply_out_act(Y: torch.Tensor, out_act: str) -> torch.Tensor:
    if out_act == "sigmoid": return torch.sigmoid(Y)
    if out_act == "tanh":    return torch.tanh(Y)
    return Y

def _save_recon_grid(model: Autoencoder, X: torch.Tensor, out_path: str,
                     n_cols: int = 12, img_hw=(28, 28)):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with torch.no_grad():
            idx = torch.randperm(X.shape[0])[:n_cols]
            Xs = X[idx]
            Y, _ = model.forward(Xs, training=False)
            out_act = getattr(model, "out_activation", "linear")
            Y = _apply_out_act(Y, out_act).clamp(0, 1)
            mse_grid = ((Y - Xs) ** 2).mean().item()
            print(f"[grid] shown-samples MSE: {mse_grid:.4f}")

        H, W = img_hw
        plt.figure(figsize=(1.2 * n_cols, 2.6))
        for i in range(n_cols):
            ax = plt.subplot(2, n_cols, i + 1)
            ax.imshow(Xs[i].reshape(H, W), cmap="gray", interpolation="nearest")
            ax.axis("off")
            if i == 0: ax.set_title("Original", fontsize=9)
            ax = plt.subplot(2, n_cols, n_cols + i + 1)
            ax.imshow(Y[i].reshape(H, W), cmap="gray", interpolation="nearest")
            ax.axis("off")
            if i == 0: ax.set_title("Reconstruction", fontsize=9)
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        print(f"Saved recon grid: {out_path}")
    except Exception as e:
        print(f"[warn] failed to save recon grid: {e}")

def _save_loss_curves(history: List[Dict], best_cfg: Dict, out_path: str):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        best_hist = None
        for h in history:
            if h["cfg"] == best_cfg:
                best_hist = h; break
        if best_hist is None:
            print("[warn] best history not found; skipping loss plot"); return
        tr, va = best_hist["train_curve"], best_hist["val_curve"]
        plt.figure(figsize=(5.5, 3.8))
        plt.plot(tr, label="train"); plt.plot(va, label="val")
        plt.xlabel("epoch"); plt.ylabel("recon loss")
        plt.title("AE training curves (best config)")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_path, dpi=150); plt.close()
        print(f"Saved loss curves: {out_path}")
    except Exception as e:
        print(f"[warn] failed to save loss curves: {e}")

# ---------------------------------
# Build model with signature-safe kwargs
# ---------------------------------
def build_model(input_dim: int, cfg: Dict) -> Autoencoder:
    cand = dict(
        input_dim=input_dim,
        enc_dims=cfg["enc_dims"],
        dec_dims=cfg["dec_dims"],
        init=cfg.get("init", "xavier"),
        activation=cfg.get("act", "relu"),
        use_batchnorm=cfg.get("bn", True),
        bn_where=cfg.get("bn_where", "both"),
        out_activation=cfg.get("out_act", "linear"),
        loss_mode=cfg.get("loss", "mse"),
    )
    sig = inspect.signature(Autoencoder)
    safe_kwargs = {k: v for k, v in cand.items() if k in sig.parameters}
    return Autoencoder(**safe_kwargs)

# ---------------------------------
# Hyperparameter search + save best
# ---------------------------------
def run_experiments(
    dataset_name: str = "fashion",
    epochs: int = 100,
    weight_decay: float = 1e-5,
):
    Xtr, Xval, Xte = get_data(dataset_name)
    D = Xtr.shape[1]
    print(f"Dataset={dataset_name}  |  train={Xtr.shape}  val={Xval.shape}  test={Xte.shape}")

    # Single best config (your benchmark winner)
    grid = [ {
        "enc_dims":[256,128], "dec_dims":[256],
        "bn": True, "bn_where":"both",
        "init":"he", "act":"relu",
        "out_act":"linear", "loss":"mse",
        "lr":3e-3, "batch":128
    } ]

    best = {"val_loss": float("inf"), "state": None, "cfg": None}
    history: List[Dict] = []

    for ci, cfg in enumerate(grid, 1):
        print(f"\n=== Config {ci}/{len(grid)}: {cfg} ===")
        model = build_model(D, cfg)

        tr_curve, val_curve = [], []
        for ep in range(1, epochs + 1):
            tr_loss = train_ae_one_epoch(model, Xtr, lr=cfg["lr"],
                                         batch_size=cfg["batch"], weight_decay=weight_decay)
            val_loss = eval_ae(model, Xval)
            tr_curve.append(tr_loss); val_curve.append(val_loss)
            print(f"[{ci}:{ep:02d}] train_recon={tr_loss:.4f}  val_recon={val_loss:.4f}")

            if val_loss < best["val_loss"]:
                best["val_loss"] = val_loss
                best["cfg"] = cfg
                best["state"] = ae_state_dict(model)

        history.append({"cfg": cfg, "train_curve": tr_curve, "val_curve": val_curve})

    assert best["state"] is not None, "No best model captured."
    print(f"\nBest config: {best['cfg']}  (val_recon={best['val_loss']:.4f})")

    # Rebuild best model and test
    best_model = build_model(D, best["cfg"])
    ae_load_state_dict(best_model, best["state"])
    test_recon = eval_ae(best_model, Xte)
    print(f"[BEST] test_recon={test_recon:.4f}")

    # Save checkpoint
    save_dir = os.path.join(ROOT, "saved_models"); os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ae_{dataset_name}_best.pt")
    torch.save({
        "meta": {
            "dataset": dataset_name,
            "input_dim": D,
            "config": best["cfg"],
            "val_recon": best["val_loss"],
            "test_recon": test_recon,
        },
        "state": best["state"],
    }, save_path)
    print(f"Saved best AE parameters to: {save_path}")

    # Visuals
    out_dir = os.path.join(ROOT, "graphs_exp"); os.makedirs(out_dir, exist_ok=True)
    _save_recon_grid(best_model, Xte, os.path.join(out_dir, f"ae_{dataset_name}_recon_grid.png"))
    _save_loss_curves(history, best["cfg"], os.path.join(out_dir, f"ae_{dataset_name}_loss.png"))

    rmse = math.sqrt(float(test_recon))
    psnr = 10.0 * math.log10(1.0 / max(float(test_recon), 1e-12))
    print(f"[BEST] RMSE={rmse:.3f}  PSNR={psnr:.2f} dB")

    return history, best, save_path

if __name__ == "__main__":
    run_experiments(dataset_name="fashion", epochs=100, weight_decay=1e-5)