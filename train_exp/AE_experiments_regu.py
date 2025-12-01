# train_exp/AE_experiments_regu.py
import os, sys, math, inspect
import torch
from typing import Dict, List

# ---- project import path ----
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dataset as ds
from models_ae import Autoencoder

# ---- headless plotting ----
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
            val_ratio=val_ratio, seed=seed
        )
        return Xtr.float(), Xval.float(), Xte.float()

    if dataset_name == "mnist" and hasattr(ds, "load_mnist_processed"):
        Xtr_full, _ytr, Xte, _yte = ds.load_mnist_processed(
            processed_dir=os.path.join(ROOT, "data/processed")
        )
        N = Xtr_full.shape[0]
        idx = torch.randperm(N)
        n_val = int(val_ratio * N)
        return Xtr_full[idx[n_val:]].float(), Xtr_full[idx[:n_val]].float(), Xte.float()

    # fallback: fashion (preprocessed)
    Xtr, _ytr, Xte, _yte = ds.load_fashion_mnist_processed(
        processed_dir=os.path.join(ROOT, "data/processed")
    )
    N = Xtr.shape[0]
    idx = torch.randperm(N)
    n_val = int(val_ratio * N)
    return Xtr[idx[n_val:]].float(), Xtr[idx[:n_val]].float(), Xte.float()

# -----------------------
# Train / Eval helpers
# -----------------------
def batches(X: torch.Tensor, batch_size: int, shuffle=True):
    N = X.shape[0]
    idx = torch.randperm(N) if shuffle else torch.arange(N)
    for i in range(0, N, batch_size):
        yield X[idx[i:i+batch_size]]

def train_epoch(model: Autoencoder, X: torch.Tensor,
                lr: float, batch_size: int, weight_decay: float):
    model.zero_grad()
    tot, cnt = 0.0, 0
    for xb in batches(X, batch_size, shuffle=True):
        yhat, cache = model.forward(xb, training=True)
        loss, dY = model.loss_and_grad(yhat, xb)

        if weight_decay > 0:
            l2 = 0.0
            for W in model.W:
                l2 += (W * W).sum()
            loss = loss + weight_decay * l2 / X.shape[0]

        model.backward(xb, cache, dY)
        model.step(lr=lr)

        tot += float(loss.item()) * xb.size(0)
        cnt += xb.size(0)
    return tot / cnt

@torch.no_grad()
def eval_epoch(model: Autoencoder, X: torch.Tensor, batch_size: int = 512):
    model.zero_grad()
    tot, cnt = 0.0, 0
    for xb in batches(X, batch_size, shuffle=False):
        yhat, _ = model.forward(xb, training=False)
        loss, _ = model.loss_and_grad(yhat, xb)
        tot += float(loss.item()) * xb.size(0)
        cnt += xb.size(0)
    return tot / cnt

# -----------------------
# State helpers
# -----------------------
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

# -----------------------
# Visuals
# -----------------------
def _apply_out_act(Y: torch.Tensor, out_act: str) -> torch.Tensor:
    if out_act == "sigmoid": return torch.sigmoid(Y)
    if out_act == "tanh":    return torch.tanh(Y)
    return Y

def save_recon_grid(model: Autoencoder, X: torch.Tensor, out_path: str,
                    n_cols: int = 12, img_hw=(28, 28)):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with torch.no_grad():
            idx = torch.randperm(X.shape[0])[:n_cols]
            Xs = X[idx]
            Y, _ = model.forward(Xs, training=False)
            out_act = getattr(model, "out_activation", "linear")
            Y = _apply_out_act(Y, out_act).clamp(0, 1)
            print(f"[grid] shown-samples MSE: {((Y - Xs) ** 2).mean().item():.4f}")

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
        print(f"[warn] recon grid failed: {e}")

def save_loss_curves(train: List[float], val: List[float], out_path: str, title="AE (regularised)"):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.figure(figsize=(5.5, 3.8))
        plt.plot(train, label="train")
        plt.plot(val, label="val")
        plt.xlabel("epoch"); plt.ylabel("recon loss"); plt.title(title)
        plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        print(f"Saved loss curves: {out_path}")
    except Exception as e:
        print(f"[warn] loss plot failed: {e}")

# -----------------------
# Model builder (signature-safe)
# -----------------------
def build_autoencoder(D: int, cfg: Dict) -> Autoencoder:
    cand = dict(
        input_dim=D,
        enc_dims=cfg["enc_dims"], dec_dims=cfg["dec_dims"],
        init=cfg.get("init", "he"), activation=cfg.get("act", "relu"),
        use_batchnorm=cfg.get("bn", True),
        bn_where=cfg.get("bn_where", "both"),
        out_activation=cfg.get("out_act", "linear"),
        loss_mode=cfg.get("loss", "mse"),
    )
    sig = inspect.signature(Autoencoder)
    kwargs = {k: v for k, v in cand.items() if k in sig.parameters}
    return Autoencoder(**kwargs)

# -----------------------
# Main experiment (L2 + EarlyStopping + small grid search)
# -----------------------
def run_experiments(
    dataset_name: str = "fashion",
    epochs: int = 120,
    patience: int = 10,
):
    Xtr, Xval, Xte = get_data(dataset_name)
    D = Xtr.shape[1]
    print(f"Dataset={dataset_name} | train={Xtr.shape}  val={Xval.shape}  test={Xte.shape}")
    print(f"Regularisers: L2(weight_decay)  EarlyStopping(patience={patience})")

    # small grid search
    base = dict(
        enc_dims=[256, 128], dec_dims=[256],
        bn=True, bn_where="both", init="he", act="relu",
        out_act="linear", loss="mse", batch=128
    )
    grid: List[Dict] = []
    for lr in [1e-3, 3e-3]:
        for wd in [1e-4, 5e-4]:
            cfg = dict(base); cfg["lr"] = lr; cfg["weight_decay"] = wd
            grid.append(cfg)

    best_global = {"val_loss": float("inf"), "state": None, "cfg": None,
                   "train_curve": None, "val_curve": None}

    for ci, cfg in enumerate(grid, 1):
        print(f"\n=== Config {ci}/{len(grid)}: {cfg} ===")
        model = build_autoencoder(D, cfg)
        tr_hist, va_hist = [], []

        best_local = float("inf")
        wait = 0
        best_state_local = None

        for ep in range(1, epochs + 1):
            tr = train_epoch(model, Xtr, lr=cfg["lr"], batch_size=cfg["batch"],
                             weight_decay=cfg["weight_decay"])
            va = eval_epoch(model, Xval)
            tr_hist.append(tr); va_hist.append(va)
            print(f"[{ci}:{ep:03d}] train_recon={tr:.4f}  val_recon={va:.4f}")

            if va < best_local - 1e-6:
                best_local = va
                best_state_local = ae_state_dict(model)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {ep} (best val={best_local:.4f})")
                    break

        # keep the best of this config
        if best_local < best_global["val_loss"]:
            best_global.update({
                "val_loss": best_local, "cfg": cfg,
                "state": best_state_local,
                "train_curve": tr_hist, "val_curve": va_hist
            })

    assert best_global["state"] is not None, "No best model captured."
    print(f"\nBest config: {best_global['cfg']}  (val_recon={best_global['val_loss']:.4f})")

    # rebuild, load, evaluate test
    best_model = build_autoencoder(D, best_global["cfg"])
    ae_load_state_dict(best_model, best_global["state"])
    test_recon = eval_epoch(best_model, Xte)
    print(f"[BEST] test_recon={test_recon:.4f}")

    # save checkpoint
    save_dir = os.path.join(ROOT, "saved_models"); os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ae_{dataset_name}_best_reg.pt")
    torch.save({
        "meta": {
            "dataset": dataset_name,
            "input_dim": D,
            "config": best_global["cfg"],
            "val_recon": best_global["val_loss"],
            "test_recon": test_recon,
        },
        "state": best_global["state"],
    }, save_path)
    print(f"Saved regularised AE parameters to: {save_path}")

    # plots
    out_dir = os.path.join(ROOT, "graphs_exp"); os.makedirs(out_dir, exist_ok=True)
    save_loss_curves(best_global["train_curve"], best_global["val_curve"],
                     os.path.join(out_dir, f"ae_{dataset_name}_regu_loss.png"),
                     title="AE (L2 + EarlyStopping)")
    save_recon_grid(best_model, Xte, os.path.join(out_dir, f"ae_{dataset_name}_regu_recon_grid.png"))

    rmse = math.sqrt(float(test_recon))
    psnr = 10.0 * math.log10(1.0 / max(float(test_recon), 1e-12))
    print(f"[BEST] RMSE={rmse:.3f}  PSNR={psnr:.2f} dB")

def main():
    run_experiments()

if __name__ == "__main__":
    main()