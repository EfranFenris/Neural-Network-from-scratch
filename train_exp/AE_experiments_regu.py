# train_exp/AE_experiments.py
import os, sys
import torch
from typing import List, Dict, Tuple

# --- project import path ---
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dataset as ds
from models_ae import Autoencoder

# --- minimal plotting (headless-safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- NUEVO: augment sencillo en 28x28 ---
def augment_batch(x, p_shift=0.7, max_shift=2, p_drop=0.1):
    # x: [B,784] en [0,1]
    B, D = x.shape
    H = W = 28
    x = x.view(B, 1, H, W)
    # shift aleatorio pequeño
    if torch.rand(()) < p_shift:
        dx = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        dy = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    # dropout de píxeles
    if p_drop > 0:
        mask = (torch.rand_like(x) > p_drop).float()
        x = x * mask
    return x.view(B, D).clamp_(0, 1)

# --- MOD: ahora acepta ruido y augment ---
def train_ae_one_epoch(model: Autoencoder, X: torch.Tensor,
                       lr: float, batch_size: int,
                       weight_decay: float = 0.0,
                       noise_std: float = 0.0,
                       use_augment: bool = False):
    model.zero_grad()
    total_loss, total_count = 0.0, 0
    for xb in batches(X, batch_size, shuffle=True):
        x_in = xb
        if use_augment:
            x_in = augment_batch(x_in)
        if noise_std > 0:
            x_in = (x_in + noise_std * torch.randn_like(x_in)).clamp(0, 1)  # denoising AE

        # forward sobre x_in pero se reconstruye el xb limpio
        x_hat, cache = model.forward(x_in, training=True)
        loss, dY = model.loss_and_grad(x_hat, xb)

        if weight_decay > 0:
            l2 = 0.0
            for W in model.W:
                l2 += (W * W).sum()
            loss = loss + weight_decay * l2 / X.shape[0]

        model.backward(x_in, cache, dY)
        model.step(lr=lr)

        total_loss += float(loss.item()) * xb.shape[0]
        total_count += xb.shape[0]
    return total_loss / total_count
# -----------------------
# Data loading (X only)
# -----------------------
def get_data(dataset_name: str = "fashion", val_ratio: float = 0.1, seed: int = 0):
    torch.manual_seed(seed)

    if dataset_name == "fashion" and hasattr(ds, "load_fashion_mnist_for_ae"):
        Xtr, Xval, Xte, *_ = ds.load_fashion_mnist_for_ae(
            processed_dir=os.path.join(ROOT, "data/processed"),
            val_ratio=val_ratio,
            seed=seed,
        )
        return Xtr.float(), Xval.float(), Xte.float()

    if dataset_name == "mnist" and hasattr(ds, "load_mnist_processed"):
        Xtr_full, _ytr, Xte, _yte = ds.load_mnist_processed()
        N = Xtr_full.shape[0]
        idx = torch.randperm(N)
        n_val = int(val_ratio * N)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        return Xtr_full[tr_idx].float(), Xtr_full[val_idx].float(), Xte.float()

    if dataset_name == "affnist" and hasattr(ds, "load_affnist_centered_processed"):
        Xtr, _ytr, Xte, _yte = ds.load_affnist_centered_processed(
            processed_dir=os.path.join(ROOT, "data/processed"),
            mat_path=os.path.join(ROOT, "data/raw/just_centered", "training_and_validation.mat"),
            subsample_train=None, subsample_test=None
        )
        N = Xtr.shape[0]
        idx = torch.randperm(N)
        n_val = int(val_ratio * N)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        return Xtr[tr_idx].float(), Xtr[val_idx].float(), Xte.float()

    # default fallback
    Xtr, _ytr, Xte, _yte = ds.load_fashion_mnist_processed(os.path.join(ROOT, "data/processed"))
    N = Xtr.shape[0]
    idx = torch.randperm(N)
    n_val = int(val_ratio * N)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return Xtr[tr_idx].float(), Xtr[val_idx].float(), Xte.float()

# ---------------------------------
# AE training utils (no labels)
# ---------------------------------
def batches(X: torch.Tensor, batch_size: int, shuffle=True):
    N = X.shape[0]
    idx = torch.randperm(N) if shuffle else torch.arange(N)
    for i in range(0, N, batch_size):
        j = idx[i:i + batch_size]
        yield X[j]

def train_ae_one_epoch(model: Autoencoder, X: torch.Tensor,
                       lr: float, batch_size: int, weight_decay: float = 0.0):
    model.zero_grad()
    total_loss = 0.0
    total_count = 0
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
    total_loss = 0.0
    total_count = 0
    for xb in batches(X, batch_size, shuffle=False):
        x_hat, _ = model.forward(xb, training=False)   # BN uses running stats
        loss, _ = model.loss_and_grad(x_hat, xb)
        total_loss += float(loss.item()) * xb.shape[0]
        total_count += xb.shape[0]
    return total_loss / total_count

# ---------------------------------
# Save / load state (parameters)
# ---------------------------------
def ae_state_dict(model: Autoencoder) -> Dict:
    d = {
        "sizes": model.sizes,
        "activation": model.activation_name,
        "use_bn": model.use_bn,
        "bn_where": getattr(model, "bn_where", "encoder"),
        "out_activation": getattr(model, "out_activation", "linear"),
        "loss_mode": getattr(model, "loss_mode", "mse"),
        "eps": model.eps,
        "momentum": model.momentum,
        "W": [w.detach().cpu().clone() for w in model.W],
        "b": [b.detach().cpu().clone() for b in model.b],
    }
    if getattr(model, "gamma", []):
        d["gamma"] = [g.detach().cpu().clone() for g in model.gamma]
        d["beta"]  = [be.detach().cpu().clone() for be in model.beta]
        d["running_mean"] = [m.detach().cpu().clone() for m in model.running_mean]
        d["running_var"]  = [v.detach().cpu().clone() for v in model.running_var]
    return d

def ae_load_state_dict(model: Autoencoder, state: Dict):
    for i in range(len(model.W)):
        model.W[i].copy_(state["W"][i])
        model.b[i].copy_(state["b"][i])
    if getattr(model, "gamma", None) and "gamma" in state:
        for i in range(len(model.gamma)):
            model.gamma[i].copy_(state["gamma"][i])
            model.beta[i].copy_(state["beta"][i])
            model.running_mean[i].copy_(state["running_mean"][i])
            model.running_var[i].copy_(state["running_var"][i])

# ---------------------------------
# Tiny built-in visuals
# ---------------------------------
def _apply_out_act(Y: torch.Tensor, out_act: str) -> torch.Tensor:
    if out_act == "sigmoid":
        return torch.sigmoid(Y)
    if out_act == "tanh":
        return torch.tanh(Y)
    return Y  # linear or unknown

def _save_recon_grid(model: Autoencoder, X: torch.Tensor, out_path: str,
                     n_cols: int = 12, img_hw=(28, 28)):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with torch.no_grad():
            idx = torch.randperm(X.shape[0])[:n_cols]
            Xs = X[idx]
            Y, _ = model.forward(Xs, training=False)

            # visualize what the loss saw
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

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
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

        tr = best_hist["train_curve"]; va = best_hist["val_curve"]
        plt.figure(figsize=(5.5, 3.8))
        plt.plot(tr, label="train")
        plt.plot(va, label="val")
        plt.xlabel("epoch"); plt.ylabel("recon loss")
        plt.title("AE training curves (best config)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved loss curves: {out_path}")
    except Exception as e:
        print(f"[warn] failed to save loss curves: {e}")

# ---------------------------------
# Hyperparameter search + save best
# ---------------------------------
def run_experiments(
    dataset_name: str = "fashion",
    epochs: int = 200,
    weight_decay: float = 1e-4,      # L2: parámetro norm penalty
    noise_std: float = 0.20,         # Injecting Noise (denoising AE)
    use_augment: bool = True,        # Data augmentation ligera
    patience: int = 20,              # Early stopping
    save_suffix: str = "_reg",       # no pisar el baseline
):
    Xtr, Xval, Xte = get_data(dataset_name)
    D = Xtr.shape[1]
    print(f"Dataset={dataset_name} | train={Xtr.shape} val={Xval.shape} test={Xte.shape}")

    grid = [
        {"enc_dims":[256,128], "dec_dims":[256], "lr":3e-3, "batch":128,
         "init":"he", "act":"relu", "bn":True, "bn_where":"both",
         "out_act":"linear", "loss":"mse"}
    ]

    best = {"val_loss": float("inf"), "state": None, "cfg": None}
    history: List[Dict] = []

    for ci, cfg in enumerate(grid, 1):
        print(f"\n=== Config {ci}/{len(grid)}: {cfg} ===")
        model = Autoencoder(
            input_dim=D,
            enc_dims=cfg["enc_dims"], dec_dims=cfg["dec_dims"],
            init=cfg["init"], activation=cfg["act"],
            use_batchnorm=cfg["bn"], bn_where=cfg["bn_where"],
            out_activation=cfg["out_act"], loss_mode=cfg["loss"],
        )

        tr_curve, val_curve = [], []
        no_improve = 0
        for ep in range(1, epochs + 1):
            tr_loss = train_ae_one_epoch(
                model, Xtr, lr=cfg["lr"], batch_size=cfg["batch"],
                weight_decay=weight_decay, noise_std=noise_std, use_augment=use_augment
            )
            val_loss = eval_ae(model, Xval)
            tr_curve.append(tr_loss); val_curve.append(val_loss)
            print(f"[{ci}:{ep:03d}] train={tr_loss:.4f}  val={val_loss:.4f}")

            if val_loss + 1e-6 < best["val_loss"]:
                best.update(val_loss=val_loss, cfg=cfg, state=ae_state_dict(model))
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping (patience {patience}) en epoch {ep}")
                    break

        history.append({"cfg": cfg, "train_curve": tr_curve, "val_curve": val_curve})

    assert best["state"] is not None
    print(f"\nBest config: {best['cfg']}  (val_recon={best['val_loss']:.4f})")

    # Reconstituye y evalúa
    cfg = best["cfg"]
    best_model = Autoencoder(
        input_dim=D,
        enc_dims=cfg["enc_dims"], dec_dims=cfg["dec_dims"],
        init=cfg["init"], activation=cfg["act"],
        use_batchnorm=cfg["bn"], bn_where=cfg["bn_where"],
        out_activation=cfg["out_act"], loss_mode=cfg["loss"],
    )
    ae_load_state_dict(best_model, best["state"])
    test_recon = eval_ae(best_model, Xte)
    print(f"[BEST] test_recon={test_recon:.4f}")

    # Guardado con sufijo _reg
    save_dir = os.path.join(ROOT, "saved_models"); os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ae_{dataset_name}_best{save_suffix}.pt")
    torch.save({
        "meta": {
            "dataset": dataset_name, "input_dim": D, "config": cfg,
            "val_recon": best["val_loss"], "test_recon": test_recon,
            "regularization": {
                "weight_decay": weight_decay, "noise_std": noise_std,
                "use_augment": use_augment, "patience": patience
            }
        },
        "state": best["state"],
    }, save_path)
    print(f"Saved best AE parameters to: {save_path}")

    # Visuales
    out_dir = os.path.join(ROOT, "graphs_exp"); os.makedirs(out_dir, exist_ok=True)
    _save_recon_grid(best_model, Xte, os.path.join(out_dir, f"ae_{dataset_name}_recon_grid{save_suffix}.png"))
    _save_loss_curves(history, cfg, os.path.join(out_dir, f"ae_{dataset_name}_loss{save_suffix}.png"))

    import math
    rmse = math.sqrt(float(test_recon)); psnr = 10.0 * math.log10(1.0 / float(test_recon))
    print(f"[BEST] RMSE={rmse:.3f}  PSNR={psnr:.2f} dB")
    return history, best, save_path

if __name__ == "__main__":
    run_experiments()