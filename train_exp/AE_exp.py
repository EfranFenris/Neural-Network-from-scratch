# train_exp/AE_experiments.py
import os, sys, math, copy
import torch
from typing import List, Dict, Tuple

# --- project import path ---
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dataset as ds
from models_ae import Autoencoder

# -----------------------
# Data loading (X only)
# -----------------------
def get_data(dataset_name: str = "mnist", val_ratio: float = 0.1, seed: int = 0):
    torch.manual_seed(seed)
    if dataset_name == "fashion" and hasattr(ds, "load_fashion_mnist_processed"):
        Xtr, _ytr, Xte, _yte = ds.load_fashion_mnist_processed()
    elif dataset_name == "affnist" and hasattr(ds, "load_affnist_centered_processed"):
        Xtr, _ytr, Xte, _yte = ds.load_affnist_centered_processed(
            processed_dir=os.path.join(ROOT, "data/processed"),
            mat_path=os.path.join(ROOT, "data/raw/just_centered/training_and_validation.mat"),
            subsample_train=None, subsample_test=None
        )
    else:
        # default to MNIST
        Xtr, _ytr, Xte, _yte = ds.load_mnist_processed()

    # make a validation split from train
    N = Xtr.shape[0]
    idx = torch.randperm(N)
    n_val = int(val_ratio * N)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtrain, Xval = Xtr[tr_idx], Xtr[val_idx]
    return Xtrain.float(), Xval.float(), Xte.float()

# ---------------------------------
# AE training utils (no labels)
# ---------------------------------
def batches(X: torch.Tensor, batch_size: int, shuffle=True):
    N = X.shape[0]
    idx = torch.randperm(N) if shuffle else torch.arange(N)
    for i in range(0, N, batch_size):
        j = idx[i:i+batch_size]
        yield X[j]

def train_ae_one_epoch(model: Autoencoder, X: torch.Tensor,
                       lr: float, batch_size: int, weight_decay: float = 0.0):
    model.zero_grad()
    total_loss = 0.0
    total_count = 0
    for xb in batches(X, batch_size, shuffle=True):
        # forward
        x_hat, cache = model.forward(xb, training=True)
        loss, dY = model.loss_and_grad(x_hat, xb)

        # manual L2 (weight decay on weights only)
        if weight_decay > 0:
            l2 = 0.0
            for W in model.W:
                l2 += (W * W).sum()
            loss = loss + weight_decay * l2 / X.shape[0]

        # backward + step
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
        x_hat, _ = model.forward(xb, training=False)
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
        "eps": model.eps,
        "momentum": model.momentum,
        "W": [w.detach().cpu().clone() for w in model.W],
        "b": [b.detach().cpu().clone() for b in model.b],
    }
    if model.use_bn:
        d["gamma"] = [g.detach().cpu().clone() for g in model.gamma]
        d["beta"]  = [b.detach().cpu().clone() for b in model.beta]
        d["running_mean"] = [m.detach().cpu().clone() for m in model.running_mean]
        d["running_var"]  = [v.detach().cpu().clone() for v in model.running_var]
    return d

def ae_load_state_dict(model: Autoencoder, state: Dict):
    for i in range(len(model.W)):
        model.W[i].copy_(state["W"][i])
        model.b[i].copy_(state["b"][i])
    if model.use_bn and "gamma" in state:
        for i in range(len(model.gamma)):
            model.gamma[i].copy_(state["gamma"][i])
            model.beta[i].copy_(state["beta"][i])
            model.running_mean[i].copy_(state["running_mean"][i])
            model.running_var[i].copy_(state["running_var"][i])

# ---------------------------------
# Hyperparameter search + save best
# ---------------------------------
def run_experiments(
    dataset_name: str = "mnist",
    epochs: int = 20,
    weight_decay: float = 0.0, 
):
    Xtr, Xval, Xte = get_data(dataset_name)

    D = Xtr.shape[1]
    print(f"Dataset={dataset_name}  |  train={Xtr.shape}  val={Xval.shape}  test={Xte.shape}")

    # grid of hyperparameters (edit as you like)
    grid = [
        # enc_dims define the encoder; decoder is mirrored via dec_dims
        {"enc_dims": [256, 64], "dec_dims": [256], "lr": 1e-2,  "batch": 128, "init": "he",     "act": "relu",    "bn": True},
        {"enc_dims": [256, 32], "dec_dims": [256], "lr": 5e-3,  "batch": 128, "init": "he",     "act": "relu",    "bn": True},
        {"enc_dims": [128, 32], "dec_dims": [128], "lr": 1e-2,  "batch": 256, "init": "xavier", "act": "sigmoid", "bn": True},
        {"enc_dims": [512,128], "dec_dims": [512], "lr": 5e-3,  "batch": 128, "init": "he",     "act": "relu",    "bn": True},
    ]

    best = {"val_loss": float("inf"), "state": None, "cfg": None}
    history: List[Dict] = []

    for ci, cfg in enumerate(grid, 1):
        print(f"\n=== Config {ci}/{len(grid)}: {cfg} ===")
        model = Autoencoder(
            input_dim=D,
            enc_dims=cfg["enc_dims"],
            dec_dims=cfg["dec_dims"],
            init=cfg["init"],
            activation=cfg["act"],
            use_batchnorm=cfg["bn"],
        )

        tr_curve, val_curve = [], []
        for ep in range(1, epochs + 1):
            tr_loss = train_ae_one_epoch(model, Xtr, lr=cfg["lr"], batch_size=cfg["batch"], weight_decay=weight_decay)
            val_loss = eval_ae(model, Xval)
            tr_curve.append(tr_loss); val_curve.append(val_loss)
            print(f"[{ci}:{ep:02d}] train_recon={tr_loss:.4f}  val_recon={val_loss:.4f}")

            # keep best-by-val
            if val_loss < best["val_loss"]:
                best["val_loss"] = val_loss
                best["cfg"] = cfg
                best["state"] = ae_state_dict(model)

        # store run summary
        history.append({"cfg": cfg, "train_curve": tr_curve, "val_curve": val_curve})

    # final evaluation on test with best state
    assert best["state"] is not None, "No best model captured."
    print(f"\nBest config: {best['cfg']}  (val_recon={best['val_loss']:.4f})")

    # rebuild same-arch model and load weights
    cfg = best["cfg"]
    best_model = Autoencoder(
        input_dim=D,
        enc_dims=cfg["enc_dims"],
        dec_dims=cfg["dec_dims"],
        init=cfg["init"],
        activation=cfg["act"],
        use_batchnorm=cfg["bn"],
    )
    ae_load_state_dict(best_model, best["state"])
    test_recon = eval_ae(best_model, Xte)
    print(f"[BEST] test_recon={test_recon:.4f}")

    # -------- Save final model (parameters) --------
    save_dir = os.path.join(ROOT, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ae_{dataset_name}_best.pt")
    torch.save({
        "meta": {
            "dataset": dataset_name,
            "input_dim": D,
            "config": cfg,
            "val_recon": best["val_loss"],
            "test_recon": test_recon,
        },
        "state": best["state"],  # all tensors (W, b, gamma, beta, running stats)
    }, save_path)
    print(f"Saved best AE parameters to: {save_path}")

    return history, best, save_path

if __name__ == "__main__":
    # Example: change dataset_name to "fashion" if you added that loader
    run_experiments(dataset_name="mnist", epochs=20, weight_decay=0.0)