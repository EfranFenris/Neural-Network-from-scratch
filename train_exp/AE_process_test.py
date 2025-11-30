# train_exp/AE_process_test.py
import os, sys, math
import torch
import numpy as np

# --- project import path ---
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dataset as ds
from models_ae import Autoencoder

# ---------- helpers ----------
def _cfg_get(cfg: dict, key: str, default):
    return cfg[key] if key in cfg else default

def _apply_out_act(Y: torch.Tensor, out_act: str) -> torch.Tensor:
    if out_act == "sigmoid":
        return torch.sigmoid(Y)
    if out_act == "tanh":
        return torch.tanh(Y)
    return Y  # linear

def load_autoencoder(ckpt_path: str) -> tuple[Autoencoder, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    cfg  = meta.get("config", {})
    D    = meta.get("input_dim", 784)

    model = Autoencoder(
        input_dim=D,
        enc_dims=cfg["enc_dims"],
        dec_dims=cfg["dec_dims"],
        init=_cfg_get(cfg, "init", "xavier"),
        activation=_cfg_get(cfg, "act", "sigmoid"),
        use_batchnorm=_cfg_get(cfg, "bn", True),
        bn_where=_cfg_get(cfg, "bn_where", "both"),
        out_activation=_cfg_get(cfg, "out_act", "sigmoid"),
        loss_mode=_cfg_get(cfg, "loss", "bce"),
    )

    # load state (supports BN)
    state = ckpt["state"]
    for i in range(len(model.W)):
        model.W[i].copy_(state["W"][i])
        model.b[i].copy_(state["b"][i])
    if getattr(model, "gamma", None) and "gamma" in state:
        for i in range(len(model.gamma)):
            model.gamma[i].copy_(state["gamma"][i])
            model.beta[i].copy_(state["beta"][i])
            model.running_mean[i].copy_(state["running_mean"][i])
            model.running_var[i].copy_(state["running_var"][i])
    return model, meta

def load_test_split(dataset_name: str):
    """Returns Xtest, ytest (float32 in [0,1])."""
    if dataset_name == "fashion" and hasattr(ds, "load_fashion_mnist_processed"):
        Xtr, ytr, Xte, yte = ds.load_fashion_mnist_processed(
            processed_dir=os.path.join(ROOT, "data/processed")
        )
        return Xte.float(), yte
    if dataset_name == "mnist" and hasattr(ds, "load_mnist_processed"):
        Xtr, ytr, Xte, yte = ds.load_mnist_processed(
            processed_dir=os.path.join(ROOT, "data/processed")
        )
        return Xte.float(), yte
    # fallback to fashion if unknown
    Xtr, ytr, Xte, yte = ds.load_fashion_mnist_processed(
        processed_dir=os.path.join(ROOT, "data/processed")
    )
    return Xte.float(), yte

# ---------- main ----------
@torch.no_grad()
def main():
    # 1) find checkpoint
    ckpt = os.path.join(ROOT, "saved_models", "ae_fashion_best.pt")
    

    # 2) load model
    model, meta = load_autoencoder(ckpt)
    cfg = meta.get("config", {})
    dataset_name = meta.get("dataset", "fashion")
    out_act = cfg.get("out_act", "linear")
    print(f"Loaded AE from {ckpt}\n  dataset={dataset_name}  config={cfg}")

    # 3) load test data
    Xte, yte = load_test_split(dataset_name)
    N, D = Xte.shape
    print(f"Test split: X={Xte.shape}  y={tuple(yte.shape) if torch.is_tensor(yte) else 'None'}")

    # 4) process test data in batches
    B = 1024
    all_Z, all_Yhat, all_mse = [], [], []
    for i in range(0, N, B):
        xb = Xte[i:i+B]
        Zb = model.encode(xb)                              # latent
        Yb, _ = model.forward(xb, training=False)          # decoder output (pre-activation)
        Yb = _apply_out_act(Yb, out_act).clamp(0, 1)       # what the loss “sees”

        # per-sample MSE
        mse = ((Yb - xb)**2).view(xb.size(0), -1).mean(dim=1)

        all_Z.append(Zb.cpu())
        all_Yhat.append(Yb.cpu())
        all_mse.append(mse.cpu())

    Z = torch.cat(all_Z, 0).numpy()
    Yhat = torch.cat(all_Yhat, 0).numpy()
    mse = torch.cat(all_mse, 0).numpy()

    mean_mse = float(mse.mean())
    rmse = math.sqrt(mean_mse)
    psnr = 10.0 * math.log10(1.0 / max(mean_mse, 1e-12))
    print(f"[TEST] mean MSE={mean_mse:.4f}  RMSE={rmse:.4f}  PSNR={psnr:.2f} dB")

    # 5) save artifacts for later parts (features as images / UMAP, etc.)
    out_dir = os.path.join(ROOT, "outputs"); os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f"ae_{dataset_name}_test_processed.npz")
    np.savez_compressed(
        out_npz,
        Z=Z,                  # latent codes [N, K]
        Yhat=Yhat,            # reconstructions [N, D]
        mse=mse,              # per-image error [N]
        y=(yte.cpu().numpy() if torch.is_tensor(yte) else None),
        config=cfg,
        dataset=dataset_name,
    )
    print(f"Saved processed test data -> {out_npz}")

if __name__ == "__main__":
    main()