# train_exp/AE_process_test.py
import os, sys, math, argparse, inspect
import torch
import numpy as np

# --- project import path ---
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dataset as ds
from models_ae import Autoencoder

# ---------- helpers ----------
def _apply_out_act(Y: torch.Tensor, out_act: str) -> torch.Tensor:
    if out_act == "sigmoid":
        return torch.sigmoid(Y)
    if out_act == "tanh":
        return torch.tanh(Y)
    return Y  # linear (default)

def _build_autoencoder_compat(D: int, cfg: dict) -> Autoencoder:
    """
    Construct Autoencoder but only pass the kwargs that the current class accepts.
    Works whether your class supports bn_where/out_activation/loss_mode or not.
    """
    candidate = {
        "input_dim": D,
        "enc_dims": cfg.get("enc_dims"),
        "dec_dims": cfg.get("dec_dims"),
        "init":     cfg.get("init", "xavier"),
        "activation": cfg.get("act", "sigmoid"),
        "use_batchnorm": cfg.get("bn", True),
        # the next three may not exist on your class — we’ll filter them:
        "bn_where": cfg.get("bn_where", "both"),
        "out_activation": cfg.get("out_act", "linear"),
        "loss_mode": cfg.get("loss", "mse"),
    }
    allowed = set(inspect.signature(Autoencoder).parameters.keys())
    safe_kwargs = {k: v for k, v in candidate.items() if (v is not None and k in allowed)}
    return Autoencoder(**safe_kwargs)

def _load_autoencoder(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    cfg  = meta.get("config", {})
    D    = int(meta.get("input_dim", 784))

    model = _build_autoencoder_compat(D, cfg)

    # load weights (incl. BN if present in both)
    state = ckpt["state"]
    for i in range(len(model.W)):
        model.W[i].copy_(state["W"][i])
        model.b[i].copy_(state["b"][i])
    if getattr(model, "gamma", None) is not None and "gamma" in state:
        for i in range(len(model.gamma)):
            model.gamma[i].copy_(state["gamma"][i])
            model.beta[i].copy_(state["beta"][i])
            model.running_mean[i].copy_(state["running_mean"][i])
            model.running_var[i].copy_(state["running_var"][i])
    return model, meta

def _load_test_split(dataset_name: str):
    """Returns Xtest (float32 in [0,1]), ytest."""
    if dataset_name == "fashion" and hasattr(ds, "load_fashion_mnist_processed"):
        _Xtr, _ytr, Xte, yte = ds.load_fashion_mnist_processed(
            processed_dir=os.path.join(ROOT, "data/processed")
        )
        return Xte.float(), yte
    if dataset_name == "mnist" and hasattr(ds, "load_mnist_processed"):
        _Xtr, _ytr, Xte, yte = ds.load_mnist_processed(
            processed_dir=os.path.join(ROOT, "data/processed")
        )
        return Xte.float(), yte
    # fallback
    _Xtr, _ytr, Xte, yte = ds.load_fashion_mnist_processed(
        processed_dir=os.path.join(ROOT, "data/processed")
    )
    return Xte.float(), yte

# ---------- main ----------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
        default=os.path.join(ROOT, "saved_models", "ae_fashion_best.pt"))
    parser.add_argument("--tag", type=str, default="")   # e.g., "_base" or "_reg"
    args = parser.parse_args()

    # 1) load model
    model, meta = _load_autoencoder(args.ckpt)
    cfg = meta.get("config", {})
    dataset_name = meta.get("dataset", "fashion")

    # choose the right post-activation for reconstructions
    out_act = cfg.get("out_act", getattr(model, "out_activation", "linear"))
    print(f"Loaded AE from {args.ckpt}\n  dataset={dataset_name}\n  config={cfg}")

    # 2) load test data
    Xte, yte = _load_test_split(dataset_name)
    N, D = Xte.shape
    print(f"Test split: X={Xte.shape}  y={'None' if yte is None else tuple(yte.shape)}")

    # 3) process test data in batches
    B = 1024
    all_Z, all_Yhat, all_mse = [], [], []
    for i in range(0, N, B):
        xb = Xte[i:i+B]
        Zb = model.encode(xb)                              # [B, K]
        Yb, _ = model.forward(xb, training=False)          # [B, D] pre-activation
        Yb = _apply_out_act(Yb, out_act).clamp(0, 1)       # what the loss “sees”
        mse = ((Yb - xb)**2).view(xb.size(0), -1).mean(dim=1)

        all_Z.append(Zb.cpu()); all_Yhat.append(Yb.cpu()); all_mse.append(mse.cpu())

    Z = torch.cat(all_Z, 0).numpy()
    Yhat = torch.cat(all_Yhat, 0).numpy()
    mse = torch.cat(all_mse, 0).numpy()

    mean_mse = float(mse.mean())
    rmse = math.sqrt(mean_mse)
    psnr = 10.0 * math.log10(1.0 / max(mean_mse, 1e-12))
    print(f"[TEST] mean MSE={mean_mse:.4f}  RMSE={rmse:.4f}  PSNR={psnr:.2f} dB")

    # 4) save artifacts (used later by UMAP/plots)
    out_dir = os.path.join(ROOT, "outputs"); os.makedirs(out_dir, exist_ok=True)
    suffix = args.tag if args.tag else ""
    out_npz = os.path.join(out_dir, f"ae_{dataset_name}_test_processed{suffix}.npz")
    np.savez_compressed(
        out_npz,
        Z=Z, Yhat=Yhat, mse=mse,
        y=(yte.cpu().numpy() if torch.is_tensor(yte) else None),
        config=cfg, dataset=dataset_name,
    )
    print(f"Saved processed test data -> {out_npz}")

if __name__ == "__main__":
    main()