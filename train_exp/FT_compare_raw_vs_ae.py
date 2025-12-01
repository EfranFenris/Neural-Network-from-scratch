import os, sys, math, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- project import path ---
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dataset as ds
from models_ae import Autoencoder  # only for loading + encode()

# ----------------- utils -----------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _cfg_get(cfg: dict, k: str, default):
    return cfg[k] if k in cfg else default

def load_autoencoder(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    meta = ck.get("meta", {})
    cfg  = meta.get("config", {})
    D    = meta.get("input_dim", 784)

    # Solo los argumentos que tu Autoencoder sí acepta
    from models_ae import Autoencoder
    ae = Autoencoder(
        input_dim=D,
        enc_dims=cfg["enc_dims"],
        dec_dims=cfg["dec_dims"],
        init=cfg.get("init", "xavier"),
        activation=cfg.get("act", "sigmoid"),
        use_batchnorm=cfg.get("bn", True),
    )

    # Carga de pesos (incluye BN si existe)
    st = ck["state"]
    for i in range(len(ae.W)):
        ae.W[i].copy_(st["W"][i])
        ae.b[i].copy_(st["b"][i])
    if getattr(ae, "gamma", None) is not None and "gamma" in st:
        for i in range(len(ae.gamma)):
            ae.gamma[i].copy_(st["gamma"][i])
            ae.beta[i].copy_(st["beta"][i])
            ae.running_mean[i].copy_(st["running_mean"][i])
            ae.running_var[i].copy_(st["running_var"][i])
    return ae, meta
def split_train_val(Xtr, ytr, val_size=6000):
    N = Xtr.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return Xtr[train_idx], ytr[train_idx], Xtr[val_idx], ytr[val_idx]

# ----------------- simple classifier -----------------
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=10, dropout=0.0):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def accuracy(model, X, y, batch=1024):
    model.eval()
    N = X.size(0)
    correct = 0
    for i in range(0, N, batch):
        xb = X[i:i+batch]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == y[i:i+batch]).sum().item()
    return correct / N

def train_classifier(model, Xtr, ytr, Xva, yva, epochs=40, batch=128, lr=1e-3, wd=1e-4, patience=10):
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    N = Xtr.size(0)

    best_val = 0.0
    best_state = None
    wait = 0
    history = {"train_acc": [], "val_acc": []}

    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            xb, yb = Xtr[idx], ytr[idx]
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # epoch metrics
        tr_acc = accuracy(model, Xtr, ytr)
        va_acc = accuracy(model, Xva, yva)
        history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)
        print(f"[epoch {ep:03d}] train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}")

        # early stopping on val_acc
        if va_acc > best_val + 1e-4:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep} (best val_acc={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val

# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="fashion", choices=["fashion","mnist"])
    p.add_argument("--ckpt", type=str, default=os.path.join(ROOT, "saved_models", "ae_fashion_best.pt"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    set_seed(args.seed)

    graphs = os.path.join(ROOT, "graphs_exp"); os.makedirs(graphs, exist_ok=True)

    # 1) data
    if args.dataset == "fashion":
        Xtr, ytr, Xte, yte = ds.load_fashion_mnist_processed(os.path.join(ROOT, "data/processed"))
    else:
        Xtr, ytr, Xte, yte = ds.load_mnist_processed(os.path.join(ROOT, "data/processed"))
    # split
    Xtr, ytr, Xva, yva = split_train_val(Xtr, ytr, val_size=6000)

    # tensors
    Xtr = Xtr.float(); Xva = Xva.float(); Xte = Xte.float()
    ytr = torch.tensor(np.array(ytr)).long()
    yva = torch.tensor(np.array(yva)).long()
    yte = torch.tensor(np.array(yte)).long()

    D = Xtr.shape[1]
    print(f"Dataset={args.dataset} | D={D}  train={tuple(Xtr.shape)}  val={tuple(Xva.shape)}  test={tuple(Xte.shape)}")

    # 2) AE encode (frozen)
    ae, meta = load_autoencoder(args.ckpt)
    with torch.no_grad():
        Ztr = ae.encode(Xtr).detach()
        Zva = ae.encode(Xva).detach()
        Zte = ae.encode(Xte).detach()
    K = Ztr.shape[1]
    print(f"Loaded AE from: {args.ckpt} | latent_dim K={K}")

    # 3) classifiers: raw vs encoded
    clf_raw = MLPClassifier(in_dim=D, hidden=256, num_classes=10, dropout=0.0)
    print("\n--- Training classifier on RAW pixels ---")
    hist_raw, best_val_raw = train_classifier(clf_raw, Xtr, ytr, Xva, yva,
                                              epochs=args.epochs, batch=args.batch, lr=args.lr, wd=args.wd, patience=10)
    test_acc_raw = accuracy(clf_raw, Xte, yte)
    print(f"[RAW] best_val_acc={best_val_raw:.4f}  test_acc={test_acc_raw:.4f}")

    clf_ae = MLPClassifier(in_dim=K, hidden=256, num_classes=10, dropout=0.0)
    print("\n--- Training classifier on AE features (ENCODED) ---")
    hist_ae, best_val_ae = train_classifier(clf_ae, Ztr, ytr, Zva, yva,
                                            epochs=args.epochs, batch=args.batch, lr=args.lr, wd=args.wd, patience=10)
    test_acc_ae = accuracy(clf_ae, Zte, yte)
    print(f"[ENCODED] best_val_acc={best_val_ae:.4f}  test_acc={test_acc_ae:.4f}")

    # 4) plots
    # accuracy curves
    plt.figure(figsize=(8,4))
    plt.plot(hist_raw["val_acc"], label="raw val_acc")
    plt.plot(hist_ae["val_acc"], label="encoded val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Validation accuracy (raw vs. encoded)")
    plt.legend(); plt.tight_layout()
    out_curve = os.path.join(graphs, f"ft_val_acc_{args.dataset}.png")
    plt.savefig(out_curve, dpi=150); plt.close()
    print("Saved:", out_curve)

    # bar chart for test accuracy
    plt.figure(figsize=(4.5,4))
    plt.bar(["raw", "encoded"], [test_acc_raw, test_acc_ae])
    plt.ylim(0,1); plt.ylabel("test accuracy"); plt.title("Final test accuracy")
    out_bar = os.path.join(graphs, f"ft_test_acc_{args.dataset}.png")
    plt.savefig(out_bar, dpi=150); plt.close()
    print("Saved:", out_bar)

    # also print a short summary
    print("\n=== Summary ===")
    print(f"RAW pixels   : val={best_val_raw:.4f}  test={test_acc_raw:.4f}")
    print(f"AE features  : val={best_val_ae:.4f}  test={test_acc_ae:.4f}")
    print("(Same classifier architecture/hparams for a fair comparison.)")

if __name__ == "__main__":
    main()