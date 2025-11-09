# train.py — BEGINNER + SANITY CHECKS
# Works with your manual models (no autograd). Produces:
#  - Learning curves (loss/acc) saved as PNGs
#  - Simple LR sanity check (very small vs very large LR)
#
# HOW TO RUN (inside your venv):
#   pip install matplotlib   # once
#   python train.py
#
import torch
import math

# Matplotlib is only for plots (evidence). Models still do manual backward.
import matplotlib.pyplot as plt
_HAVE_PLT = True


from models import MyFFNetworkForClassification, MyFFNetworkForRegression


def train_one_epoch(model, X, y, lr: float, batch_size: int, task: str):
    """
    Runs one epoch of training on the given model and data using mini-batch SGD.
    Args:
        model: The neural network (manual forward/backward/step).
        X: Input features (tensor).
        y: Target labels or values (tensor).
        lr: Learning rate for SGD.
        batch_size: Mini-batch size.
        task: 'clf' for classification, 'reg' for regression.
    Returns:
        avg_loss: Average loss over the epoch.
        acc_or_None: Accuracy (classification) or None (regression).
    """
    model.zero_grad()
    N = X.shape[0]
    perm = torch.randperm(N)
    total_loss, total_seen = 0.0, 0
    total_correct = 0  # used only for classification

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = perm[start:end]
        xb, yb = X[idx], y[idx]

        # 1) Forward (manual in your model)
        y_pred, cache = model.forward(xb, training=True)

        # 2) Loss and gradient at output (no autograd)
        loss, dY = model.loss_and_grad(y_pred, yb)

        # 3) Backward + SGD step (manual)
        model.backward(xb, cache, dY)
        model.step(lr)

        # 4) Metrics
        bs = xb.shape[0]
        total_loss += float(loss.item()) * bs
        total_seen += bs
        if task == "clf":
            preds = y_pred.argmax(dim=1)
            total_correct += int((preds == yb).sum().item())

    avg_loss = total_loss / total_seen
    if task == "clf": #clf means classification
        return avg_loss, total_correct / total_seen
    else:
        return avg_loss, None


@torch.no_grad()
def evaluate(model, X, y, task: str):
    """
    Evaluates the model on the given data in evaluation mode (no gradient tracking).
    Args:
        model: The neural network.
        X: Input features (tensor).
        y: Target labels or values (tensor).
        task: 'clf' for classification, 'reg' for regression.
    Returns:
        loss: Loss value (cross-entropy or MSE).
        acc_or_mse: Accuracy (classification) or MSE (regression).
    """
    y_pred, _ = model.forward(X, training=False)
    if task == "clf":
        # Cross-entropy loss value for info (uses model's helper)
        loss, _ = model.loss_and_grad(y_pred, y)  # returns grad too, ignore it
        acc = float((y_pred.argmax(dim=1) == y).float().mean().item())
        return float(loss.item()), acc #.item() converts single-value tensor to Python float
    else:
        # MSE.   
        diff = y_pred - y
        mse = float((diff ** 2).mean().item())
        return mse, mse  # keep interface consistent


def fit(model, Xtr, ytr, Xte, yte, task: str, epochs: int, lr: float, batch_size: int):
    """
    Trains the model for a given number of epochs, tracking learning curves.
    Args:
        model: The neural network.
        Xtr, ytr: Training data and labels.
        Xte, yte: Test/validation data and labels.
        task: 'clf' or 'reg'.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
    Returns:
        train_losses, train_metrics, test_losses, test_metrics: Lists of loss/metric values per epoch.
    """
    train_losses, train_metrics = [], []
    test_losses, test_metrics = [], []

    for ep in range(1, epochs + 1):
        tr_loss, tr_metric = train_one_epoch(model, Xtr, ytr, lr, batch_size, task)
        te_loss, te_metric = evaluate(model, Xte, yte, task)

        train_losses.append(tr_loss)
        train_metrics.append(tr_metric if tr_metric is not None else tr_loss)
        test_losses.append(te_loss)
        test_metrics.append(te_metric)

        if task == "clf":
            print(f"[{ep:02d}] train_loss={tr_loss:.4f}  train_acc={tr_metric:.3f}  test_acc={te_metric:.3f}")
        else:
            print(f"[{ep:02d}] train_mse={tr_loss:.4f}  test_mse={te_metric:.4f}")

    return (train_losses, train_metrics, test_losses, test_metrics)


def plot_learning_curves(title, train_losses, test_losses, train_metric, test_metric, metric_name, outfile):
    """
    Plots and saves learning curves (loss and metric) for training and test sets.
    Args:
        title: Plot title.
        train_losses, test_losses: Lists of loss values per epoch.
        train_metric, test_metric: Lists of metric values (acc or mse) per epoch.
        metric_name: Name of the metric ('acc' or 'mse').
        outfile: Base filename for saving plots (will be placed in graphs_toy/).
    """
    if not _HAVE_PLT:
        return
    import os
    outdir = "graphs_toy"
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, os.path.basename(outfile))
    loss_path = base.replace(".png", "_loss.png")
    metric_path = base.replace(".png", f"_{metric_name}.png")

    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"{title} — Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path); plt.close()

    plt.figure()
    plt.plot(train_metric, label=f"train {metric_name}")
    plt.plot(test_metric, label=f"test {metric_name}")
    plt.xlabel("epoch"); plt.ylabel(metric_name); plt.title(f"{title} — {metric_name}"); plt.legend()
    plt.tight_layout()
    plt.savefig(metric_path); plt.close()
    print(f"Saved plots to: {loss_path} and {metric_path}")


def run_classification_demo():
    """Binary classification: inside/outside circle (nonlinear)."""
    torch.manual_seed(0)
    N = 400
    X = torch.randn(N, 2)
    
    print(f"X.shape = {X.shape} (N={N})")
    y = (X.norm(dim=1) > 1.0).long()  # 0 or 1

    # Split 80/20
    idx = torch.randperm(N)
    n_tr = int(0.8 * N)
    tr, te = idx[:n_tr], idx[n_tr:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # Simple 1-hidden-layer MLP (BN off to keep code easy)
    model = MyFFNetworkForClassification(
        input_dim=2, hidden_dim=12, output_dim=2,
        num_hidden_layers=1, init="he", activation="relu", use_batchnorm=False
    )

    # Train
    curves = fit(model, Xtr, ytr, Xte, yte, task="clf", epochs=100, lr=1e-1, batch_size=64)
    tr_loss, tr_metric, te_loss, te_metric = curves

    # Plot
    plot_learning_curves(
        title="Classification (circle)",
        train_losses=tr_loss, test_losses=te_loss,
        train_metric=tr_metric, test_metric=te_metric,
        metric_name="acc", outfile="clf_curves.png"
    )


def run_regression_demo():
    """Regression: y = 2*x1 - 3*x2 + noise (easy sanity check)."""
    torch.manual_seed(0)
    N = 500
    X = torch.randn(N, 2)
    y = (2 * X[:, :1] - 3 * X[:, 1:2]) + 0.1 * torch.randn(N, 1)  # shape [N,1]

    # Split 80/20
    idx = torch.randperm(N)
    n_tr = int(0.8 * N)
    tr, te = idx[:n_tr], idx[n_tr:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    model = MyFFNetworkForRegression(
        input_dim=2, hidden_dim=16, output_dim=1,
        num_hidden_layers=1, init="xavier", activation="relu", use_batchnorm=False
    )

    curves = fit(model, Xtr, ytr, Xte, yte, task="reg", epochs=100, lr=5e-3, batch_size=64)
    tr_loss, tr_metric, te_loss, te_metric = curves

    # For regression, metric == loss (MSE), so we still save two plots for consistency.
    plot_learning_curves(
        title="Regression (linear target)",
        train_losses=tr_loss, test_losses=te_loss,
        train_metric=tr_metric, test_metric=te_metric,
        metric_name="mse", outfile="reg_curves.png"
    )



if __name__ == "__main__":
    # 1) Standard runs with curves (save PNGs)
    run_classification_demo()
    run_regression_demo()



    print("\nDone. If matplotlib is installed, you should see PNGs:\n"
          " - clf_curves_loss.png, clf_curves_acc.png\n"
          " - reg_curves_loss.png, reg_curves_mse.png\n"
          " - lr_sanity_clf.png\n")
