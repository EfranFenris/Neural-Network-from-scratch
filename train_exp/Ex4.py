import os
import sys
import torch
import matplotlib.pyplot as plt

# ----------------------------------------
# Import from project root
# ----------------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import MyFFNetworkForClassification
from dataset import load_mnist_processed


# ---------- Helpers ----------

def iterate_minibatches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    idx = torch.arange(N)
    if shuffle:
        idx = idx[torch.randperm(N)]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sel = idx[start:end]
        yield X[sel], y[sel]


def get_params_and_grads(model):
    """
    Return list of (param_tensor, grad_tensor) for all learnable params.
    This matches how BaseNetwork stores things.
    """
    pairs = []
    # Linear layers
    for i in range(model.L_total):
        pairs.append((model.W[i], model.dW[i]))
        pairs.append((model.b[i], model.db[i]))
    # BatchNorm params if used (we keep BN off here, but code is generic)
    if getattr(model, "use_bn", False):
        for i in range(model.L_hidden):
            pairs.append((model.gamma[i], model.dgamma[i]))
            pairs.append((model.beta[i], model.dbeta[i]))
    return pairs


# ---------- Optimizer steps (manual, no autograd) ----------

def sgd_step(model, lr):
    """Classic SGD: p = p - lr * grad"""
    for p, g in get_params_and_grads(model):
        p -= lr * g


def momentum_step(model, state, lr, momentum=0.9):
    """
    SGD with momentum.
    v = mu * v - lr * g
    p = p + v
    state["v"] is a list of tensors with same shapes as params.
    """
    params_grads = get_params_and_grads(model)
    if "v" not in state:
        state["v"] = [torch.zeros_like(p) for (p, g) in params_grads]

    for i, (p, g) in enumerate(params_grads):
        v = state["v"][i]
        v.mul_(momentum).add_(-lr * g)
        p.add_(v)


def adam_step(model, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam optimizer (manual).
    m, v = 1st & 2nd moment estimates.
    """
    params_grads = get_params_and_grads(model)

    if "m" not in state:
        state["m"] = [torch.zeros_like(p) for (p, g) in params_grads]
        state["v"] = [torch.zeros_like(p) for (p, g) in params_grads]
        state["t"] = 0

    state["t"] += 1
    t = state["t"]

    for i, (p, g) in enumerate(params_grads):
        m = state["m"][i]
        v = state["v"][i]

        # m_t = beta1 * m_{t-1} + (1-beta1) * g
        m.mul_(beta1).add_((1.0 - beta1) * g)
        # v_t = beta2 * v_{t-1} + (1-beta2) * g^2
        v.mul_(beta2).add_((1.0 - beta2) * (g * g))

        # bias correction
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        # param update
        p.add_(-lr * m_hat / (torch.sqrt(v_hat) + eps))


# ---------- Train & eval loops for this experiment ----------

def train_one_epoch_with_opt(model, X, y, optimizer_name, opt_state,
                             lr=1e-2, batch_size=128):
    """
    One epoch of manual training with chosen optimizer.
    Returns (avg_loss, avg_acc).
    """
    model.zero_grad()
    N = X.shape[0]
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for xb, yb in iterate_minibatches(X, y, batch_size, shuffle=True):
        # Forward
        y_pred, cache = model.forward(xb, training=True)

        # Loss + dY
        loss, dY = model.loss_and_grad(y_pred, yb)
        # Backward: fills model.dW, model.db, ...
        model.backward(xb, cache, dY)

        # Optimizer step
        if optimizer_name == "sgd":
            sgd_step(model, lr)
        elif optimizer_name == "momentum":
            momentum_step(model, opt_state, lr, momentum=0.9)
        elif optimizer_name == "adam":
            adam_step(model, opt_state, lr, beta1=0.9, beta2=0.999)
        else:
            raise ValueError("Unknown optimizer")

        # Metrics
        bs = xb.shape[0]
        total_loss += float(loss.item()) * bs
        total_seen += bs
        preds = y_pred.argmax(dim=1)
        total_correct += int((preds == yb).sum().item())

    avg_loss = total_loss / total_seen
    avg_acc = total_correct / total_seen
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, X, y):
    y_pred, _ = model.forward(X, training=False)
    # reuse loss_and_grad just to get the scalar loss (ignore grad)
    loss, _ = model.loss_and_grad(y_pred, y)
    acc = float((y_pred.argmax(dim=1) == y).float().mean().item())
    return float(loss.item()), acc


def build_base_model():
    """Same architecture for all optimizers."""
    return MyFFNetworkForClassification(
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        num_hidden_layers=2,
        init="he",
        activation="relu",
        use_batchnorm=False,
    )


def run_for_optimizer(name, Xtr, ytr, Xte, yte, epochs, lr, batch_size):
    """
    Train a fresh model with the given optimizer.
    Returns dict with train/test loss & acc curves.
    """
    # Make runs comparable: fix seed before building
    torch.manual_seed(0)
    model = build_base_model()
    opt_state = {}

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch_with_opt(
            model, Xtr, ytr,
            optimizer_name=name,
            opt_state=opt_state,
            lr=lr,
            batch_size=batch_size,
        )
        te_loss, te_acc = evaluate(model, Xte, yte)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        print(f"[{name.upper()}][{ep:02d}] "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
              f"test_loss={te_loss:.4f}  test_acc={te_acc:.3f}")

    return {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "test_loss": test_losses,
        "test_acc": test_accs,
    }


# ---------- Main experiment ----------

def main():
    torch.manual_seed(0)
    os.makedirs(os.path.join(ROOT, "graphs_exp"), exist_ok=True)

    # Load MNIST (processed tensors)
    Xtr, ytr, Xte, yte = load_mnist_processed()

    # Subsample for speed
    max_train = 2000
    max_test = 500
    if Xtr.shape[0] > max_train:
        Xtr = Xtr[:max_train]
        ytr = ytr[:max_train]
    if Xte.shape[0] > max_test:
        Xte = Xte[:max_test]
        yte = yte[:max_test]

    print(f"Using {Xtr.shape[0]} train and {Xte.shape[0]} test samples.")

    epochs = 10
    batch_size = 128

    # Learning rates tuned a bit so they all behave:
    results = {}
    results["sgd"] = run_for_optimizer("sgd", Xtr, ytr, Xte, yte,
                                       epochs=epochs, lr=1e-2,
                                       batch_size=batch_size)
    results["momentum"] = run_for_optimizer("momentum", Xtr, ytr, Xte, yte,
                                            epochs=epochs, lr=1e-2,
                                            batch_size=batch_size)
    results["adam"] = run_for_optimizer("adam", Xtr, ytr, Xte, yte,
                                        epochs=epochs, lr=1e-3,
                                        batch_size=batch_size)

    # ----- Plot: train loss -----
    plt.figure()
    for name, label in [("sgd", "SGD"),
                        ("momentum", "SGD+Momentum"),
                        ("adam", "Adam")]:
        plt.plot(results[name]["train_loss"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Ex4: Optimizers — train loss")
    plt.legend()
    plt.tight_layout()
    f_train = os.path.join(ROOT, "graphs_exp", "ex4_optim_train_loss.png")
    plt.savefig(f_train)
    plt.close()

    # ----- Plot: test loss -----
    plt.figure()
    for name, label in [("sgd", "SGD"),
                        ("momentum", "SGD+Momentum"),
                        ("adam", "Adam")]:
        plt.plot(results[name]["test_loss"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Test loss")
    plt.title("Ex4: Optimizers — test loss")
    plt.legend()
    plt.tight_layout()
    f_test = os.path.join(ROOT, "graphs_exp", "ex4_optim_test_loss.png")
    plt.savefig(f_test)
    plt.close()

    print("\nSaved plots:")
    print(" ", f_train)
    print(" ", f_test)
    print("\nUse them to comment how Momentum and Adam compare to plain SGD.")


if __name__ == "__main__":
    main()