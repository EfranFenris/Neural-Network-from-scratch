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
from train import train_one_epoch, evaluate
from dataset import load_mnist_processed


def build_model(use_bn: bool):
    """Same architecture; only differ in BatchNorm flag."""
    return MyFFNetworkForClassification(
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        num_hidden_layers=2,
        init="he",            # as requested
        activation="relu",    # as requested
        use_batchnorm=use_bn, # key difference
    )


def run_experiment(Xtr, ytr, Xte, yte,
                   lr=1e-2, epochs=10, batch_size=128):
    """Train model without BN and with BN, collect curves."""

    # Model A: no BatchNorm
    model_no_bn = build_model(use_bn=False)
    no_bn_train_loss, no_bn_test_loss = [], []

    # Model B: with BatchNorm
    model_bn = build_model(use_bn=True)
    bn_train_loss, bn_test_loss = [], []

    # ----- Train both models in parallel (same epochs) -----
    for ep in range(1, epochs + 1):
        # --- No BN ---
        tr_loss_a, _ = train_one_epoch(
            model_no_bn, Xtr, ytr,
            lr=lr, batch_size=batch_size, task="clf"
        )
        te_loss_a, te_acc_a = evaluate(model_no_bn, Xte, yte, task="clf")
        no_bn_train_loss.append(tr_loss_a)
        no_bn_test_loss.append(te_loss_a)

        # --- With BN ---
        tr_loss_b, _ = train_one_epoch(
            model_bn, Xtr, ytr,
            lr=lr, batch_size=batch_size, task="clf"
        )
        te_loss_b, te_acc_b = evaluate(model_bn, Xte, yte, task="clf")
        bn_train_loss.append(tr_loss_b)
        bn_test_loss.append(te_loss_b)

        print(
            f"[epoch {ep:02d}] "
            f"NoBN: train={tr_loss_a:.4f}, test={te_loss_a:.4f}, acc={te_acc_a:.3f} | "
            f"BN: train={tr_loss_b:.4f}, test={te_loss_b:.4f}, acc={te_acc_b:.3f}"
        )

    return (no_bn_train_loss, no_bn_test_loss,
            bn_train_loss, bn_test_loss)


def main():
    torch.manual_seed(0)
    os.makedirs(os.path.join(ROOT, "graphs_exp"), exist_ok=True)

    # ----- Load MNIST -----
    Xtr, ytr, Xte, yte = load_mnist_processed()

    # Subsample for speed (enough to see effect)
    max_train = 20000
    max_test = 5000
    if Xtr.shape[0] > max_train:
        Xtr = Xtr[:max_train]
        ytr = ytr[:max_train]
    if Xte.shape[0] > max_test:
        Xte = Xte[:max_test]
        yte = yte[:max_test]

    print(f"Using {Xtr.shape[0]} train and {Xte.shape[0]} test samples.")

    # ----- Run experiment -----
    (no_bn_tr, no_bn_te,
     bn_tr, bn_te) = run_experiment(Xtr, ytr, Xte, yte,
                                    lr=1e-2, epochs=10, batch_size=128)

    # ----- Plot train loss -----
    plt.figure()
    plt.plot(no_bn_tr, label="No BN - train")
    plt.plot(bn_tr, label="With BN - train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Ex3: BatchNorm — train loss")
    plt.legend()
    plt.tight_layout()
    out_train = os.path.join(ROOT, "graphs_exp", "ex3_bn_train_loss.png")
    plt.savefig(out_train)
    plt.close()

    # ----- Plot test loss -----
    plt.figure()
    plt.plot(no_bn_te, label="No BN - test")
    plt.plot(bn_te, label="With BN - test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Ex3: BatchNorm — test loss")
    plt.legend()
    plt.tight_layout()
    out_test = os.path.join(ROOT, "graphs_exp", "ex3_bn_test_loss.png")
    plt.savefig(out_test)
    plt.close()

    print("\nSaved plots:")
    print(" ", out_train)
    print(" ", out_test)
    print("\nNow you can compare: with BN vs without BN.")


if __name__ == "__main__":
    main()

#Notes about BatchNorm from
#Batch Normalization normalizes the pre-activation values of each neuron within a mini-batch 
# to have zero mean and unit variance, then applies a learnable scale (\gamma) and shift (\beta).
#  This keeps activations in a stable range across layers, which stabilizes and amplifies 
# gradient flow, makes the loss surface smoother, and lets us safely use larger learning rates. 
# As a side effect, the use of batch statistics injects a bit of noise that acts as regularization.
#  In practice, this usually leads to faster and more stable training compared to the same network without BN.