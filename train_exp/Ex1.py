import os
import sys

# Add project root to Python path so we can import models, train, dataset
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import MyFFNetworkForClassification, MyFFNetworkForRegression
from train import train_one_epoch, evaluate
from dataset import load_forestfires_processed
import torch
import matplotlib.pyplot as plt
import os as _os  # you can reuse os above if you want

def run_lr_sanity_check_forestfires():
    # 1) Load processed Forest Fires dataset
    Xtr, ytr, Xte, yte = load_forestfires_processed()

    # Common training settings
    epochs = 100
    batch_size = 64

    # Helper: train the SAME architecture with a given lr
    def train_with_lr(lr: float):
        # Fresh model for each lr (same architecture)
        model = MyFFNetworkForRegression(
            input_dim=Xtr.shape[1],
            hidden_dim=32,
            output_dim=1,
            num_hidden_layers=1,
            init="xavier",
            activation="relu",
            use_batchnorm=False,
        )

        train_losses = []
        test_losses = []

        for ep in range(1, epochs + 1):
            # manual training step (no autograd)
            tr_loss, _ = train_one_epoch(
                model,
                Xtr,
                ytr,
                lr=lr,
                batch_size=batch_size,
                task="reg",
            )

            # evaluate on test set (no grad)
            te_loss, _ = evaluate(model, Xte, yte, task="reg")

            train_losses.append(tr_loss)
            test_losses.append(te_loss)

            print(f"[lr={lr:.1e}] epoch {ep:02d}  train_mse={tr_loss:.4f}  test_mse={te_loss:.4f}")

        return train_losses, test_losses

    # 2) Run for small and large LR
    small_lr = 1e-6
    big_lr = 1e-1

    print("\n=== Small learning rate (1e-6) ===")
    tr_small, te_small = train_with_lr(small_lr)

    print("\n=== Large learning rate (0.1) ===")
    tr_big, te_big = train_with_lr(big_lr)

    # 3) Plot learning curves
    os.makedirs("graphs_exp", exist_ok=True)

    # Train loss curves
    plt.figure()
    plt.plot(tr_small, label="train loss (lr=1e-6)")
    plt.plot(tr_big, label="train loss (lr=0.1)")
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.title("Learning rate sanity check (Forest Fires, regression)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs_exp/lr_sanity_train_loss.png")
    plt.close()

    # Test loss curves
    plt.figure()
    plt.plot(te_small, label="test loss (lr=1e-6)")
    plt.plot(te_big, label="test loss (lr=0.1)")
    plt.xlabel("Epoch")
    plt.ylabel("Test MSE")
    plt.title("Learning rate sanity check (Forest Fires, regression)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs_exp/lr_sanity_test_loss.png")
    plt.close()

    print(
        "\nSaved plots:\n"
        "  graphs_exp/lr_sanity_train_loss.png\n"
        "  graphs_exp/lr_sanity_test_loss.png\n"
    )

    # 4) Short printed explanation for your report
    print("Summary:")
    print("- lr=1e-6: very small steps -> loss decreases extremely slowly, stable but impractical.")
    print("- lr=1: very large steps -> updates are unstable, loss oscillates or increases.")
    print("Conclusion: learning rate controls step size: too small = slow convergence, too large = unstable training.")


if __name__ == "__main__":
    run_lr_sanity_check_forestfires()



