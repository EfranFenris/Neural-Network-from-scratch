import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(__file__))

def py():
    cand = os.path.join(ROOT, ".venv", "bin", "python")
    return cand if os.path.exists(cand) else (os.environ.get("PYTHON","python3"))

def run(cmd):
    print("$", " ".join(cmd)); subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run([py(), os.path.join(ROOT, "train_exp", "AE_process_test.py"),
         "--ckpt", os.path.join(ROOT, "saved_models", "ae_fashion_best.pt"), "--tag", "_base"])
    run([py(), os.path.join(ROOT, "train_exp", "AE_process_test.py"),
         "--ckpt", os.path.join(ROOT, "saved_models", "ae_fashion_best_reg.pt"), "--tag", "_reg"])
    run([py(), os.path.join(ROOT, "train_exp", "AE_umap_compare.py")])
    print("✓ Created graphs_exp/ae_fashion_umap_compare.png and graphs_exp/ae_fashion_recon_compare.png")