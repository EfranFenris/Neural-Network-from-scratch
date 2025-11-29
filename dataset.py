# dataset.py
# Forest Fires dataset helpers.
#
# Two main things:
#   1) make_forestfires_processed(...)  -> create data/processed/forestfires_processed.csv
#   2) load_forestfires_processed(...)  -> return train/test tensors for experiments
#
# This keeps train_exp.py very simple.

import os
import numpy as np
import pandas as pd
import torch

# -------- 1) Build processed dataset from raw --------

def make_forestfires_processed(
    raw_path: str = "data/raw/forestfires.csv",
    processed_path: str = "data/processed/forestfires_processed.csv",
    log_target: bool = True,
) -> str:
    """
    Read the raw Forest Fires CSV, encode categorical features,
    and save a numeric processed CSV.

    Processed file will contain:
      - all input features as numeric columns
      - 'target' column = area (or log(1+area) if log_target=True)

    Returns:
        processed_path (str)
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found at {raw_path}")

    df = pd.read_csv(raw_path)

    if "area" not in df.columns:
        raise ValueError("Expected 'area' column in forestfires.csv")

    # One-hot encode month and day => new numeric columns month_xx, day_xx
    df = pd.get_dummies(df, columns=["month", "day"], drop_first=False)

    # Extract target
    y = df["area"].astype("float32").values
    if log_target:
        y = np.log1p(y)  # log(1 + area) to reduce skew
    df_features = df.drop(columns=["area"])

    # Build processed DataFrame: all features + single 'target' column
    df_proc = df_features.copy()
    df_proc["target"] = y

    # Ensure output directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    # Save
    df_proc.to_csv(processed_path, index=False)
    print(f"Saved processed Forest Fires dataset to {processed_path}")
    return processed_path


# -------- 2) Load processed dataset as tensors --------

def load_forestfires_processed(
    processed_path: str = "data/processed/forestfires_processed.csv",
    test_ratio: float = 0.2,
    seed: int = 0,
    standardize: bool = True,
):
    """
    Load the processed Forest Fires dataset and return train/test tensors.

    Assumes processed CSV was created by make_forestfires_processed() and has:
      - all feature columns (numeric)
      - 'target' column (already log-transformed if desired)

    Returns:
        X_tr (torch.FloatTensor): [N_train, D_in]
        y_tr (torch.FloatTensor): [N_train, 1]
        X_te (torch.FloatTensor): [N_test, D_in]
        y_te (torch.FloatTensor): [N_test, 1]
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed file not found at {processed_path}. "
            f"Run make_forestfires_processed() first."
        )

    df = pd.read_csv(processed_path)

    if "target" not in df.columns:
        raise ValueError("Processed file must contain a 'target' column.")

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values.astype("float32")
    y = df["target"].values.astype("float32").reshape(-1, 1)

    N = X.shape[0]
    torch.manual_seed(seed)
    perm = torch.randperm(N)
    n_tr = int((1.0 - test_ratio) * N)
    idx_tr = perm[:n_tr]
    idx_te = perm[n_tr:]

    X_tr = torch.from_numpy(X)[idx_tr]
    y_tr = torch.from_numpy(y)[idx_tr]
    X_te = torch.from_numpy(X)[idx_te]
    y_te = torch.from_numpy(y)[idx_te]

    if standardize:
        mean = X_tr.mean(dim=0, keepdim=True)
        std = X_tr.std(dim=0, keepdim=True)
        std[std == 0] = 1.0  # avoid division by zero
        X_tr = (X_tr - mean) / std
        X_te = (X_te - mean) / std

    return X_tr.float(), y_tr.float(), X_te.float(), y_te.float()


# Only create the processed dataset if this script is run directly
if __name__ == "__main__":
    make_forestfires_processed()


# ================================
# MNIST (CSV) helpers
# ================================
# Raw files expected:
#   data/raw/mnist_train.csv
#   data/raw/mnist_test.csv
#
# Format:
#   each row: [label, p0, p1, ..., p783]
#   label in 0..9
#   pixel values in 0..255
#
# We will create:
#   data/processed/mnist_train.pt
#   data/processed/mnist_test.pt
#
# Each .pt file is:
#   {"X": FloatTensor [N, 784] in [0,1],
#    "y": LongTensor  [N]}
# ================================




# ================================
# MNIST (CSV) helpers
# ================================
import os
import torch
import pandas as pd


def make_mnist_processed(
    raw_train_path: str = "data/raw/mnist_train.csv",
    raw_test_path: str = "data/raw/mnist_test.csv",
    processed_dir: str = "data/processed",
):
    """
    Convert raw MNIST CSVs (with header row: label,pixel0,...)
    into normalized tensors and save as .pt files.

    Output:
      data/processed/mnist_train.pt
      data/processed/mnist_test.pt

    Each .pt file:
      {"X": FloatTensor [N, 784] in [0,1],
       "y": LongTensor  [N]}
    """
    if not os.path.exists(raw_train_path):
        raise FileNotFoundError(f"MNIST train CSV not found at {raw_train_path}")
    if not os.path.exists(raw_test_path):
        raise FileNotFoundError(f"MNIST test CSV not found at {raw_test_path}")

    os.makedirs(processed_dir, exist_ok=True)

    # ---- Train ----
    # header is present, so don't use header=None
    df_tr = pd.read_csv(raw_train_path)
    if "label" not in df_tr.columns:
        raise ValueError("Expected a 'label' column in MNIST train CSV.")
    y_tr = df_tr["label"].astype("int64").values
    X_tr = df_tr.drop(columns=["label"]).astype("float32").values / 255.0

    X_tr = torch.from_numpy(X_tr)
    y_tr = torch.from_numpy(y_tr)

    torch.save({"X": X_tr, "y": y_tr},
               os.path.join(processed_dir, "mnist_train.pt"))

    # ---- Test ----
    df_te = pd.read_csv(raw_test_path)
    if "label" not in df_te.columns:
        raise ValueError("Expected a 'label' column in MNIST test CSV.")
    y_te = df_te["label"].astype("int64").values
    X_te = df_te.drop(columns=["label"]).astype("float32").values / 255.0

    X_te = torch.from_numpy(X_te)
    y_te = torch.from_numpy(y_te)

    torch.save({"X": X_te, "y": y_te},
               os.path.join(processed_dir, "mnist_test.pt"))

    print(f"Saved processed MNIST to {processed_dir}/mnist_train.pt and mnist_test.pt")


def load_mnist_processed(processed_dir: str = "data/processed"):
    """
    Load processed MNIST tensors, creating them if needed.

    Returns:
        X_tr: [N_train, 784] float32 in [0,1]
        y_tr: [N_train]      int64 labels 0..9
        X_te: [N_test, 784]
        y_te: [N_test]
    """
    train_pt = os.path.join(processed_dir, "mnist_train.pt")
    test_pt = os.path.join(processed_dir, "mnist_test.pt")

    if not (os.path.exists(train_pt) and os.path.exists(test_pt)):
        make_mnist_processed(
            raw_train_path=os.path.join("data", "raw", "mnist_train.csv"),
            raw_test_path=os.path.join("data", "raw", "mnist_test.csv"),
            processed_dir=processed_dir,
        )

    tr = torch.load(train_pt)
    te = torch.load(test_pt)

    return tr["X"].float(), tr["y"].long(), te["X"].float(), te["y"].long()
# ================================
# Loading affNIST centered dataset (for Ex5)
# ================================

import os
import numpy as np
import torch
import scipy.io as spio

def _mat_to_dict(path):
    """Load .mat and convert MATLAB structs to plain dicts."""
    data = spio.loadmat(path, struct_as_record=False, squeeze_me=True)

    def _to_dict(obj):
        if isinstance(obj, spio.matlab.mio5_params.mat_struct):
            return {name: _to_dict(getattr(obj, name)) for name in obj._fieldnames}
        return obj

    return {k: _to_dict(v) for k, v in data.items()}


def make_affnist_centered_processed(
    mat_path: str = "data/raw/just_centered/training_and_validation.mat",
    processed_dir: str = "data/processed",
    train_ratio: float = 0.8,
    seed: int = 0,
) -> None:
    """
    Build cached tensors from affNIST 'just_centered' and save:
      data/processed/affnist_centered_train.pt
      data/processed/affnist_centered_test.pt
    """
    os.makedirs(processed_dir, exist_ok=True)
    dd = _mat_to_dict(mat_path)["affNISTdata"]

    # X: (1600, N) -> (N,1600), normalize to [0,1]
    X = np.asarray(dd["image"], dtype=np.float32).T
    if X.max() > 1.5:
        X = X / 255.0

    # y: sometimes '0' is encoded as 10 -> map back to 0
    y = np.asarray(dd["label_int"]).astype(np.int64)
    if y.max() == 10:
        y[y == 10] = 0

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    # split
    N = X.shape[0]
    torch.manual_seed(seed)
    perm = torch.randperm(N)
    n_tr = int(train_ratio * N)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]

    torch.save({"X": X[tr_idx], "y": y[tr_idx]},
               os.path.join(processed_dir, "affnist_centered_train.pt"))
    torch.save({"X": X[te_idx], "y": y[te_idx]},
               os.path.join(processed_dir, "affnist_centered_test.pt"))

    print(f"Saved processed affNIST to {processed_dir}/affnist_centered_train.pt "
          f"and {processed_dir}/affnist_centered_test.pt")


def load_affnist_centered_processed(
    processed_dir: str = "data/processed",
    mat_path: str = "data/raw/just_centered/training_and_validation.mat",
    train_ratio: float = 0.8,
    seed: int = 0,
    subsample_train: int | None = None,
    subsample_test: int | None = None,
):
    """
    Load cached affNIST centered tensors; build them from the .mat if missing.

    Returns:
        X_tr: [N_tr, 1600] float32 in [0,1]
        y_tr: [N_tr]       int64
        X_te: [N_te, 1600]
        y_te: [N_te]
    """
    tr_pt = os.path.join(processed_dir, "affnist_centered_train.pt")
    te_pt = os.path.join(processed_dir, "affnist_centered_test.pt")

    if not (os.path.exists(tr_pt) and os.path.exists(te_pt)):
        make_affnist_centered_processed(
            mat_path=mat_path,
            processed_dir=processed_dir,
            train_ratio=train_ratio,
            seed=seed,
        )

    tr = torch.load(tr_pt)
    te = torch.load(te_pt)
    Xtr, ytr = tr["X"].float(), tr["y"].long()
    Xte, yte = te["X"].float(), te["y"].long()

    # optional fast runs
    if subsample_train is not None and Xtr.shape[0] > subsample_train:
        Xtr, ytr = Xtr[:subsample_train], ytr[:subsample_train]
    if subsample_test is not None and Xte.shape[0] > subsample_test:
        Xte, yte = Xte[:subsample_test], yte[:subsample_test]

    return Xtr, ytr, Xte, yte


# ================================
# Fashion-MNIST (CSV) helpers
# ================================
import os
import torch
import pandas as pd

def make_fashion_mnist_processed(
    raw_train_path: str = "data/raw/fashion-mnist_train.csv",
    raw_test_path: str  = "data/raw/fashion-mnist_test.csv",
    processed_dir: str  = "data/processed",
) -> None:
    """
    Convert Fashion-MNIST CSVs (label + 784 pixels) into normalized tensors and
    save as:
        data/processed/fashion_mnist_train.pt
        data/processed/fashion_mnist_test.pt
    Each file: {"X": FloatTensor [N,784] in [0,1], "y": LongTensor [N]}
    """
    if not os.path.exists(raw_train_path):
        raise FileNotFoundError(f"Train CSV not found: {raw_train_path}")
    if not os.path.exists(raw_test_path):
        raise FileNotFoundError(f"Test CSV not found: {raw_test_path}")

    os.makedirs(processed_dir, exist_ok=True)

    # ---- train ----
    df_tr = pd.read_csv(raw_train_path)
    if "label" not in df_tr.columns:
        raise ValueError("Expected a 'label' column in fashion-mnist_train.csv")
    y_tr = df_tr["label"].astype("int64").values
    X_tr = df_tr.drop(columns=["label"]).astype("float32").values / 255.0
    torch.save({"X": torch.from_numpy(X_tr), "y": torch.from_numpy(y_tr)},
               os.path.join(processed_dir, "fashion_mnist_train.pt"))

    # ---- test ----
    df_te = pd.read_csv(raw_test_path)
    if "label" not in df_te.columns:
        raise ValueError("Expected a 'label' column in fashion-mnist_test.csv")
    y_te = df_te["label"].astype("int64").values
    X_te = df_te.drop(columns=["label"]).astype("float32").values / 255.0
    torch.save({"X": torch.from_numpy(X_te), "y": torch.from_numpy(y_te)},
               os.path.join(processed_dir, "fashion_mnist_test.pt"))

    print(f"Saved processed Fashion-MNIST to {processed_dir}/fashion_mnist_*.pt")


def load_fashion_mnist_processed(
    processed_dir: str = "data/processed",
):
    """
    Load (or build if missing) Fashion-MNIST tensors.

    Returns:
        X_tr: [N_train, 784] float32 in [0,1]
        y_tr: [N_train]      int64 labels 0..9
        X_te: [N_test, 784]
        y_te: [N_test]
    """
    train_pt = os.path.join(processed_dir, "fashion_mnist_train.pt")
    test_pt  = os.path.join(processed_dir, "fashion_mnist_test.pt")

    if not (os.path.exists(train_pt) and os.path.exists(test_pt)):
        make_fashion_mnist_processed()  # uses default raw paths

    tr = torch.load(train_pt)
    te = torch.load(test_pt)
    return tr["X"].float(), tr["y"].long(), te["X"].float(), te["y"].long()


def load_fashion_mnist_for_ae(
    processed_dir: str = "data/processed",
    val_ratio: float = 0.1,
    seed: int = 0,
    limit_train: int | None = None,
    limit_test: int | None = None,
):
    """
    Convenience loader for autoencoders (reconstruction task).
    Splits train into train/val and returns inputs only (labels also returned for logging).

    Returns:
        X_tr, X_val, X_te  (float32 in [0,1])
        y_tr, y_val, y_te  (int64 labels, optional for monitoring)
    """
    X_tr, y_tr, X_te, y_te = load_fashion_mnist_processed(processed_dir)

    if limit_train is not None and X_tr.shape[0] > limit_train:
        X_tr, y_tr = X_tr[:limit_train], y_tr[:limit_train]
    if limit_test is not None and X_te.shape[0] > limit_test:
        X_te, y_te = X_te[:limit_test], y_te[:limit_test]

    torch.manual_seed(seed)
    N = X_tr.shape[0]
    idx = torch.randperm(N)
    n_val = int(val_ratio * N)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr_, y_tr_   = X_tr[tr_idx], y_tr[tr_idx]
    X_val_, y_val_ = X_tr[val_idx], y_tr[val_idx]

    return X_tr_, X_val_, X_te, y_tr_, y_val_, y_te