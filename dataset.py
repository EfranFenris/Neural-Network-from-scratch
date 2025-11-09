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