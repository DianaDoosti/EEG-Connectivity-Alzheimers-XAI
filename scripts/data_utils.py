"""
Loading the .npz archive, filtering to the AD-vs-CN binary subset,
per-band normalization (fit on training data only), and the PyTorch
Dataset wrapper.
"""

import json
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

import config


@dataclass
class RawData:
    X: np.ndarray               # (n_epochs, 5, 19, 19), float
    y_bin: np.ndarray           # (n_epochs,), int in {0, 1}
    subject_ids: np.ndarray     # (n_epochs,)
    labels_text: np.ndarray     # (n_epochs,), original text labels
    epoch_numbers: np.ndarray   # (n_epochs,)
    metadata: dict


def load_npz(file_path: str) -> RawData:
    
    with np.load(file_path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        subject_ids = data["subject_ids"]
        labels = data["labels"]
        epoch_numbers = data["epoch_numbers"]
        metadata = json.loads(data["metadata_json"].item())

    # Sanity checks
    assert X.ndim == 4, f"Expected 4D X, got shape {X.shape}"
    assert X.shape[1:] == (5, 19, 19), f"Unexpected X shape {X.shape}"
    assert X.shape[0] == len(y) == len(subject_ids) == len(labels) == len(epoch_numbers)
    assert np.isfinite(X).all(), "X contains non-finite values"
    assert X.min() >= -1e-6 and X.max() <= 1 + 1e-6, "X outside expected wPLI range [0, 1]"

    symmetry_error = np.max(np.abs(X - np.swapaxes(X, -1, -2)))
    assert symmetry_error < 1e-5, f"X is not symmetric (max error {symmetry_error})"
    assert np.allclose(np.diagonal(X, axis1=-2, axis2=-1), 0.0, atol=1e-6), \
        "X diagonal is not zero"

    return RawData(
        X=X,
        y_bin=None,  # filled in by filter_to_binary_task
        subject_ids=subject_ids,
        labels_text=labels,
        epoch_numbers=epoch_numbers,
        metadata=metadata,
    )


def filter_to_binary_task(raw: RawData):
    """
    Keep only epochs belonging to POSITIVE_CLASS_LABEL / NEGATIVE_CLASS_LABEL
    (AD / CN by default), dropping the third diagnostic group (e.g. FTD)
    entirely.Returns a new RawData with y_bin populated (0 = negative class,
    1 = positive class).
    """
    labels_str = np.array([str(v) for v in raw.labels_text])
    keep_mask = np.isin(labels_str, [config.POSITIVE_CLASS_LABEL, config.NEGATIVE_CLASS_LABEL])

    if keep_mask.sum() == 0:
        unique_vals = np.unique(labels_str)
        raise ValueError(
            f"No epochs matched labels {config.POSITIVE_CLASS_LABEL!r}/"
            f"{config.NEGATIVE_CLASS_LABEL!r}. Labels found in the file: "
            f"{unique_vals.tolist()}. Update config.POSITIVE_CLASS_LABEL / "
            f"config.NEGATIVE_CLASS_LABEL to match."
        )

    X = raw.X[keep_mask]
    subject_ids = raw.subject_ids[keep_mask]
    labels_text = labels_str[keep_mask]
    epoch_numbers = raw.epoch_numbers[keep_mask]

    y_bin = np.array([config.CLASS_TO_INT[lbl] for lbl in labels_text], dtype=np.int64)

    # Re-verify one-diagnosis-per-subject on the filtered subset
    
    for subject in np.unique(subject_ids):
        subj_labels = np.unique(labels_text[subject_ids == subject])
        assert len(subj_labels) == 1, f"Subject {subject} has mixed labels after filtering"

    n_subjects = len(np.unique(subject_ids))
    n_pos_subjects = len(np.unique(subject_ids[y_bin == 1]))
    n_neg_subjects = len(np.unique(subject_ids[y_bin == 0]))
    print(
        f"[data] Filtered to binary task "
        f"{config.POSITIVE_CLASS_LABEL} vs {config.NEGATIVE_CLASS_LABEL}: "
        f"{X.shape[0]} epochs from {n_subjects} subjects "
        f"({n_pos_subjects} {config.POSITIVE_CLASS_LABEL}, "
        f"{n_neg_subjects} {config.NEGATIVE_CLASS_LABEL})"
    )

    return RawData(
        X=X,
        y_bin=y_bin,
        subject_ids=subject_ids,
        labels_text=labels_text,
        epoch_numbers=epoch_numbers,
        metadata=raw.metadata,
    )


def fit_band_normalization(X_train: np.ndarray):
    """
    Compute per-band mean/std from TRAINING epochs only, excluding the
    structurally-zero diagonal (which carries no signal and would bias
    the statistics if included).
    Returns (mean, std) each of shape (5,), broadcastable over (N, 5, 19, 19).
    """
    n_channels = X_train.shape[-1]
    off_diag_mask = ~np.eye(n_channels, dtype=bool)   # True everywhere except diagonal

    # X_train[:, :, off_diag_mask] -> (N, 5, n_channels*(n_channels-1))
    off_diag_vals = X_train[:, :, off_diag_mask]

    mean = off_diag_vals.mean(axis=(0, 2))    # (5,)
    std = off_diag_vals.std(axis=(0, 2))      # (5,)
    std = np.where(std < 1e-8, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def apply_band_normalization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean_b = mean.reshape(1, -1, 1, 1)
    std_b = std.reshape(1, -1, 1, 1)
    X_norm = ((X - mean_b) / std_b).astype(np.float32)

    n_channels = X_norm.shape[-1]
    diag_idx = np.arange(n_channels)
    X_norm[:, :, diag_idx, diag_idx] = 0.0

    return X_norm


class ConnectivityDataset(Dataset):
    """Wraps already-normalized (or raw) connectivity matrices for PyTorch."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
