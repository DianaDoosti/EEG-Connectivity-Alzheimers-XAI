import os
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# Optional styling for attribution heatmaps
plt.rcParams["figure.dpi"] = 120
plt.rcParams["image.cmap"] = "RdBu_r"

from data_utils import load_npz, filter_to_binary_task

# --------------------------------------------------------------
# Environment & Data Loading Utilities
# --------------------------------------------------------------

def get_code_dir() -> Path:
    """Return the root directory of the current script or execution context."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        pass

    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            nb_file = ip.user_ns.get("__vsc_ipynb_file__")
            if nb_file:
                return Path(nb_file).resolve().parent
    except Exception:
        pass

    return Path(os.getcwd()).resolve()


def load_outer_fold_test_data(
    fold_idx: int,
    data_npz_path: str,
    folds_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load test set epochs and labels for a specific outer cross-validation fold.

    Returns
    -------
    X_test : np.ndarray
        Raw connectivity matrices of shape (n_test_epochs, n_bands, 19, 19).
    y_test : np.ndarray
        Binary targets (0: CN, 1: AD).
    epoch_subject_ids : np.ndarray
        Subject IDs corresponding to each test epoch.
    """
    raw = load_npz(data_npz_path)
    data = filter_to_binary_task(raw)

    X = data.X                 # Shape: (n_epochs, 5, 19, 19)
    y = data.y_bin             # Shape: (n_epochs,)
    subject_ids = data.subject_ids  # Shape: (n_epochs,)

    fold_path = Path(folds_dir) / f"outer_fold_{fold_idx}.json"
    if not fold_path.exists():
        raise FileNotFoundError(f"Fold indices file not found: {fold_path}")

    with open(fold_path, "r") as f:
        fold_info = json.load(f)

    test_subject_ids = set(fold_info["test_subjects"])

    # Match subject IDs (ensuring string comparison)
    subj_str = subject_ids.astype(str)
    test_mask = np.isin(subj_str, list(test_subject_ids))

    X_test = X[test_mask]
    y_test = y[test_mask]
    epoch_subject_ids = subject_ids[test_mask]

    print(
        f"[load_outer_fold_test_data] Fold {fold_idx}: "
        f"{X_test.shape[0]} test epochs across "
        f"{len(np.unique(epoch_subject_ids))} subjects."
    )

    return X_test, y_test, epoch_subject_ids


# --------------------------------------------------------------
# Canonical Channel Definitions
# --------------------------------------------------------------

# Order of 19 10-20 channels enforced during feature extraction (Anterior -> Posterior)
channel_names: List[str] = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2"
]

anterior_to_posterior: List[str] = list(channel_names)

# Re-indexing mapping (Identity since feature extraction already ordered data A->P)
channel_perm: List[int] = [channel_names.index(ch) for ch in anterior_to_posterior]


# --------------------------------------------------------------
# Model Wrapper & Attribution Computation
# --------------------------------------------------------------

import gc
import numpy as np
import torch
from typing import Union, Optional
from captum.attr import IntegratedGradients


class LogitDiffWrapper(torch.nn.Module):
    """
    Wraps binary classifier to output scalar difference: Logit(AD) - Logit(CN).
    Enables target-independent attribution calculation.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)  # Shape: (N, 2)
        # Logit(Class 1 / AD) - Logit(Class 0 / CN)
        return (logits[:, 1] - logits[:, 0]).unsqueeze(1)  # Shape: (N, 1)


def compute_ig_attributions(
    loaded,
    X_raw: np.ndarray,
    n_steps: int = 100,
    target: Union[int, np.ndarray] = 1,
    device: Optional[torch.device] = None,
    use_logit_diff: bool = False,
    batch_size: int = 4,  # <-- Added: Process in small batches to save VRAM/RAM
) -> np.ndarray:
    """
    Computes symmetrized Integrated Gradients attributions in batches.

    Returns
    -------
    A_sym : np.ndarray
        Symmetrized attribution maps of shape (n_epochs, n_bands, 19, 19).
    """
    if len(X_raw) == 0:
        return np.empty((0, 5, 19, 19), dtype=np.float32)

    if device is None:
        device = next(loaded.model.parameters()).device
    
    from load_saved_model import preprocess_for_model

    # Ensure model is in eval mode
    loaded.model.eval()

    # Preprocess full dataset
    X_norm = preprocess_for_model(loaded, X_raw)
    n_epochs = X_norm.shape[0]

    # Setup IG wrapper
    if use_logit_diff:
        model_for_ig = LogitDiffWrapper(loaded.model)
        target_val = None
    else:
        model_for_ig = loaded.model
        if np.isscalar(target):
            target_val = np.full(n_epochs, target, dtype=np.int64)
        else:
            target_val = np.asarray(target, dtype=np.int64)

    ig = IntegratedGradients(model_for_ig)
    attributions_list = []

    # Process in chunks of `batch_size` to prevent Out-Of-Memory (OOM)
    for start_idx in range(0, n_epochs, batch_size):
        end_idx = min(start_idx + batch_size, n_epochs)
        
        # Slice batch and transfer to device
        x_batch = torch.from_numpy(X_norm[start_idx:end_idx].astype(np.float32)).to(device)
        baseline_batch = torch.zeros_like(x_batch)

        if use_logit_diff:
            attr_batch = ig.attribute(
                x_batch,
                baselines=baseline_batch,
                n_steps=n_steps,
                internal_batch_size=batch_size,  # Controls memory during step expansion
            )
        else:
            t_batch = torch.tensor(target_val[start_idx:end_idx], dtype=torch.int64, device=device)
            attr_batch = ig.attribute(
                x_batch,
                target=t_batch,
                baselines=baseline_batch,
                n_steps=n_steps,
                internal_batch_size=batch_size,
            )

        attributions_list.append(attr_batch.detach().cpu().numpy())

        # Free GPU memory explicitly
        del x_batch, baseline_batch, attr_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all batches along the epoch axis
    A = np.concatenate(attributions_list, axis=0)

    # Enforce matrix symmetry for functional connectivity: A_sym = (A + A^T) / 2
    A_sym = (A + A.transpose(0, 1, 3, 2)) / 2.0
    return A_sym


def get_correctly_classified_subset(
    loaded,
    X_raw: np.ndarray,
    y: np.ndarray,
    epoch_subject_ids: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
):
    """
    Evaluates epochs, aggregates predictions per subject using mean P(AD),
    and filters data to keep ONLY correctly predicted subjects.

    Subject-level prediction:
        mean P(AD) >= 0.5 -> AD
        mean P(AD) <  0.5 -> CN
    """

    if device is None:
        device = next(loaded.model.parameters()).device

    from load_saved_model import preprocess_for_model

    loaded.model.eval()
    X_norm = preprocess_for_model(loaded, X_raw)

    # --------------------------------------------------------------
    # Collect epoch-level P(AD)
    # --------------------------------------------------------------
    epoch_probs_ad = []

    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):

            xb = torch.from_numpy(
                X_norm[i:i + batch_size].astype(np.float32)
            ).to(device)

            logits = loaded.model(xb)

            probs = torch.softmax(logits, dim=1)

            epoch_probs_ad.append(
                probs[:, 1].cpu().numpy()
            )

    epoch_probs_ad = np.concatenate(epoch_probs_ad, axis=0)

    # --------------------------------------------------------------
    # Subject-level prediction using mean P(AD)
    # --------------------------------------------------------------
    unique_subjects = np.unique(epoch_subject_ids)

    correct_subject_ids = set()

    for sid in unique_subjects:

        mask = epoch_subject_ids == sid

        true_label = int(y[mask][0])

        mean_prob_ad = float(
            epoch_probs_ad[mask].mean()
        )

        subj_pred = int(mean_prob_ad >= 0.5)

        if subj_pred == true_label:
            correct_subject_ids.add(sid)

    # --------------------------------------------------------------
    # Keep all epochs belonging to correctly classified subjects
    # --------------------------------------------------------------
    correct_epoch_mask = np.isin(
        epoch_subject_ids,
        list(correct_subject_ids)
    )

    return (
        X_raw[correct_epoch_mask],
        y[correct_epoch_mask],
        epoch_subject_ids[correct_epoch_mask],
        correct_subject_ids,
    )

# --------------------------------------------------------------
# Aggregation & Visualization Functions
# --------------------------------------------------------------

def aggregate_attributions_to_subjects(
    attributions: np.ndarray,
    epoch_subject_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Average epoch attributions per subject -> {subj_id: (5, 19, 19)}."""
    unique_subjects = np.unique(epoch_subject_ids)
    subj_maps = {}

    for subj in unique_subjects:
        mask = (epoch_subject_ids == subj)
        maps_subj = attributions[mask]
        subj_maps[subj] = maps_subj.mean(axis=0)

    return subj_maps


def aggregate_group_importance(
    subject_maps: Dict[str, np.ndarray],
    subject_labels: Dict[str, int],
) -> Dict[int, Dict[str, Union[np.ndarray, int]]]:
    """Group-level average of subject attribution maps (signed and absolute)."""
    groups = {0: [], 1: []}
    for subj, m in subject_maps.items():
        label = subject_labels[subj]
        groups[label].append(m)

    result = {}
    for label, maps in groups.items():
        if len(maps) == 0:
            continue
        stack = np.stack(maps, axis=0)
        result[label] = {
            "mean_signed": stack.mean(axis=0),
            "mean_abs": np.abs(stack).mean(axis=0),
            "n_subjects": stack.shape[0],
        }
    return result


def plot_subject_attribution(
    attribution: np.ndarray,
    subject_id: str,
    method: str = "IG",
    channel_labels: Optional[List[str]] = channel_names,
    perm: Optional[List[int]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot band-wise 19x19 electrode connectivity attribution heatmaps."""
    if perm is not None:
        attribution = attribution[:, perm, :][:, :, perm]

    n_bands = attribution.shape[0]
    if vmin is None or vmax is None:
        vmax_auto = float(np.max(np.abs(attribution)))
        vmin = -vmax_auto if vmin is None else vmin
        vmax = vmax_auto if vmax is None else vmax

    fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 3.5))
    if n_bands == 1:
        axes = [axes]

    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"][:n_bands]

    for b, ax in enumerate(axes):
        im = ax.imshow(attribution[b], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"{band_names[b]}\n{method}, Subj: {subject_id}", fontsize=10)
        
        if channel_labels is not None:
            ax.set_xticks(range(len(channel_labels)))
            ax.set_yticks(range(len(channel_labels)))
            ax.set_xticklabels(channel_labels, rotation=90, fontsize=6)
            ax.set_yticklabels(channel_labels, fontsize=6)
        else:
            ax.set_xlabel("Channel Index")
            ax.set_ylabel("Channel Index")
            
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.show()
