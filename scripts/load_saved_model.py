"""
Utility for loading saved model checkpoints produced by nested_cv.py.

Each checkpoint (models/outer_fold_N_model.pt) contains everything needed
to reconstruct the exact trained model and correctly preprocess new data
for it: weights, full architecture config, winning HP config, and the
per-band normalization stats fit on that fold's training data.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# import smoke_out_config as config 
import config 
from data_utils import apply_band_normalization
from model import EEGConnectivityCNN, build_model_from_config_dict


@dataclass
class LoadedModel:
    model: EEGConnectivityCNN          # reconstructed, weights loaded, .eval() already called
    outer_fold: int
    model_config: dict                 # architecture (conv_config, fc_units, dropout, use_pooling, ...)
    hp_config: dict                    # winning HP combo for this fold
    normalization_mean: Optional[np.ndarray]   # per-band mean, shape (n_bands,), or None
    normalization_std: Optional[np.ndarray]    # per-band std, shape (n_bands,), or None
    final_epochs_trained: int
    class_to_int: dict                 # e.g. {"CN": 0, "AD": 1}
    checkpoint_path: str


def load_checkpoint(path: str, device: Optional[torch.device] = None) -> LoadedModel:
    """
    Load a single checkpoint file and reconstruct the trained model,
    ready for inference (already in eval mode).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = build_model_from_config_dict(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        outer_fold=checkpoint["outer_fold"],
        model_config=checkpoint["model_config"],
        hp_config=checkpoint["hp_config"],
        normalization_mean=checkpoint.get("normalization_mean"),
        normalization_std=checkpoint.get("normalization_std"),
        final_epochs_trained=checkpoint.get("final_epochs_trained"),
        class_to_int=checkpoint.get("class_to_int", config.CLASS_TO_INT),
        checkpoint_path=path,
    )


def load_all_folds(models_dir: Optional[str] = None, device: Optional[torch.device] = None):
    """
    Load every outer_fold_N_model.pt in models_dir.
    Returns a dict: {outer_fold_index: LoadedModel}, sorted by fold index.
    """
    models_dir = models_dir or os.path.join(config.OUTPUT_DIR, "models")

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    checkpoint_files = sorted(
        f for f in os.listdir(models_dir)
        if f.startswith("outer_fold_") and f.endswith("_model.pt")
    )
    if not checkpoint_files:
        raise FileNotFoundError(f"No model checkpoints found in {models_dir}")

    loaded = {}
    for fname in checkpoint_files:
        info = load_checkpoint(os.path.join(models_dir, fname), device=device)
        loaded[info.outer_fold] = info

    return dict(sorted(loaded.items()))


def preprocess_for_model(loaded: LoadedModel, X_raw: np.ndarray) -> np.ndarray:
    """
    Apply the same per-band normalization this model was trained with.
    X_raw: (n_epochs, n_bands, size, size), unnormalized.
    Returns the normalized array, ready to pass to the model.
    """
    if loaded.normalization_mean is None or loaded.normalization_std is None:
        return X_raw.astype(np.float32)
    return apply_band_normalization(X_raw, loaded.normalization_mean, loaded.normalization_std)


@torch.no_grad()
def predict(loaded: LoadedModel, X_raw: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
    """
    Full predict pipeline for new data: normalize using this fold's saved
    stats, run through the model, return softmax probabilities (N, n_classes).
    """
    if device is None:
        device = next(loaded.model.parameters()).device

    X_norm = preprocess_for_model(loaded, X_raw)
    x = torch.from_numpy(X_norm.astype(np.float32)).to(device)
    logits = loaded.model(x)
    return torch.softmax(logits, dim=1).cpu().numpy()


def verify_architecture(loaded: LoadedModel) -> dict:
    """
    Empirically confirms the model's actual behavior rather than trusting
    the saved config alone -- runs a dummy input through just the conv
    stack and reports the real final spatial size. Useful before Grad-CAM
    (need to know the true feature-map resolution for upsampling) or when
    double-checking whether pooling was actually applied.
    """
    device = next(loaded.model.parameters()).device
    in_channels = loaded.model_config["in_channels"]
    input_size = config.INPUT_SIZE

    dummy = torch.zeros(1, in_channels, input_size, input_size, device=device)
    with torch.no_grad():
        features = loaded.model.conv_blocks(dummy)

    return {
        "outer_fold": loaded.outer_fold,
        "declared_use_pooling": loaded.model_config.get("use_pooling", "unknown (older checkpoint)"),
        "conv_config": loaded.model_config["conv_config"],
        "final_feature_map_shape": tuple(features.shape),  # (1, C, H, W)
    }


def print_summary(loaded_models: dict):
    """Prints a readable summary of every loaded fold's model."""
    print(f"{'Fold':<6}{'Conv config':<32}{'FC':<6}{'Dropout':<9}{'LR':<10}{'Batch':<7}"
          f"{'Pooling':<9}{'Final feat. map':<18}{'Epochs':<8}")
    print("-" * 105)
    for fold_idx, loaded in loaded_models.items():
        arch = verify_architecture(loaded)
        hp = loaded.hp_config
        print(
            f"{fold_idx:<6}"
            f"{str(hp['conv_config']):<32}"
            f"{hp.get('fc_units', '-'):<6}"
            f"{hp['dropout']:<9}"
            f"{hp['learning_rate']:<10}"
            f"{hp['batch_size']:<7}"
            f"{str(arch['declared_use_pooling']):<9}"
            f"{str(arch['final_feature_map_shape']):<18}"
            f"{loaded.final_epochs_trained:<8}"
        )


if __name__ == "__main__":
    print(f"[load_saved_models] Loading all checkpoints from: "
          f"{os.path.join(config.OUTPUT_DIR, 'models')}\n")
    models = load_all_folds()
    print_summary(models)
    print(f"\n[load_saved_models] Loaded {len(models)} model(s) successfully.")