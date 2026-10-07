"""
Single train/validate run for one hyperparameter configuration.
This is the building block called repeatedly by the nested CV loop.
"""

import copy
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data_utils import ConnectivityDataset
from model import EEGConnectivityCNN
from reproducibility import make_generator, seed_worker


@dataclass
class TrainResult:
    model: EEGConnectivityCNN
    best_epoch: int
    best_val_loss: float
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)


def compute_class_weights(y_train: np.ndarray, n_classes: int, device) -> torch.Tensor:
    """Inverse-frequency class weights computed from the training fold only."""
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)  # guard divide-by-zero
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    conv_config,
    fc_units,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    use_pooling: bool,
    device: torch.device,
    seed: int,
    max_epochs: int = None,
    verbose: bool = False,
) -> TrainResult:
    
    """
    Train one model with one HP config, validating each epoch, and keep
    the weights from the epoch with the lowest validation loss
    (early stopping with patience, LR reduced on plateau).
    """
    max_epochs = max_epochs or config.MAX_EPOCHS

    train_ds = ConnectivityDataset(X_train, y_train)
    val_ds = ConnectivityDataset(X_val, y_val)

    gen = make_generator(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        generator=gen,
        worker_init_fn=seed_worker if config.NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = EEGConnectivityCNN(
        conv_config=conv_config,
        fc_units=fc_units,
        dropout=dropout,
        in_channels=config.IN_CHANNELS,
        n_classes=config.N_CLASSES,
        use_pooling=use_pooling,
        input_size=config.INPUT_SIZE,
    ).to(device)

    class_weights = compute_class_weights(y_train, config.N_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # weighted CE over softmax logits

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0

    train_loss_history, val_loss_history = [], []

    for epoch in range(max_epochs):
        # ---- train ----
        model.train()
        running_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running_loss / max(n_seen, 1)

        # ---- validate ----
        model.eval()
        running_val_loss, n_val_seen = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                running_val_loss += loss.item() * xb.size(0)
                n_val_seen += xb.size(0)
        val_loss = running_val_loss / max(n_val_seen, 1)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        scheduler.step(val_loss)

        if verbose:
            print(f"  epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(
            f"Training produced no valid val_loss (diverged?). "
            f"lr={learning_rate}, conv_config={conv_config}, fc_units={fc_units}, "
            f"use_pooling={use_pooling}"
        )
    model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
    )


def run_fixed_epoch_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    conv_config,
    fc_units,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    use_pooling: bool,
    device: torch.device,
    seed: int,
    verbose: bool = False,
) -> TrainResult:
    
    """
    Trains for exactly n_epochs on X_train/y_train with NO validation set
    and NO early stopping. Used for the final refit on the full
    outer-training set.
    """
    train_ds = ConnectivityDataset(X_train, y_train)
    gen = make_generator(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        generator=gen,
        worker_init_fn=seed_worker if config.NUM_WORKERS > 0 else None,
    )

    model = EEGConnectivityCNN(
        conv_config=conv_config,
        fc_units=fc_units,
        dropout=dropout,
        in_channels=config.IN_CHANNELS,
        n_classes=config.N_CLASSES,
        use_pooling=use_pooling,
        input_size=config.INPUT_SIZE,
    ).to(device)

    class_weights = compute_class_weights(y_train, config.N_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
    )

    train_loss_history = []
    for epoch in range(n_epochs):
        model.train()
        running_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running_loss / max(n_seen, 1)
        train_loss_history.append(train_loss)
        if verbose:
            print(f"  [final refit] epoch {epoch:03d}  train_loss={train_loss:.4f}")

    return TrainResult(
        model=model,
        best_epoch=n_epochs - 1,
        best_val_loss=float("nan"),  # no validation set used, by design
        train_loss_history=train_loss_history,
        val_loss_history=[],         # intentionally empty
    )


@torch.no_grad()
def predict_probabilities(model: EEGConnectivityCNN, X: np.ndarray, device, batch_size: int = 64):
    """Returns softmax probabilities, shape (N, n_classes)."""
    model.eval()
    ds = ConnectivityDataset(X, np.zeros(X.shape[0], dtype=np.int64))  # dummy y, unused
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_probs = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)