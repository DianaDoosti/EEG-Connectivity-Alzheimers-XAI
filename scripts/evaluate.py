"""
Epoch-level predictions -> subject-level predictions -> metrics.
Also holds plotting functions for training curves, confusion matrices,
and ROC curves.
"""

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class SubjectLevelResult:
    subject_ids: np.ndarray
    y_true: np.ndarray          # (n_subjects,) 0/1
    y_prob: np.ndarray          # (n_subjects,) probability of positive class
    y_pred: np.ndarray          # (n_subjects,) 0/1


def aggregate_to_subject_level(
    subject_ids: np.ndarray, y_true_epoch: np.ndarray, y_prob_epoch: np.ndarray
) -> SubjectLevelResult:
    """
    Average epoch-level predicted probabilities within each subject, then
    threshold at 0.5 to get one prediction per subject.
    """
    unique_subjects = np.unique(subject_ids)
    y_true_subj = np.empty(len(unique_subjects), dtype=np.int64)
    y_prob_subj = np.empty(len(unique_subjects), dtype=np.float64)

    for i, subj in enumerate(unique_subjects):
        mask = subject_ids == subj
        subj_true = np.unique(y_true_epoch[mask])
        assert len(subj_true) == 1, f"Subject {subj} has mixed epoch labels"
        y_true_subj[i] = subj_true[0]
        # y_prob_epoch assumed shape (N, 2); take probability of class 1 (positive)
        y_prob_subj[i] = y_prob_epoch[mask, 1].mean()

    y_pred_subj = (y_prob_subj >= 0.5).astype(np.int64)

    return SubjectLevelResult(
        subject_ids=unique_subjects,
        y_true=y_true_subj,
        y_prob=y_prob_subj,
        y_pred=y_pred_subj,
    )


def compute_metrics(result: SubjectLevelResult) -> dict:
    """Balanced accuracy, sensitivity, specificity, F1, AUC, accuracy."""
    y_true, y_pred, y_prob = result.y_true, result.y_pred, result.y_prob

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan   # recall of positive class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan  # happens if a fold has only one class present

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "n_subjects": len(y_true),
    }


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_training_curves(train_loss, val_loss, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_loss, label="Train loss")
    if val_loss:
        ax.plot(val_loss, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted cross-entropy loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_inner_cv_training_curves(histories, title, save_path):
    """
    Plot training and validation loss for each inner-CV fold separately,
    plus a sixth panel showing all five folds overlaid.

    histories is a list of dictionaries containing:
        {
            "inner_fold": int,
            "train_loss": [...],
            "val_loss": [...]
        }
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    axes = axes.ravel()

    # -------------------------------------------------------------
    # Panels 1-5: individual inner folds
    # -------------------------------------------------------------
    for i, history in enumerate(histories):
        fold = history["inner_fold"] + 1
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]

        ax = axes[i]

        ax.plot(
            train_loss,
            label="Train loss",
            linewidth=2
        )

        ax.plot(
            val_loss,
            linestyle="--",
            label="Validation loss",
            linewidth=2
        )

        ax.set_title(f"Inner fold {fold}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted cross-entropy loss")
        ax.grid(alpha=0.25)
        ax.legend()

    # -------------------------------------------------------------
    # Panel 6: all five folds overlaid
    # -------------------------------------------------------------
    ax = axes[5]

    for history in histories:
        fold = history["inner_fold"] + 1

        ax.plot(
            history["train_loss"],
            label=f"Fold {fold} train"
        )

        ax.plot(
            history["val_loss"],
            linestyle="--",
            label=f"Fold {fold} validation"
        )

    ax.set_title("All inner folds")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(result: SubjectLevelResult, class_names, title, save_path):
    cm = confusion_matrix(result.y_true, result.y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(result: SubjectLevelResult, title, save_path):
    fpr, tpr, _ = roc_curve(result.y_true, result.y_prob)
    try:
        auc = roc_auc_score(result.y_true, result.y_prob)
    except ValueError:
        auc = float("nan")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
