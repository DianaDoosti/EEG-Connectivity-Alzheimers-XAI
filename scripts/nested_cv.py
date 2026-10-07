"""
Nested StratifiedGroupKFold cross-validation.

Outer loop (5 folds): produces an unbiased subject-level performance
estimate, and a final saved model per fold.

Inner loop (5 folds), run inside each outer-training set: used only to
select the best hyperparameter configuration (via random search) for
that outer fold. Never touches the outer-test subjects.

Selection rule: for each sampled HP config, train across the 5 inner
folds, aggregate epoch-level predictions to subject-level per inner-val
fold, compute subject-level balanced accuracy per fold, then average
across the 5 inner folds. The config with the highest mean balanced
accuracy wins; ties are broken by lower mean validation loss.
"""

import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

import config
from data_utils import fit_band_normalization, apply_band_normalization
from evaluate import (
    aggregate_to_subject_level,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    plot_inner_cv_training_curves,
)
from model import build_model_from_config_dict
from reproducibility import seed_everything
from training import predict_probabilities, run_fixed_epoch_training, run_training


def get_device() -> torch.device:
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_random_configs(rng: np.random.RandomState, n_configs: int):
    all_combos = [
        {"conv_config": conv, "fc_units": fc, "dropout": drop, "learning_rate": lr, "batch_size": bs}
        for conv in config.CONV_CONFIGS
        for fc in config.FC_UNITS
        for drop in config.DROPOUT_VALUES
        for lr in config.LEARNING_RATES
        for bs in config.BATCH_SIZES
    ]
    n_configs = min(n_configs, len(all_combos))
    idx = rng.choice(len(all_combos), size=n_configs, replace=False)
    return [all_combos[i] for i in idx]


def run_inner_cv_for_config(
    X_outer_train, y_outer_train, subj_outer_train,
    hp_config, device, split_seed, init_seed_base,
):
    """
    Trains this HP config across N_INNER_FOLDS StratifiedGroupKFold splits
    of the outer-training subjects. Returns dict with mean/std balanced
    accuracy, mean val loss, and mean best_epoch (used later to decide the
    epoch budget for the final refit on the full outer-train set).

    split_seed: fixed for the whole outer fold, same for every HP config,
                so all configs are compared on identical inner-fold splits.
    init_seed_base: varies per HP config, controls weight init / dropout
                     masks / dataloader shuffling for this config's runs.
    """
    inner_cv = StratifiedGroupKFold(
        n_splits=config.N_INNER_FOLDS, shuffle=True, random_state=split_seed
    )

    bal_accs, val_losses, best_epochs = [], [], []
    training_histories = []

    for inner_fold_idx, (tr_idx, val_idx) in enumerate(
        inner_cv.split(X_outer_train, y_outer_train, groups=subj_outer_train)
    ):
        train_subjects = set(subj_outer_train[tr_idx])
        val_subjects = set(subj_outer_train[val_idx])
        assert train_subjects.isdisjoint(val_subjects), "Inner fold subject leakage detected"

        X_tr_raw, y_tr = X_outer_train[tr_idx], y_outer_train[tr_idx]
        X_val_raw, y_val = X_outer_train[val_idx], y_outer_train[val_idx]
        subj_val = subj_outer_train[val_idx]

        if config.NORMALIZE_PER_BAND:
            mean, std = fit_band_normalization(X_tr_raw)
            X_tr = apply_band_normalization(X_tr_raw, mean, std)
            X_val = apply_band_normalization(X_val_raw, mean, std)
        else:
            X_tr, X_val = X_tr_raw, X_val_raw

        run_seed = init_seed_base * 1000 + inner_fold_idx
        seed_everything(run_seed)

        result = run_training(
            X_tr, y_tr, X_val, y_val,
            conv_config=hp_config["conv_config"],
            fc_units=hp_config["fc_units"],
            dropout=hp_config["dropout"],
            learning_rate=hp_config["learning_rate"],
            batch_size=hp_config["batch_size"],
            use_pooling=config.USE_POOLING,
            device=device,
            seed=run_seed,
        )

        training_histories.append({
             "inner_fold": inner_fold_idx,
             "train_loss": result.train_loss_history,
             "val_loss": result.val_loss_history,
        })

        val_probs = predict_probabilities(result.model, X_val, device)
        subj_result = aggregate_to_subject_level(subj_val, y_val, val_probs)
        metrics = compute_metrics(subj_result)

        bal_accs.append(metrics["balanced_accuracy"])
        val_losses.append(result.best_val_loss)
        best_epochs.append(result.best_epoch)

    return {
        "hp_config": hp_config,
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "std_balanced_accuracy": float(np.std(bal_accs)),
        "mean_val_loss": float(np.mean(val_losses)),
        "mean_best_epoch": float(np.mean(best_epochs)),
        "per_fold_balanced_accuracy": [float(v) for v in bal_accs],
        "training_histories": training_histories,
    }


def select_best_config(search_results: list) -> dict:
    """Rank by mean balanced accuracy (desc), tie-break by mean val loss (asc)."""
    ranked = sorted(
        search_results,
        key=lambda r: (-r["mean_balanced_accuracy"], r["mean_val_loss"]),
    )
    return ranked[0]


def run_nested_cv(X, y, subject_ids):
    """
    Full nested CV: outer StratifiedGroupKFold with per-fold inner random
    search for HP selection, final refit on the full outer-train set, and
    evaluation on the held-out outer-test subjects. Saves everything to
    config.OUTPUT_DIR.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    models_dir = os.path.join(config.OUTPUT_DIR, "models")
    figures_dir = os.path.join(config.OUTPUT_DIR, "figures")
    folds_dir = os.path.join(config.OUTPUT_DIR, "fold_indices")
    for d in (models_dir, figures_dir, folds_dir):
        os.makedirs(d, exist_ok=True)

    device = get_device()
    print(f"[nested_cv] Using device: {device}")

    outer_cv = StratifiedGroupKFold(
        n_splits=config.N_OUTER_FOLDS, shuffle=True, random_state=config.GLOBAL_SEED
    )

    outer_fold_results = []

    summary_path = os.path.join(
        config.OUTPUT_DIR,
        "outer_cv_summary.json"
    )

    if config.RESUME and os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                outer_fold_results = json.load(f)

            print(
                f"[nested_cv] Resuming previous run: "
                f"loaded {len(outer_fold_results)} completed outer-fold results."
            )
        except (json.JSONDecodeError, OSError):
            print(
                "[nested_cv] Existing outer_cv_summary.json could not be read. "
                "Starting results summary from scratch."
            )
            outer_fold_results = []


    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y, groups=subject_ids)
    ):
        fold_start = time.time()

        # -------------------------------------------------------------
        # Resume check: skip outer folds that were completed previously
        # -------------------------------------------------------------
        completion_path = os.path.join(
            models_dir,
            f"outer_fold_{outer_fold_idx}_COMPLETE"
        )


        if config.RESUME and os.path.exists(completion_path):
            print(
                f"[nested_cv] Outer fold {outer_fold_idx + 1}/"
                f"{config.N_OUTER_FOLDS} already completed. Skipping."
            )
            continue

        print(f"\n{'=' * 70}\n[nested_cv] Outer fold {outer_fold_idx + 1}/{config.N_OUTER_FOLDS}\n{'=' * 70}")

        train_subjects = set(subject_ids[outer_train_idx])
        test_subjects = set(subject_ids[outer_test_idx])
        assert train_subjects.isdisjoint(test_subjects), "Outer fold subject leakage detected"

        # Save fold indices (subject IDs, not row indices, for readability
        # and independence from any future re-ordering of the array).
        fold_index_record = {
            "outer_fold": outer_fold_idx,
            "train_subjects": sorted(str(s) for s in train_subjects),
            "test_subjects": sorted(str(s) for s in test_subjects),
        }
        with open(os.path.join(folds_dir, f"outer_fold_{outer_fold_idx}.json"), "w") as f:
            json.dump(fold_index_record, f, indent=2)

        X_outer_train, y_outer_train = X[outer_train_idx], y[outer_train_idx]
        subj_outer_train = subject_ids[outer_train_idx]
        X_outer_test, y_outer_test = X[outer_test_idx], y[outer_test_idx]
        subj_outer_test = subject_ids[outer_test_idx]

        # ---- Inner random search for this outer fold ----
        rng = np.random.RandomState(config.GLOBAL_SEED + outer_fold_idx)
        candidate_configs = sample_random_configs(rng, config.N_RANDOM_CONFIGS)

        # Fixed once per outer fold -- every HP config this outer fold is
        # evaluated on the exact same 5 inner-CV splits.
        inner_split_seed = config.GLOBAL_SEED + outer_fold_idx

        search_results = []
        for cfg_idx, hp_config in enumerate(candidate_configs):
            print(
                f"  [inner search] config {cfg_idx + 1}/{len(candidate_configs)}: "
                f"conv={hp_config['conv_config']} dropout={hp_config['dropout']} "
                f"lr={hp_config['learning_rate']} batch={hp_config['batch_size']}"
            )
            res = run_inner_cv_for_config(
                X_outer_train, y_outer_train, subj_outer_train,
                hp_config, device,
                split_seed=inner_split_seed,
                init_seed_base=config.GLOBAL_SEED + outer_fold_idx * 100 + cfg_idx,
            )
            print(
                f"    -> mean inner balanced accuracy = "
                f"{res['mean_balanced_accuracy']:.3f} +/- {res['std_balanced_accuracy']:.3f}, "
                f"mean val loss = {res['mean_val_loss']:.4f}"
            )
            search_results.append(res)

        best = select_best_config(search_results)
        print(f"  [inner search] BEST config: {best['hp_config']} "
              f"(mean balanced accuracy = {best['mean_balanced_accuracy']:.3f})")

        plot_inner_cv_training_curves(
            best["training_histories"],
            title=(
                f"Outer fold {outer_fold_idx + 1} - "
                f"best HP configuration: inner-CV training/validation loss"
            ),
            save_path=os.path.join(
                figures_dir,
                f"outer_fold_{outer_fold_idx}_best_inner_cv_loss.png"
            ),
        )

        # Save the full search log for this outer fold (for the reproducibility
        # writeup / for the person running this to inspect).
        search_results_for_json = []

        for result in search_results:
            result_copy = result.copy()
            result_copy.pop("training_histories", None)
            search_results_for_json.append(result_copy)

        with open(
            os.path.join(
                config.OUTPUT_DIR,
                f"hp_search_outer_fold_{outer_fold_idx}.json"
            ),
            "w"
        ) as f:
            json.dump(search_results_for_json, f, indent=2)

        # ---- Final refit on the FULL outer-training set ----
        # No further held-out validation split is available at this point
        # (all outer-train subjects are used for the final fit), so the
        # epoch budget is fixed at the average best_epoch found during the
        # inner search, rather than using early stopping again.
        final_epochs = max(1, round(best["mean_best_epoch"]) + 1)
        print(f"  [final refit] Training on all {len(train_subjects)} outer-train subjects "
              f"for {final_epochs} epochs (fixed, from inner-CV average best epoch).")

        if config.NORMALIZE_PER_BAND:
            final_mean, final_std = fit_band_normalization(X_outer_train)
            X_outer_train_norm = apply_band_normalization(X_outer_train, final_mean, final_std)
            X_outer_test_norm = apply_band_normalization(X_outer_test, final_mean, final_std)
        else:
            final_mean, final_std = None, None
            X_outer_train_norm, X_outer_test_norm = X_outer_train, X_outer_test

        final_seed = config.GLOBAL_SEED + outer_fold_idx * 1000 + 999
        seed_everything(final_seed)

        # Fixed-epoch training with NO validation set: the epoch budget was
        # already decided by the inner-CV search above, so no further
        # val-loss-based checkpoint selection happens here. This is the
        # leakage-safe way to do the final refit -- outer-test is not
        # touched until the evaluation step immediately below.
        final_result = run_fixed_epoch_training(
            X_outer_train_norm, y_outer_train,
            conv_config=best["hp_config"]["conv_config"],
            fc_units=best["hp_config"]["fc_units"],
            dropout=best["hp_config"]["dropout"],
            learning_rate=best["hp_config"]["learning_rate"],
            batch_size=best["hp_config"]["batch_size"],
            use_pooling=config.USE_POOLING,
            n_epochs=final_epochs,
            device=device,
            seed=final_seed,
        )

        # ---- Evaluate on outer-test (truly held out) ----
        test_probs = predict_probabilities(final_result.model, X_outer_test_norm, device)
        subj_result = aggregate_to_subject_level(subj_outer_test, y_outer_test, test_probs)
        metrics = compute_metrics(subj_result)

        # Save per-subject predictions for this fold (needed for error
        # analysis and for combining folds into one pooled ROC later).
        pd.DataFrame({
            "subject_id": subj_result.subject_ids,
            "y_true": subj_result.y_true,
            "y_prob": subj_result.y_prob,
            "y_pred": subj_result.y_pred,
        }).to_csv(
            os.path.join(config.OUTPUT_DIR, f"predictions_outer_fold_{outer_fold_idx}.csv"),
            index=False,
        )

        print(f"  [outer test] balanced_accuracy={metrics['balanced_accuracy']:.3f} "
              f"sensitivity={metrics['sensitivity']:.3f} specificity={metrics['specificity']:.3f} "
              f"f1={metrics['f1']:.3f} auc={metrics['auc']:.3f}")

        # ---- Save everything for this fold ----
        checkpoint = {
            "model_state_dict": final_result.model.state_dict(),
            "model_config": final_result.model.get_config_dict(),
            "hp_config": best["hp_config"],
            "normalization_mean": final_mean,
            "normalization_std": final_std,
            "final_epochs_trained": final_epochs,
            "outer_fold": outer_fold_idx,
            "class_to_int": config.CLASS_TO_INT,
        }
        torch.save(checkpoint, os.path.join(models_dir, f"outer_fold_{outer_fold_idx}_model.pt"))

        plot_training_curves(
            final_result.train_loss_history, val_loss=None,
            title=f"Outer fold {outer_fold_idx + 1} final refit (train loss only -- "
                  f"no validation set used here by design, see README)",
            save_path=os.path.join(figures_dir, f"outer_fold_{outer_fold_idx}_training_curve.png"),
        )
        plot_confusion_matrix(
            subj_result, config.CLASS_NAMES,
            title=f"Outer fold {outer_fold_idx + 1} confusion matrix (subject-level)",
            save_path=os.path.join(figures_dir, f"outer_fold_{outer_fold_idx}_confusion_matrix.png"),
        )
        plot_roc_curve(
            subj_result,
            title=f"Outer fold {outer_fold_idx + 1} ROC (subject-level)",
            save_path=os.path.join(figures_dir, f"outer_fold_{outer_fold_idx}_roc_curve.png"),
        )

        outer_fold_results.append({
            "outer_fold": outer_fold_idx,
            "best_hp_config": best["hp_config"],
            "metrics": metrics,
            "n_train_subjects": len(train_subjects),
            "n_test_subjects": len(test_subjects),
            "elapsed_seconds": time.time() - fold_start,
        })

        # Mark this outer fold as completely finished.
        # This file is created only after the model, figures, and results
        # have all been successfully written.
        with open(completion_path, "w") as f:
            f.write("complete\n")

        # Save progress after every completed outer fold so that
        # results from completed folds are not lost if the run is interrupted.
        with open(
            os.path.join(config.OUTPUT_DIR, "outer_cv_summary.json"), "w"
        ) as f:
            json.dump(outer_fold_results, f, indent=2)

    with open(os.path.join(config.OUTPUT_DIR, "outer_cv_summary.json"), "w") as f:
        json.dump(outer_fold_results, f, indent=2)

    return outer_fold_results
