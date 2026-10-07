"""
Entry point. Run this file to execute the full pipeline:

  1. Load the .npz archive.
  2. Filter to the binary AD-vs-CN task (drops the FTD group).
  3. Run nested StratifiedGroupKFold CV (5 outer x 5 inner), with random
     search over hyperparameters in each outer fold's inner loop.
  4. Save a final model checkpoint per outer fold, all fold indices,
     the full HP search log, training curves, confusion matrices, ROC
     curves.
  5. Aggregate everything into a final results table and summary plot.

"""

import json
import os
import time

import config
from data_utils import filter_to_binary_task, load_npz
from nested_cv import run_nested_cv
from reproducibility import seed_everything
from summarize_results import summarize


def main():
    seed_everything(config.GLOBAL_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"[main] Loading data from: {config.DATA_PATH}")
    raw = load_npz(config.DATA_PATH)
    print(f"[main] Loaded X shape={raw.X.shape}, "
          f"{len(set(raw.subject_ids.tolist()))} subjects total.")

    binary_data = filter_to_binary_task(raw)

    # Save a copy of the exact config used for this run, for reproducibility.
    run_config_record = {
        "positive_class": config.POSITIVE_CLASS_LABEL,
        "negative_class": config.NEGATIVE_CLASS_LABEL,
        "n_outer_folds": config.N_OUTER_FOLDS,
        "n_inner_folds": config.N_INNER_FOLDS,
        "n_random_configs": config.N_RANDOM_CONFIGS,
        "conv_configs_search_space": config.CONV_CONFIGS,
        "dropout_values": config.DROPOUT_VALUES,
        "learning_rates": config.LEARNING_RATES,
        "batch_sizes": config.BATCH_SIZES,
        "fixed_fc_units": config.FC_UNITS,
        "fixed_optimizer": config.OPTIMIZER,
        "fixed_weight_decay": config.WEIGHT_DECAY,
        "max_epochs": config.MAX_EPOCHS,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "global_seed": config.GLOBAL_SEED,
        "normalize_per_band": config.NORMALIZE_PER_BAND,
        "n_epochs_in_dataset": int(binary_data.X.shape[0]),
        "n_subjects_in_dataset": int(len(set(binary_data.subject_ids.tolist()))),
        "original_npz_metadata": binary_data.metadata,
    }
    with open(os.path.join(config.OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(run_config_record, f, indent=2, default=str)

    start = time.time()
    run_nested_cv(binary_data.X, binary_data.y_bin, binary_data.subject_ids)
    elapsed = time.time() - start
    print(f"\n[main] Nested CV finished in {elapsed / 60:.1f} minutes.")

    summarize()
    print(f"\n[main] Done. All outputs written to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
