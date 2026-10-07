"""
Smoke-test entry point.

Runs the complete pipeline using the reduced smoke-test configuration
defined in smoke_out_config.py.

This is intended only to verify that the entire pipeline executes
correctly from start to finish.

"""

import json
import os
import sys
import time

# ---------------------------------------------------------------------
# Use smoke-test configuration instead of the real config.py
# ---------------------------------------------------------------------

import smoke_test_config as config

sys.modules["config"] = config


from data_utils import filter_to_binary_task, load_npz
from nested_cv import run_nested_cv
from reproducibility import seed_everything
from summarize_results import summarize


def main():

    print("=" * 70)
    print("SMOKE TEST")
    print("=" * 70)

    print(f"[smoke] Loading data from: {config.DATA_PATH}")
    print(f"[smoke] Output directory: {config.OUTPUT_DIR}")

    # -------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------

    seed_everything(config.GLOBAL_SEED)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    raw = load_npz(config.DATA_PATH)

    print(
        f"[smoke] Loaded X shape={raw.X.shape}, "
        f"{len(set(raw.subject_ids.tolist()))} subjects total."
    )

    # -------------------------------------------------------------
    # Filter to AD vs CN
    # -------------------------------------------------------------

    binary_data = filter_to_binary_task(raw)

    print(
        f"[smoke] Binary dataset: "
        f"{binary_data.X.shape[0]} epochs, "
        f"{len(set(binary_data.subject_ids.tolist()))} subjects."
    )

    # -------------------------------------------------------------
    # Save exact configuration used for this smoke test
    # -------------------------------------------------------------

    run_config_record = {
        "smoke_test": True,

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
        "lr_scheduler_patience": config.LR_SCHEDULER_PATIENCE,
        "lr_scheduler_factor": config.LR_SCHEDULER_FACTOR,

        "global_seed": config.GLOBAL_SEED,
        "normalize_per_band": config.NORMALIZE_PER_BAND,

        "n_epochs_in_dataset": int(binary_data.X.shape[0]),
        "n_subjects_in_dataset": int(
            len(set(binary_data.subject_ids.tolist()))
        ),

        "original_npz_metadata": binary_data.metadata,
    }

    config_path = os.path.join(
        config.OUTPUT_DIR,
        "smoke_test_config_used.json"
    )

    with open(config_path, "w") as f:
        json.dump(
            run_config_record,
            f,
            indent=2,
            default=str,
        )

    print(f"[smoke] Saved configuration to: {config_path}")

    # -------------------------------------------------------------
    # Run nested CV
    # -------------------------------------------------------------

    start = time.time()

    run_nested_cv(
        binary_data.X,
        binary_data.y_bin,
        binary_data.subject_ids,
    )

    elapsed = time.time() - start

    print(
        f"\n[smoke] Nested CV finished in "
        f"{elapsed / 60:.2f} minutes."
    )

    # -------------------------------------------------------------
    # Generate final summary
    # -------------------------------------------------------------

    summarize()

    # -------------------------------------------------------------
    # Finished
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(
        f"[smoke] All outputs written to: "
        f"{config.OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()