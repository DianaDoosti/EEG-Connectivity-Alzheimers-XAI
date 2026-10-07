"""
Reads outer_cv_summary.json (written by nested_cv.run_nested_cv) and
produces a clean, summary table (CSV + Markdown) plus
a bar plot of subject-level balanced accuracy across the 5 outer folds.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def summarize(output_dir: str = None):
    output_dir = output_dir or config.OUTPUT_DIR
    summary_path = os.path.join(output_dir, "outer_cv_summary.json")
    with open(summary_path) as f:
        results = json.load(f)

    rows = []
    for r in results:
        row = {"outer_fold": r["outer_fold"] + 1}
        row.update(r["metrics"])
        row["n_train_subjects"] = r["n_train_subjects"]
        row["n_test_subjects"] = r["n_test_subjects"]
        row["best_conv_config"] = str(r["best_hp_config"]["conv_config"])
        row["best_fully_connected_config"] = r["best_hp_config"]["fc_units"]    ###########
        row["best_dropout"] = r["best_hp_config"]["dropout"]
        row["best_learning_rate"] = r["best_hp_config"]["learning_rate"]
        row["best_batch_size"] = r["best_hp_config"]["batch_size"]
        rows.append(row)

    df = pd.DataFrame(rows)

    metric_cols = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"]
    mean_row = {"outer_fold": "Mean"}
    std_row = {"outer_fold": "Std"}
    for col in metric_cols:
        mean_row[col] = df[col].mean()
        std_row[col] = df[col].std()
    summary_df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    csv_path = os.path.join(output_dir, "final_results_table.csv")
    md_path = os.path.join(output_dir, "final_results_table.md")
    summary_df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(f"# AD vs CN -- 5-fold subject-wise nested CV results\n\n")
        f.write(summary_df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\nBinary chance-level balanced accuracy = 0.500\n")

    print(f"[summarize] Wrote {csv_path}")
    print(f"[summarize] Wrote {md_path}")
    print(summary_df.to_string(index=False))

    # Bar plot: balanced accuracy per fold + mean line
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fold_vals = df["balanced_accuracy"].values
    ax.bar(df["outer_fold"].astype(str), fold_vals, color="#4C72B0")
    ax.axhline(fold_vals.mean(), color="black", linestyle="--",
               label=f"Mean = {fold_vals.mean():.3f}")
    ax.axhline(0.5, color="red", linestyle=":", label="Chance = 0.500")
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Subject-level balanced accuracy")
    ax.set_title("AD vs CN -- balanced accuracy per outer fold")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "balanced_accuracy_per_fold.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[summarize] Wrote {fig_path}")

    return summary_df


if __name__ == "__main__":
    summarize()
