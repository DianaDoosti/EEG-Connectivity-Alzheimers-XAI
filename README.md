# EEG Connectivity + CNN for Alzheimer's Disease Classification, with Integrated Gradients

Project description : classifying **Alzheimer's disease (AD) vs. cognitively normal (CN)** participants from resting-state EEG, using band-resolved functional connectivity matrices as input to a small CNN, with **subject-wise nested cross-validation** and **Integrated Gradients (IG)** for interpretability.

## Overview

Key design choices:

- **Features:** pairwise-orthogonalized amplitude-envelope correlation (AEC-c), computed per 4-s epoch in five bands (delta, theta, alpha, beta, gamma). Each epoch becomes a `(5, 19, 19)` tensor; the five bands act as the CNN's input channels.
- **Evaluation unit:** the **subject**, not the epoch. All epochs of a subject stay in the same fold, and epoch-level probabilities are averaged into one prediction per subject.
- **Leakage control:** normalization statistics, class weights, hyperparameter selection, and early stopping use training data only. Outer-test subjects are touched once, at the end.

---

## Dataset

This project uses the publicly available resting-state EEG dataset:

Miltiadous, A., Tzimourta, K. D., Afrantou, T., Ioannidis, P., Grigoriadis, N., Tsalikakis, D. G., Angelidis, P., Tsipouras, M. G., Glavas, E., Giannakeas, N., & Tzallas, A. T. (2024). *A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects*. OpenNeuro. [Dataset]. [doi:10.18112/openneuro.ds004504.v1.0.8](https://doi.org/10.18112/openneuro.ds004504.v1.0.8)

- 19 scalp channels, EEGLAB `.set` files; the **preprocessed** version of the public dataset was used.
- Subjects: AD = 1–36, CN = 37–65, FTD = 66–88.
- Only **AD vs. CN** (65 subjects: 36 AD, 29 CN) is used for classification. FTD recordings are loaded during extraction but excluded from the binary task.

---

## How to run

Run from the repository root.

**1. Download the data.** Get the dataset from OpenNeuro and place it in `eeg_dataset/` (with the `AD/`, `CN/`, `FTD/` diagnostic folders expected by the notebook).

**2. Extract connectivity features.** Run `connectivity_extraction.ipynb`. It creates `eeg_connectivity_dataset/` and writes the final `.npz` file (shape `(N_epochs, 5, 19, 19)`) to `eeg_connectivity_dataset/concatenated/`.

**3. Point the config to your file.** In `scripts/config.py`, make sure the data filename matches the file in `eeg_connectivity_dataset/concatenated/`.

**4. (Optional) Smoke test.** A fast end-to-end check that the pipeline runs:
```bash
python scripts/smoke_test_main.py
```

**5. Train and evaluate (nested CV).**
```bash
python scripts/main.py
```
Outputs (models, fold indices, per-fold metrics, `outer_cv_summary.json`) are written to `output/`.

**6. Summarize results.**
```bash
python scripts/summarize_results.py
```
Produces `final_results_table.csv`, `final_results_table.md`, and `figures/balanced_accuracy_per_fold.png`.

**7. Interpretability.** Integrated Gradients analyses can be performed using the post-hoc interpretability notebook:
```bash
python scripts/post_hoc_integrated_gradients.ipynb
```
The notebook loads trained fold models and computes Integrated Gradients attribution maps for the input connectivity matrices.
