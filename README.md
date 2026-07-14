# Student Performance Prediction — Multimodal ML Pipeline

A research-grade, multi-dataset machine learning pipeline for predicting student
academic outcomes. Covers OULAD LMS data, xAPI classroom engagement, UCI datasets,
and a unified multi-source platform with SHAP explainability.

---

## Results Summary

| Dataset | Mode | Best Model | Accuracy | F1-Macro | ROC-AUC |
|---|---|---|---|---|---|
| **OULAD** (32,593 students) | **Binary** | LightGBM | **95.05%** | **95.04%** | **0.988** |
| OULAD | 4-class | XGBoost | 75.95% | 70.46% | 0.930 |
| xAPI | Binary | CatBoost | 97.92% | 97.22% | 0.989 |
| xAPI | 4-class | XGBoost | 77.08% | 77.92% | 0.899 |
| UCI Dropout | Binary | CatBoost | 82.94% | 82.92% | 0.905 |
| UCI Student Perf | Binary | XGBoost | 81.34% | 67.64% | 0.753 |

### Extended model comparison (OULAD, all 8 models)

| Model | Binary Acc | Binary F1 | 4-class Acc | 4-class F1 |
|---|---|---|---|---|
| LightGBM | **95.05%** | **95.04%** | 75.32% | 71.06% |
| XGBoost | 95.03% | 95.02% | **75.95%** | **70.46%** |
| CatBoost | 94.97% | 94.96% | 72.85% | 70.22% |
| MLP | 94.75% | 94.75% | 74.08% | 68.23% |
| Bagged DT | 94.72% | 94.72% | 73.52% | 70.76% |
| Extra Trees | 94.55% | 94.55% | 74.87% | 69.59% |
| Random Forest | 94.28% | 94.28% | 74.37% | 67.77% |
| Decision Tree | 92.82% | 92.81% | 69.67% | 66.92% |

---

## Repository Structure

```
.
├── PIPELINE SCRIPTS
│   ├── oulad_pipeline.py              # Step 1a: build oulad_ml_table.csv from raw OULAD CSVs
│   ├── oulad_pipeline_v2.py           # Step 1b: richer temporal features → oulad_ml_table_v2.csv
│   ├── oulad_baseline.py              # Step 2:  baseline XGB/LGB/CAT/RF on the ML table
│   ├── high_accuracy_pipeline.py      # Step 3:  high-accuracy 4-model training (XGB/LGB/CAT/RF)
│   ├── extended_models_pipeline.py    # Step 4:  extended models (ET/BDT/DT/MLP/DNN)
│   ├── generate_missing_shap.py       # Step 5:  fill any missing SHAP files
│   ├── update_main_results.py         # Step 6:  regenerate all 10 comparison figures
│   ├── hierarchical_pipeline.py       # Optional: cascaded binary→4-class architecture
│   ├── multisource_ablation.py        # Optional: unified OULAD+xAPI+UCI platform
│   ├── synthetic_platform.py          # Optional: single CatBoost multi-source platform
│   └── generate_all_figures.py        # Optional: full 35-figure paper figure generator
│
├── UTILITIES
│   ├── download_datasets.py           # Auto-download UCI datasets
│   ├── dataset_analysis.py            # General CSV audit and feature quality report
│   ├── oulad_dataset_forensics.py     # Deep OULAD data quality audit
│   ├── check_fixes.py                 # Verify all research integrity fixes are in place
│   ├── build_colab_notebook.py        # Regenerate the Colab notebook from source
│   └── dir_str_with_file.py           # Print workspace file tree
│
├── NOTEBOOKS
│   └── OULAD_All_Models_Colab.ipynb   # Google Colab: all 9 models end-to-end (GPU support)
│
├── DATA (not committed — see Step 0)
│   ├── oulad_ml_table.csv             # Built by oulad_pipeline.py
│   ├── oulad_ml_table_v2.csv          # Built by oulad_pipeline_v2.py
│   ├── oulad_ml_table_week8.csv       # Early-warning table (Week 8 cutoff)
│   ├── academics/data.csv
│   ├── dropout/data.csv
│   ├── xAPI/xAPI-Edu-Data.csv
│   └── student+performance/student/
│
├── RESULTS
│   ├── results/high_accuracy/         # XGB/LGB/CAT/RF predictions + SHAP
│   ├── results/extended/              # ET/BDT/DT/MLP predictions + SHAP
│   ├── results/baseline_benchmark/    # oulad_baseline full-course outputs
│   ├── results/baseline_week8/        # oulad_baseline Week-8 outputs
│   ├── results/hierarchical/          # hierarchical_pipeline outputs
│   ├── results/synthetic_platform/    # synthetic_platform outputs
│   └── results/all_models_comparison.csv
│
└── FIGURES
    └── figures/main_results/          # B01–B05, M01–M05 publication figures
```

---

## Prerequisites

### Python version
Python 3.10 or later is required.

### Install dependencies
```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost shap optuna \
            matplotlib seaborn joblib torch torchvision
```

Or if a `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

---

## Complete Run Order

Follow these steps in sequence. Each step depends on the outputs of the previous one.

---

### Step 0 — Download raw datasets

```bash
python download_datasets.py
```

Downloads UCI datasets (Dropout, Student Performance, etc.) automatically.
For OULAD, download the raw CSVs manually from https://analyse.kmi.open.ac.uk/open_dataset
and place the seven files (`studentInfo.csv`, `studentVle.csv`, `studentAssessment.csv`,
`assessments.csv`, `vle.csv`, `studentRegistration.csv`, `courses.csv`) in the project root.

**Expected time:** 1–5 minutes (network dependent)

---

### Step 1 — Build the OULAD ML tables

Two versions are available. Run both to support all downstream scripts.

**V1 — standard features** (used by `oulad_baseline.py`, `high_accuracy_pipeline.py`):
```bash
python oulad_pipeline.py --root . --output oulad_ml_table.csv --chunksize 200000
```

**V2 — rich temporal features** (used by `extended_models_pipeline.py`, `multisource_ablation.py`):
```bash
python oulad_pipeline_v2.py --root . --output oulad_ml_table_v2.csv
```

**What it does:**  
Joins `studentInfo`, `studentVle`, `studentAssessment`, `assessments`, and
`studentRegistration` into one row-per-student-course ML table.

**Outputs:**
- `oulad_ml_table.csv` — ~32,593 rows × ~50 features
- `oulad_ml_table_v2.csv` — same rows, ~67 features including day-level temporal signals

**Expected time:** 3–10 minutes each (depends on disk speed)

---

### Step 2 — Run the baseline (optional but useful for reference)

```bash
python oulad_baseline.py \
    --input oulad_ml_table.csv \
    --target final_result \
    --model all \
    --mode benchmark \
    --shap-sample 300 \
    --output-dir results/baseline_benchmark
```

For the early-warning (Week-8) variant:
```bash
python oulad_baseline.py \
    --input oulad_ml_table_week8.csv \
    --target final_result \
    --model all \
    --mode early-warning \
    --shap-sample 300 \
    --output-dir results/baseline_week8
```

**Outputs:** `results/baseline_benchmark/` — feature importance CSVs, SHAP CSVs, metrics

**Expected time:** 5–15 minutes

---

### Step 3 — High-accuracy pipeline (XGBoost / LightGBM / CatBoost / Random Forest)

This is the primary training step. Run binary and 4-class together:

```bash
python high_accuracy_pipeline.py \
    --dataset oulad \
    --model xgboost,lightgbm,catboost,random_forest \
    --both \
    --cv-folds 5 \
    --n-jobs 1 \
    --output-dir results/high_accuracy
```

**Key flags:**
| Flag | Effect |
|---|---|
| `--binary` | Binary only (AtRisk vs Success) |
| `--both` | Binary + 4-class in one run |
| `--dataset all` | Run all datasets (oulad, dropout, xapi, uci_perf, …) |
| `--tune` | Optuna hyperparameter tuning — adds ~10 min per model |
| `--stacking` | Add a stacking ensemble on top |
| `--no-shap` | Skip SHAP computation (faster) |
| `--output-dir` | Where predictions, models, and SHAP CSVs are saved |

**Outputs** → `results/high_accuracy/`:
```
high_accuracy_results.csv
predictions_oulad_binary_{model}.csv     (4 files)
predictions_oulad_4class_{model}.csv     (4 files)
shap_oulad_binary_{model}.csv            (where SHAP ran)
shap_oulad_4class_{model}.csv            (where SHAP ran)
model_oulad_{mode}_{model}.pkl           (saved models)
encoder_oulad_{mode}_{model}.pkl         (label encoders)
```

**Expected time:** 20–40 minutes (all 4 models × 2 modes, no tuning)

> **Note:** If you only want a quick test, run one model:
> ```bash
> python high_accuracy_pipeline.py --dataset oulad --model lightgbm --both --output-dir results/high_accuracy
> ```

---

### Step 4 — Extended models (Extra Trees / Bagged DT / Decision Tree / MLP / DNN)

```bash
python extended_models_pipeline.py \
    --models et,bdt,dt,mlp \
    --mode both \
    --cv-folds 5 \
    --output-dir results/extended
```

To also include the PyTorch DNN (requires GPU or extra patience on CPU):
```bash
python extended_models_pipeline.py \
    --models et,bdt,dt,mlp,dnn \
    --mode both \
    --cv-folds 5 \
    --output-dir results/extended
```

To skip SHAP here (faster — SHAP is handled in Step 5):
```bash
python extended_models_pipeline.py --models et,bdt,dt,mlp --mode both --no-shap
```

**Outputs** → `results/extended/`:
```
extended_results.csv
predictions_oulad_binary_{model}.csv     (per model)
predictions_oulad_4class_{model}.csv     (per model)
shap_oulad_binary_{model}.csv            (if --no-shap not set)
shap_oulad_4class_{model}.csv            (if --no-shap not set)
```

**Expected time:** 15–30 minutes (4 sklearn models × 2 modes, no DNN)

---

### Step 5 — Generate missing SHAP files

If any SHAP files are missing (e.g., because `--no-shap` was used in Steps 3/4,
or because the ET 4-class run timed out), this script fills them all:

```bash
python generate_missing_shap.py
```

**What it does:**
- Scans all three SHAP search directories (`results/`, `results/high_accuracy/`, `results/extended/`)
- For tree models (RF, ET, BDT, DT): uses SHAP `TreeExplainer` — fast and exact
- For MLP/DNN: uses sklearn `permutation_importance` as a proxy — same CSV schema
- Saves any missing files to `results/extended/`

**Expected time:** 5–20 minutes (depends on which models are missing)

---

### Step 6 — Regenerate all comparison figures

```bash
python update_main_results.py
```

This merges `results/high_accuracy/high_accuracy_results.csv` with
`results/extended/extended_results.csv` and regenerates all 10 publication figures.

**Outputs** → `figures/main_results/`:

| File | Contents |
|---|---|
| `B01_binary_model_performance.png` | Bar chart: all models, 4 metrics, binary task |
| `B02_binary_confusion_matrix_comparison.png` | Confusion matrix grid, all models |
| `B03_binary_roc_comparison.png` | ROC curves with AUC, all models |
| `B04_binary_precision_recall_comparison.png` | PR curves with AUC, all models |
| `B05_binary_shap_importance_comparison.png` | SHAP bar charts, all models |
| `M01_four_class_model_performance.png` | Bar chart: all models, 4-class task |
| `M02_four_class_confusion_matrix_comparison.png` | Confusion matrices, 4-class |
| `M03_four_class_macro_roc_comparison.png` | Macro ROC curves, 4-class |
| `M04_four_class_classwise_roc.png` | Per-class OvR ROC for best model |
| `M05_four_class_shap_importance_comparison.png` | SHAP bar charts, 4-class |

Also saves `results/all_models_comparison.csv` — a merged summary table.

**Expected time:** < 1 minute

---

## Quick-run Commands (copy-paste in order)

```bash
# 0. Download datasets
python download_datasets.py

# 1. Build OULAD ML tables
python oulad_pipeline.py    --root . --output oulad_ml_table.csv --chunksize 200000
python oulad_pipeline_v2.py --root . --output oulad_ml_table_v2.csv

# 2. Baseline (optional reference)
python oulad_baseline.py --input oulad_ml_table.csv --target final_result --model all --mode benchmark --shap-sample 300 --output-dir results/baseline_benchmark

# 3. High-accuracy: XGBoost / LightGBM / CatBoost / Random Forest
python high_accuracy_pipeline.py --dataset oulad --model xgboost,lightgbm,catboost,random_forest --both --cv-folds 5 --n-jobs 1 --output-dir results/high_accuracy

# 4. Extended models: ET / Bagged DT / DT / MLP
python extended_models_pipeline.py --models et,bdt,dt,mlp --mode both --cv-folds 5 --output-dir results/extended

# 5. Fill any missing SHAP files
python generate_missing_shap.py

# 6. Regenerate all 10 figures
python update_main_results.py
```

---

## Optional Pipelines

These run independently and produce their own outputs. They do **not** need to
run before Step 6.

### Hierarchical pipeline (cascaded binary → sub-class)

```bash
python hierarchical_pipeline.py --skip-ctgan --output-dir results/hierarchical --save-graphs
```

Implements a 3-model cascade:
1. Model 1: AtRisk vs Success (~95% accuracy)
2. Model 2A: Fail vs Withdrawn (AtRisk branch)
3. Model 2B: Pass vs Distinction (Success branch)

**Outputs** → `results/hierarchical/`

---

### Multi-source ablation study (OULAD + xAPI + UCI)

```bash
# Full research run — benchmark mode (uses all features including late ones)
python multisource_ablation.py --mode benchmark --experiment all --cv-folds 5 --shap-sample 300 --report-students 3

# Early-warning mode (drops end-of-course features)
python multisource_ablation.py --mode early-warning --experiment all --cv-folds 5 --shap-sample 300 --report-students 3
```

**Experiment options:**
| Value | What runs |
|---|---|
| `main` | Unified model on all sources |
| `source-ablation` | WITH vs WITHOUT the source-identity feature |
| `dataset-ablation` | All 7 subsets: OULAD only, xAPI only, OULAD+xAPI, etc. |
| `all` | All three above |

**Outputs** → `results/multi_source_results.csv`, per-student intervention reports

---

### Unified CatBoost platform

```bash
python synthetic_platform.py --mode benchmark --shap-sample 300 --report-students 3
python synthetic_platform.py --mode early-warning --shap-sample 300 --report-students 3
```

Single CatBoost model trained on the merged OULAD + xAPI + UCI feature store.
Produces risk scores (High / Medium / Low) with actionable intervention text.

**Outputs** → `results/synthetic_platform/`

---

### Full 35-figure paper generator

```bash
# Run all experiments + generate all figures
python generate_all_figures.py

# Use cached experiment output only (no rerun)
python generate_all_figures.py --graphs-only

# Force rerun of all experiments
python generate_all_figures.py --force
```

Orchestrates all the above pipelines in sequence and generates the complete
figure set used in the research paper.

> **Warning:** Full run takes 2–4 hours. Use `--graphs-only` if experiment
> outputs already exist in `results/`.

---

## Research Integrity Verification

```bash
python check_fixes.py
```

Verifies that all 16 research integrity fixes are present in the codebase:

| Fix | Description |
|---|---|
| Fix 1/10 | Train-only median imputation (no test leakage) |
| Fix 2 | Temporal leakage audit — late features documented |
| Fix 3 | Single train/test split shared by all models in an experiment |
| Fix 4 | Small-dataset audit for 100% accuracy |
| Fix 6 | Target harmonisation warning for Medium class |
| Fix 7 | Pre-computed split shared across ablation experiments |
| Fix 8/11 | Source identifier excluded from final unified model |
| Fix 9 | Binary ROC-AUC uses correct single-column branch |
| Fix 12 | Per-source evaluation after unified model training |
| Fix 13/14 | Signed SHAP direction (toward High risk) |
| Fix 15 | Actionable vs model-driver feature separation |
| Fix 16 | Student overlap check / GroupShuffleSplit |

---

## Google Colab Notebook

`OULAD_All_Models_Colab.ipynb` — runs the full 9-model comparison in Colab
with GPU acceleration. Upload `oulad_ml_table.csv` when prompted.

**To regenerate the notebook from source:**
```bash
python build_colab_notebook.py
```

**Models included in the notebook:**
XGBoost · LightGBM · CatBoost · Random Forest · Extra Trees · Bagged DT ·
Decision Tree · MLP · DNN (PyTorch, GPU/CPU)

---

## Dataset Notes

| Dataset | Records | ML table required | Notes |
|---|---|---|---|
| OULAD | 32,593 | Yes — run Steps 1a/1b | 7 raw CSVs → 1 ML table |
| UCI Dropout | 4,424 | No | `dropout/data.csv` used directly |
| xAPI | 480 | No | `xAPI/xAPI-Edu-Data.csv` used directly |
| UCI Student Performance | 1,044 | No | `student+performance/student/student-mat.csv` |
| Student Mental Health | 101 | No | `Student Mental health.csv` |
| Campus Placement | 215 | No | `Placement_Data_Full_Class.csv` |
| Student Academics | 131 | No | `academics/data.csv` |

---

## Temporal Leakage Warning

All features in `oulad_ml_table.csv` cover the **full course duration**.
This makes every model trained on it a **final-outcome predictor**, not an
early-warning system.

Features that are end-of-course (leakage for early-warning):
`avg_score`, `assessment_completion_ratio`, `inactivity_days`,
`assessment_score_trend`, `week_click_sum_1_12`, `longest_inactive_gap`

For genuine early-warning, use `oulad_ml_table_week8.csv` with:
```bash
python oulad_baseline.py --input oulad_ml_table_week8.csv --mode early-warning
```
or:
```bash
python multisource_ablation.py --mode early-warning
```

---

## Evaluation Metrics Reported

Every script reports the same standard set:

| Metric | Why |
|---|---|
| Accuracy | Overall correctness |
| F1-Macro | Unweighted average across classes — penalises ignoring rare classes |
| F1-Weighted | Class-size-weighted F1 |
| Balanced Accuracy | Mean recall per class — robust to imbalance |
| Cohen's Kappa | Agreement above chance |
| MCC | Matthews Correlation Coefficient — best single metric for imbalanced data |
| ROC-AUC | Discrimination ability (OvR macro for multiclass) |

---

## Citation

If using this pipeline in research, please cite the source datasets:

- **OULAD:** Kuzilek et al. (2017), *Open University Learning Analytics Dataset*, Scientific Data
- **UCI Dropout:** Realinho et al. (2022), *Predicting Student Dropout and Academic Success*, Data
- **xAPI:** Amrieh et al. (2016), *Mining Educational Data to Predict Student Performance*
- **UCI Student Performance:** Cortez & Silva (2008), *Using Data Mining to Predict Secondary School Performance*
