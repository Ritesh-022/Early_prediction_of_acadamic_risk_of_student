#!/usr/bin/env python3
"""
generate_main_results.py

Generates 10 publication-quality combined figures for binary and 4-class
classification results. All figures are saved to figures/main_results/.

Figures produced
----------------
Binary:
  B01_binary_model_performance.png
  B02_binary_confusion_matrix_comparison.png
  B03_binary_roc_comparison.png
  B04_binary_precision_recall_comparison.png
  B05_binary_shap_importance_comparison.png

4-class:
  M01_four_class_model_performance.png
  M02_four_class_confusion_matrix_comparison.png
  M03_four_class_macro_roc_comparison.png
  M04_four_class_classwise_roc.png
  M05_four_class_shap_importance_comparison.png

Usage
-----
    python generate_main_results.py

All data is read from existing results/ prediction CSVs and SHAP CSVs.
No model rerun is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

# ============================================================
# PATHS
# ============================================================

ROOT       = Path(__file__).resolve().parent
PRED_DIR   = ROOT / "results" / "high_accuracy"
SHAP_DIR   = ROOT / "results"
HA_CSV     = PRED_DIR / "high_accuracy_results.csv"
OUT_DIR    = ROOT / "figures" / "main_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["xgboost", "lightgbm", "catboost", "random_forest"]
MODEL_LABELS = {
    "xgboost":       "XGBoost",
    "lightgbm":      "LightGBM",
    "catboost":      "CatBoost",
    "random_forest": "Random Forest",
}
COLORS = {
    "xgboost":       "#4878D0",
    "lightgbm":      "#EE854A",
    "catboost":      "#6ACC65",
    "random_forest": "#D65F5F",
}

BINARY_CLASSES  = ["AtRisk", "Success"]
FOUR_CLASSES    = ["Distinction", "Fail", "Pass", "Withdrawn"]


# ============================================================
# HELPERS
# ============================================================

def save(name: str) -> None:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path.relative_to(ROOT)}")


def load_pred(mode: str, model: str) -> pd.DataFrame | None:
    """Load predictions_oulad_{mode}_{model}.csv"""
    p = PRED_DIR / f"predictions_oulad_{mode}_{model}.csv"
    if not p.exists():
        print(f"  [MISSING] {p.name}")
        return None
    return pd.read_csv(p)


def load_shap(mode: str, model: str) -> pd.DataFrame | None:
    """Load shap_oulad_{mode}_{model}.csv — checks results/ and results/high_accuracy/."""
    fname = f"shap_oulad_{mode}_{model}.csv"
    # Check primary location first, then high_accuracy subdir
    for search_dir in (SHAP_DIR, PRED_DIR):
        p = search_dir / fname
        if p.exists():
            return pd.read_csv(p)
    print(f"  [MISSING] {fname}")
    return None


def load_metrics(dataset_prefix: str) -> pd.DataFrame:
    """Filter high_accuracy_results.csv for a given dataset prefix."""
    df = pd.read_csv(HA_CSV)
    sub = df[df["dataset"].str.startswith(dataset_prefix)].copy()
    sub["model_key"] = sub["model"].str.lower().str.replace(" ", "_")
    return sub


def panel_label(idx: int) -> str:
    return "(" + "abcdefghij"[idx] + ")"


# ============================================================
# B01  Binary model performance bar chart
# ============================================================

def b01_binary_model_performance() -> None:
    print("\n[B01] Binary model performance...")
    metrics = load_metrics("oulad_binary")
    if metrics.empty:
        print("  [SKIP] No binary metrics found.")
        return

    cols    = ["accuracy", "f1_macro", "balanced_accuracy", "roc_auc"]
    labels  = ["Accuracy", "F1 Macro", "Balanced Acc", "ROC-AUC"]
    ordered = [m for m in MODELS if m in metrics["model_key"].values]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    n = len(ordered)
    w = 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    for i, model_key in enumerate(ordered):
        row = metrics[metrics["model_key"] == model_key].iloc[0]
        vals = [row.get(c, np.nan) for c in cols]
        bars = ax.bar(x + offsets[i], vals, w,
                      label=MODEL_LABELS[model_key],
                      color=COLORS[model_key], alpha=0.88)
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.003, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.88, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Binary Classification — Model Performance Comparison\n(OULAD: AtRisk vs Success)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save("B01_binary_model_performance.png")


# ============================================================
# B02  Binary confusion matrix — 4 panels
# ============================================================

def b02_binary_confusion_matrices() -> None:
    print("\n[B02] Binary confusion matrices...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        "Binary Classification — Confusion Matrices per Model\n(OULAD: AtRisk vs Success)",
        fontsize=13, y=1.02
    )

    for i, model_key in enumerate(MODELS):
        ax = axes[i]
        df = load_pred("binary", model_key)
        if df is None:
            ax.set_visible(False)
            continue

        labels = sorted(df["y_true"].unique())
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")
        ax.set_title(f"{panel_label(i)}  {MODEL_LABELS[model_key]}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if i == 0 else "")

    save("B02_binary_confusion_matrix_comparison.png")


# ============================================================
# B03  Binary ROC — all models on one plot
# ============================================================

def b03_binary_roc() -> None:
    print("\n[B03] Binary ROC comparison...")
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_key in MODELS:
        df = load_pred("binary", model_key)
        if df is None:
            continue

        # Resolve score column: prefer prob_AtRisk, then 1-prob_Success, then any prob_ col
        if "prob_AtRisk" in df.columns:
            scores = df["prob_AtRisk"]
        elif "prob_Success" in df.columns:
            scores = 1.0 - df["prob_Success"]
        else:
            prob_cols = [c for c in df.columns if c.lower().startswith("prob_")]
            if not prob_cols:
                print(f"  [SKIP] No probability columns for {model_key}")
                continue
            scores = df[prob_cols[0]]

        y_true_bin = (df["y_true"] == "AtRisk").astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2,
                color=COLORS[model_key],
                label=f"{MODEL_LABELS[model_key]}  AUC = {roc_auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Binary ROC Curves — All Models\n(OULAD: AtRisk vs Success)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    save("B03_binary_roc_comparison.png")


# ============================================================
# B04  Binary Precision-Recall — all models on one plot
# ============================================================

def b04_binary_pr() -> None:
    print("\n[B04] Binary Precision-Recall comparison...")
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_key in MODELS:
        df = load_pred("binary", model_key)
        if df is None:
            continue

        y_true_bin = (df["y_true"] == "AtRisk").astype(int)
        if "prob_AtRisk" in df.columns:
            scores = df["prob_AtRisk"]
        elif "prob_Success" in df.columns:
            scores = 1.0 - df["prob_Success"]
        else:
            continue

        precision, recall, _ = precision_recall_curve(y_true_bin, scores)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=2,
                color=COLORS[model_key],
                label=f"{MODEL_LABELS[model_key]}  AUC = {pr_auc:.4f}")

    # Baseline: proportion of positive class
    baseline = (df["y_true"] == "AtRisk").mean() if df is not None else 0.5
    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Baseline (prevalence = {baseline:.2f})")

    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Binary Precision-Recall Curves — All Models\n(OULAD: AtRisk vs Success)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    save("B04_binary_precision_recall_comparison.png")


# ============================================================
# B05  Binary SHAP importance — 4 panels (or 3 if RF missing)
# ============================================================

def b05_binary_shap() -> None:
    print("\n[B05] Binary SHAP importance comparison...")
    # Random Forest SHAP may not be saved — collect what exists
    available = [(m, load_shap("binary", m)) for m in MODELS]
    available = [(m, df) for m, df in available if df is not None]

    if not available:
        print("  [SKIP] No SHAP files found.")
        return

    n = len(available)
    ncols = min(n, 4)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6), sharey=False)
    if ncols == 1:
        axes = [axes]

    fig.suptitle(
        "Binary Classification — SHAP Feature Importance per Model\n(OULAD: AtRisk vs Success)",
        fontsize=13, y=1.02
    )

    for i, (model_key, shap_df) in enumerate(available):
        ax = axes[i]
        feat_col = next(
            (c for c in shap_df.columns if "feature" in c.lower()), None
        )
        imp_col = next(
            (c for c in shap_df.columns
             if "importance" in c.lower() or "shap" in c.lower()), None
        )
        if feat_col is None or imp_col is None:
            ax.set_visible(False)
            continue

        shap_df[imp_col] = pd.to_numeric(shap_df[imp_col], errors="coerce")
        plot_df = (
            shap_df[[feat_col, imp_col]].dropna()
            .sort_values(imp_col, ascending=False).head(15)
            .sort_values(imp_col)
        )
        ax.barh(plot_df[feat_col], plot_df[imp_col],
                color=COLORS[model_key], alpha=0.85)
        ax.set_title(f"{panel_label(i)}  {MODEL_LABELS[model_key]}", fontsize=11)
        ax.set_xlabel("Mean |SHAP|")
        if i == 0:
            ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.3)

    save("B05_binary_shap_importance_comparison.png")


# ============================================================
# M01  4-class model performance bar chart
# ============================================================

def m01_four_class_model_performance() -> None:
    print("\n[M01] 4-class model performance...")
    metrics = load_metrics("oulad_4class")
    if metrics.empty:
        print("  [SKIP] No 4-class metrics found.")
        return

    cols   = ["accuracy", "f1_macro", "balanced_accuracy", "roc_auc"]
    labels = ["Accuracy", "F1 Macro", "Balanced Acc", "ROC-AUC"]
    ordered = [m for m in MODELS if m in metrics["model_key"].values]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    n = len(ordered)
    w = 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    for i, model_key in enumerate(ordered):
        row = metrics[metrics["model_key"] == model_key].iloc[0]
        vals = [row.get(c, np.nan) for c in cols]
        bars = ax.bar(x + offsets[i], vals, w,
                      label=MODEL_LABELS[model_key],
                      color=COLORS[model_key], alpha=0.88)
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.003, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.60, 1.00)
    ax.set_ylabel("Score")
    ax.set_title("4-Class Classification — Model Performance Comparison\n"
                 "(OULAD: Distinction / Fail / Pass / Withdrawn)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save("M01_four_class_model_performance.png")


# ============================================================
# M02  4-class confusion matrices — 4 panels
# ============================================================

def m02_four_class_confusion_matrices() -> None:
    print("\n[M02] 4-class confusion matrices...")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        "4-Class Classification — Confusion Matrices per Model\n"
        "(OULAD: Distinction / Fail / Pass / Withdrawn)",
        fontsize=13, y=1.02
    )

    for i, model_key in enumerate(MODELS):
        ax = axes[i]
        df = load_pred("4class", model_key)
        if df is None:
            ax.set_visible(False)
            continue

        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=FOUR_CLASSES)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=FOUR_CLASSES)
        disp.plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")
        ax.set_title(f"{panel_label(i)}  {MODEL_LABELS[model_key]}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if i == 0 else "")
        # Rotate x tick labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    save("M02_four_class_confusion_matrix_comparison.png")


# ============================================================
# M03  4-class macro ROC — all models on one plot
# ============================================================

def m03_four_class_macro_roc() -> None:
    print("\n[M03] 4-class macro ROC comparison...")
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_key in MODELS:
        df = load_pred("4class", model_key)
        if df is None:
            continue

        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if not prob_cols:
            print(f"  [SKIP] No probability columns for {model_key}")
            continue

        y_true_bin = label_binarize(df["y_true"], classes=FOUR_CLASSES)
        prob_mat   = df[prob_cols].values

        # macro-average ROC
        all_fpr = np.unique(np.concatenate([
            roc_curve(y_true_bin[:, k], prob_mat[:, k])[0]
            for k in range(len(FOUR_CLASSES))
        ]))
        mean_tpr = np.zeros_like(all_fpr)
        for k in range(len(FOUR_CLASSES)):
            fpr_k, tpr_k, _ = roc_curve(y_true_bin[:, k], prob_mat[:, k])
            mean_tpr += np.interp(all_fpr, fpr_k, tpr_k)
        mean_tpr /= len(FOUR_CLASSES)
        macro_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr, lw=2,
                color=COLORS[model_key],
                label=f"{MODEL_LABELS[model_key]}  Macro AUC = {macro_auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("4-Class Macro-Average ROC Curves — All Models\n"
                 "(OULAD: Distinction / Fail / Pass / Withdrawn)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    save("M03_four_class_macro_roc_comparison.png")


# ============================================================
# M04  4-class class-wise ROC — best model (LightGBM by F1)
# ============================================================

def m04_four_class_classwise_roc() -> None:
    print("\n[M04] 4-class class-wise ROC (best model)...")

    # Pick best model by F1 macro from results CSV
    metrics = load_metrics("oulad_4class")
    best_key = "lightgbm"  # default
    if not metrics.empty:
        best_row = metrics.sort_values("f1_macro", ascending=False).iloc[0]
        best_key = best_row["model_key"]

    df = load_pred("4class", best_key)
    if df is None:
        print("  [SKIP] No 4-class prediction file found.")
        return

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        print("  [SKIP] No probability columns.")
        return

    y_true_bin = label_binarize(df["y_true"], classes=FOUR_CLASSES)
    prob_mat   = df[prob_cols].values

    class_colors = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for k, cls in enumerate(FOUR_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, k], prob_mat[:, k])
        cls_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=class_colors[k],
                label=f"{cls} vs Rest   AUC = {cls_auc:.4f}")

    # Macro average
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_true_bin[:, k], prob_mat[:, k])[0]
        for k in range(len(FOUR_CLASSES))
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for k in range(len(FOUR_CLASSES)):
        fpr_k, tpr_k, _ = roc_curve(y_true_bin[:, k], prob_mat[:, k])
        mean_tpr += np.interp(all_fpr, fpr_k, tpr_k)
    mean_tpr /= len(FOUR_CLASSES)
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, "k-", lw=2.5,
            label=f"Macro-average   AUC = {macro_auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"4-Class One-vs-Rest ROC Curves\n"
        f"(Best model: {MODEL_LABELS.get(best_key, best_key)})"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    save("M04_four_class_classwise_roc.png")


# ============================================================
# M05  4-class SHAP importance — available model panels
# ============================================================

def m05_four_class_shap() -> None:
    print("\n[M05] 4-class SHAP importance comparison...")
    available = [(m, load_shap("4class", m)) for m in MODELS]
    available = [(m, df) for m, df in available if df is not None]

    if not available:
        print("  [SKIP] No 4-class SHAP files found.")
        return

    n = len(available)
    ncols = min(n, 4)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6), sharey=False)
    if ncols == 1:
        axes = [axes]

    fig.suptitle(
        "4-Class Classification — SHAP Feature Importance per Model\n"
        "(OULAD: Distinction / Fail / Pass / Withdrawn)",
        fontsize=13, y=1.02
    )

    for i, (model_key, shap_df) in enumerate(available):
        ax = axes[i]
        feat_col = next(
            (c for c in shap_df.columns if "feature" in c.lower()), None
        )
        imp_col = next(
            (c for c in shap_df.columns
             if "importance" in c.lower() or "shap" in c.lower()), None
        )
        if feat_col is None or imp_col is None:
            ax.set_visible(False)
            continue

        shap_df[imp_col] = pd.to_numeric(shap_df[imp_col], errors="coerce")
        plot_df = (
            shap_df[[feat_col, imp_col]].dropna()
            .sort_values(imp_col, ascending=False).head(15)
            .sort_values(imp_col)
        )
        ax.barh(plot_df[feat_col], plot_df[imp_col],
                color=COLORS[model_key], alpha=0.85)
        ax.set_title(f"{panel_label(i)}  {MODEL_LABELS[model_key]}", fontsize=11)
        ax.set_xlabel("Mean |SHAP|")
        if i == 0:
            ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.3)

    save("M05_four_class_shap_importance_comparison.png")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 54)
    print("  MAIN RESULTS — PUBLICATION FIGURE GENERATOR")
    print("=" * 54)
    print(f"  Output → {OUT_DIR}")

    # Binary
    b01_binary_model_performance()
    b02_binary_confusion_matrices()
    b03_binary_roc()
    b04_binary_pr()
    b05_binary_shap()

    # 4-class
    m01_four_class_model_performance()
    m02_four_class_confusion_matrices()
    m03_four_class_macro_roc()
    m04_four_class_classwise_roc()
    m05_four_class_shap()

    generated = sorted(OUT_DIR.glob("*.png"))
    print("\n" + "=" * 54)
    print(f"  DONE — {len(generated)} figures saved to:")
    print(f"  {OUT_DIR}")
    print("=" * 54)
    for p in generated:
        print(f"  ✓ {p.name}")


if __name__ == "__main__":
    main()
