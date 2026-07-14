#!/usr/bin/env python3
"""
update_main_results.py
======================
Merges extended_results.csv (DNN / MLP / ET / BDT / DT) with the existing
high_accuracy_results.csv (XGB / LGB / CAT / RF) and regenerates the 10
main-results publication figures, now showing ALL models side-by-side.

Updated figures are saved to figures/main_results/ and overwrite the old ones.

Usage
-----
    python update_main_results.py

Run AFTER:
    python extended_models_pipeline.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, auc, confusion_matrix,
    precision_recall_curve, roc_curve,
)
from sklearn.preprocessing import label_binarize

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

ROOT     = Path(__file__).resolve().parent
HA_DIR   = ROOT / "results" / "high_accuracy"
EXT_DIR  = ROOT / "results" / "extended"
OUT_DIR  = ROOT / "figures" / "main_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical model display order (existing → new)
MODEL_ORDER = [
    "xgboost", "lightgbm", "catboost", "random_forest",
    "et", "bdt", "dt", "mlp", "dnn",
]
MODEL_LABELS = {
    "xgboost":       "XGBoost",
    "lightgbm":      "LightGBM",
    "catboost":      "CatBoost",
    "random_forest": "Random Forest",
    "et":            "Extra Trees",
    "bdt":           "Bagged DT",
    "dt":            "Decision Tree",
    "mlp":           "MLP",
    "dnn":           "DNN",
}
# Colour palette — 9 distinct colours
PALETTE = [
    "#4878D0", "#EE854A", "#6ACC65", "#D65F5F",
    "#B47CC7", "#C4AD66", "#77BEDB", "#E47298", "#70A89F",
]
COLORS = {k: PALETTE[i] for i, k in enumerate(MODEL_ORDER)}

BINARY_CLASSES = ["AtRisk", "Success"]
FOUR_CLASSES   = ["Distinction", "Fail", "Pass", "Withdrawn"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save(name: str) -> None:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path.relative_to(ROOT)}")


def load_all_metrics() -> pd.DataFrame:
    """Merge high_accuracy_results.csv with extended_results.csv."""
    frames = []

    ha_path = HA_DIR / "high_accuracy_results.csv"
    if ha_path.exists():
        frames.append(pd.read_csv(ha_path))
    else:
        print(f"  [WARN] {ha_path} not found — existing models will be missing.")

    ext_path = EXT_DIR / "extended_results.csv"
    if ext_path.exists():
        frames.append(pd.read_csv(ext_path))
    else:
        print(f"  [WARN] {ext_path} not found — run extended_models_pipeline.py first.")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["model_key"] = df["model"].str.lower().str.replace(" ", "_")
    return df


def load_pred(mode: str, model_key: str) -> Optional[pd.DataFrame]:
    """Look for prediction CSV in extended/ first, then high_accuracy/."""
    for d in [EXT_DIR, HA_DIR]:
        p = d / f"predictions_oulad_{mode}_{model_key}.csv"
        if p.exists():
            return pd.read_csv(p)
    return None


def load_shap(mode: str, model_key: str) -> Optional[pd.DataFrame]:
    """Look for SHAP CSV in extended/, results/high_accuracy/, then results/."""
    for d in [EXT_DIR, HA_DIR, ROOT / "results"]:
        p = d / f"shap_oulad_{mode}_{model_key}.csv"
        if p.exists():
            return pd.read_csv(p)
    return None


def ordered_models(available_keys: List[str]) -> List[str]:
    """Return model keys in canonical display order, filtered to available."""
    return [k for k in MODEL_ORDER if k in available_keys]


def panel_label(idx: int) -> str:
    return "(" + "abcdefghijklmnop"[idx] + ")"


# ══════════════════════════════════════════════════════════════════════════════
# B01  Binary model performance
# ══════════════════════════════════════════════════════════════════════════════

def b01_binary_model_performance(metrics: pd.DataFrame) -> None:
    print("\n[B01] Binary model performance...")
    sub = metrics[metrics["dataset"].str.startswith("oulad_binary")].copy()
    if sub.empty:
        print("  [SKIP] No binary metrics."); return

    cols      = ["accuracy", "f1_macro", "balanced_accuracy", "roc_auc"]
    col_labels = ["Accuracy", "F1 Macro", "Balanced Acc", "ROC-AUC"]
    keys      = ordered_models(sub["model_key"].tolist())

    n, x, w = len(keys), np.arange(len(cols)), 0.8 / max(len(keys), 1)
    offsets  = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 5))
    for i, mk in enumerate(keys):
        row  = sub[sub["model_key"] == mk].iloc[0]
        vals = [row.get(c, np.nan) for c in cols]
        bars = ax.bar(x + offsets[i], vals, w,
                      label=MODEL_LABELS.get(mk, mk),
                      color=COLORS.get(mk, "#888888"), alpha=0.88)
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        float(v) + 0.003, f"{float(v):.3f}",
                        ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x); ax.set_xticklabels(col_labels)
    ax.set_ylim(0.85, 1.01); ax.set_ylabel("Score")
    ax.set_title("Binary Classification — All Models Performance\n"
                 "(OULAD: AtRisk vs Success)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    save("B01_binary_model_performance.png")


# ══════════════════════════════════════════════════════════════════════════════
# B02  Binary confusion matrices
# ══════════════════════════════════════════════════════════════════════════════

def b02_binary_confusion_matrices() -> None:
    print("\n[B02] Binary confusion matrices...")
    keys = [mk for mk in MODEL_ORDER if load_pred("binary", mk) is not None]
    if not keys:
        print("  [SKIP] No binary prediction files found."); return

    ncols = min(len(keys), 5)
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    fig.suptitle("Binary Classification — Confusion Matrices (All Models)\n"
                 "(OULAD: AtRisk vs Success)", fontsize=13, y=1.01)

    for i, mk in enumerate(keys):
        ax = axes[i]
        df = load_pred("binary", mk)
        labels = sorted(df["y_true"].unique())
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=labels)
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, values_format="d", colorbar=False, cmap="Blues")
        ax.set_title(f"{panel_label(i)}  {MODEL_LABELS.get(mk, mk)}", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual" if i % ncols == 0 else "")

    for j in range(len(keys), len(axes)):
        axes[j].set_visible(False)

    save("B02_binary_confusion_matrix_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# B03  Binary ROC comparison
# ══════════════════════════════════════════════════════════════════════════════

def b03_binary_roc() -> None:
    print("\n[B03] Binary ROC comparison...")
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0

    for mk in MODEL_ORDER:
        df = load_pred("binary", mk)
        if df is None: continue

        y_bin = (df["y_true"] == "AtRisk").astype(int)
        if "prob_AtRisk" in df.columns:
            scores = df["prob_AtRisk"]
        elif "prob_Success" in df.columns:
            scores = 1.0 - df["prob_Success"]
        else:
            continue

        fpr, tpr, _ = roc_curve(y_bin, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=COLORS.get(mk, "#888888"),
                label=f"{MODEL_LABELS.get(mk, mk):<18} AUC = {roc_auc:.4f}")
        plotted += 1

    if not plotted:
        print("  [SKIP] No binary prediction files with probabilities."); plt.close(); return

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlim([-0.01, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Binary ROC Curves — All Models\n(OULAD: AtRisk vs Success)")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    save("B03_binary_roc_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# B04  Binary Precision-Recall
# ══════════════════════════════════════════════════════════════════════════════

def b04_binary_pr() -> None:
    print("\n[B04] Binary Precision-Recall comparison...")
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0; last_df = None

    for mk in MODEL_ORDER:
        df = load_pred("binary", mk)
        if df is None: continue

        y_bin = (df["y_true"] == "AtRisk").astype(int)
        if "prob_AtRisk" in df.columns:
            scores = df["prob_AtRisk"]
        elif "prob_Success" in df.columns:
            scores = 1.0 - df["prob_Success"]
        else:
            continue

        prec, rec, _ = precision_recall_curve(y_bin, scores)
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, lw=2, color=COLORS.get(mk, "#888888"),
                label=f"{MODEL_LABELS.get(mk, mk):<18} AUC = {pr_auc:.4f}")
        plotted += 1; last_df = df

    if not plotted:
        print("  [SKIP] No binary prediction files with probabilities."); plt.close(); return

    baseline = (last_df["y_true"] == "AtRisk").mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Baseline (prevalence={baseline:.2f})")
    ax.set_xlim([0, 1.01]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Binary Precision-Recall Curves — All Models\n(OULAD: AtRisk vs Success)")
    ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=0.3)
    save("B04_binary_precision_recall_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# B05  Binary SHAP importance
# ══════════════════════════════════════════════════════════════════════════════

def b05_binary_shap() -> None:
    print("\n[B05] Binary SHAP importance comparison...")
    available = [(mk, load_shap("binary", mk)) for mk in MODEL_ORDER]
    available = [(mk, df) for mk, df in available if df is not None]
    if not available:
        print("  [SKIP] No SHAP files found."); return

    n = len(available)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 6 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    fig.suptitle("Binary Classification — SHAP Feature Importance (All Models)\n"
                 "(OULAD: AtRisk vs Success)", fontsize=13, y=1.01)

    for i, (mk, shap_df) in enumerate(available):
        ax = axes[i]
        feat_col = next((c for c in shap_df.columns if "feature" in c.lower()), None)
        imp_col  = next((c for c in shap_df.columns
                         if "importance" in c.lower() or "shap" in c.lower()), None)
        if feat_col is None or imp_col is None:
            ax.set_visible(False); continue

        shap_df[imp_col] = pd.to_numeric(shap_df[imp_col], errors="coerce")
        plot_df = (shap_df[[feat_col, imp_col]].dropna()
                   .sort_values(imp_col, ascending=False).head(15)
                   .sort_values(imp_col))
        ax.barh(plot_df[feat_col], plot_df[imp_col],
                color=COLORS.get(mk, "#888888"), alpha=0.85)
        ax.set_title(f"{panel_label(i)}  {MODEL_LABELS.get(mk, mk)}", fontsize=10)
        ax.set_xlabel("Mean |SHAP|")
        if i % ncols == 0: ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.3)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    save("B05_binary_shap_importance_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# M01  4-class model performance
# ══════════════════════════════════════════════════════════════════════════════

def m01_four_class_model_performance(metrics: pd.DataFrame) -> None:
    print("\n[M01] 4-class model performance...")
    sub = metrics[metrics["dataset"].str.startswith("oulad_4class")].copy()
    if sub.empty:
        print("  [SKIP] No 4-class metrics."); return

    cols       = ["accuracy", "f1_macro", "balanced_accuracy", "roc_auc"]
    col_labels = ["Accuracy", "F1 Macro", "Balanced Acc", "ROC-AUC"]
    keys       = ordered_models(sub["model_key"].tolist())

    n, x, w = len(keys), np.arange(len(cols)), 0.8 / max(len(keys), 1)
    offsets  = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 5))
    for i, mk in enumerate(keys):
        row  = sub[sub["model_key"] == mk].iloc[0]
        vals = [row.get(c, np.nan) for c in cols]
        bars = ax.bar(x + offsets[i], vals, w,
                      label=MODEL_LABELS.get(mk, mk),
                      color=COLORS.get(mk, "#888888"), alpha=0.88)
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        float(v) + 0.003, f"{float(v):.3f}",
                        ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x); ax.set_xticklabels(col_labels)
    ax.set_ylim(0.55, 1.00); ax.set_ylabel("Score")
    ax.set_title("4-Class Classification — All Models Performance\n"
                 "(OULAD: Distinction / Fail / Pass / Withdrawn)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    save("M01_four_class_model_performance.png")


# ══════════════════════════════════════════════════════════════════════════════
# M02  4-class confusion matrices
# ══════════════════════════════════════════════════════════════════════════════

def m02_four_class_confusion_matrices() -> None:
    print("\n[M02] 4-class confusion matrices...")
    keys = [mk for mk in MODEL_ORDER if load_pred("4class", mk) is not None]
    if not keys:
        print("  [SKIP] No 4-class prediction files found."); return

    ncols = min(len(keys), 5)
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    fig.suptitle("4-Class Classification — Confusion Matrices (All Models)\n"
                 "(OULAD: Distinction / Fail / Pass / Withdrawn)", fontsize=13, y=1.01)

    for i, mk in enumerate(keys):
        ax  = axes[i]
        df  = load_pred("4class", mk)
        cm  = confusion_matrix(df["y_true"], df["y_pred"], labels=FOUR_CLASSES)
        ConfusionMatrixDisplay(cm, display_labels=FOUR_CLASSES).plot(
            ax=ax, values_format="d", colorbar=False, cmap="Blues")
        ax.set_title(f"{panel_label(i)}  {MODEL_LABELS.get(mk, mk)}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if i % ncols == 0 else "")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    for j in range(len(keys), len(axes)):
        axes[j].set_visible(False)

    save("M02_four_class_confusion_matrix_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# M03  4-class macro ROC comparison
# ══════════════════════════════════════════════════════════════════════════════

def m03_four_class_macro_roc() -> None:
    print("\n[M03] 4-class macro ROC comparison...")
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0

    for mk in MODEL_ORDER:
        df = load_pred("4class", mk)
        if df is None: continue
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if not prob_cols: continue

        y_bin    = label_binarize(df["y_true"], classes=FOUR_CLASSES)
        prob_mat = df[prob_cols].values

        all_fpr = np.unique(np.concatenate([
            roc_curve(y_bin[:, k], prob_mat[:, k])[0]
            for k in range(len(FOUR_CLASSES))
        ]))
        mean_tpr = np.zeros_like(all_fpr)
        for k in range(len(FOUR_CLASSES)):
            fpr_k, tpr_k, _ = roc_curve(y_bin[:, k], prob_mat[:, k])
            mean_tpr += np.interp(all_fpr, fpr_k, tpr_k)
        mean_tpr /= len(FOUR_CLASSES)
        macro_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr, lw=2, color=COLORS.get(mk, "#888888"),
                label=f"{MODEL_LABELS.get(mk, mk):<18} Macro AUC = {macro_auc:.4f}")
        plotted += 1

    if not plotted:
        print("  [SKIP] No 4-class prediction files with probabilities."); plt.close(); return

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlim([-0.01, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("4-Class Macro-Average ROC — All Models\n"
                 "(OULAD: Distinction / Fail / Pass / Withdrawn)")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    save("M03_four_class_macro_roc_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# M04  4-class class-wise ROC — best model by F1
# ══════════════════════════════════════════════════════════════════════════════

def m04_four_class_classwise_roc(metrics: pd.DataFrame) -> None:
    print("\n[M04] 4-class class-wise ROC (best model)...")
    sub = metrics[metrics["dataset"].str.startswith("oulad_4class")]

    best_key = "lightgbm"
    if not sub.empty:
        best_row = sub.sort_values("f1_macro", ascending=False).iloc[0]
        best_key = best_row["model_key"]

    df = load_pred("4class", best_key)
    if df is None:
        print(f"  [SKIP] No prediction file for {best_key}."); return

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        print("  [SKIP] No probability columns."); return

    y_bin    = label_binarize(df["y_true"], classes=FOUR_CLASSES)
    prob_mat = df[prob_cols].values
    cls_colors = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for k, cls in enumerate(FOUR_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, k], prob_mat[:, k])
        cls_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=cls_colors[k],
                label=f"{cls} vs Rest   AUC = {cls_auc:.4f}")

    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, k], prob_mat[:, k])[0]
        for k in range(len(FOUR_CLASSES))
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for k in range(len(FOUR_CLASSES)):
        fpr_k, tpr_k, _ = roc_curve(y_bin[:, k], prob_mat[:, k])
        mean_tpr += np.interp(all_fpr, fpr_k, tpr_k)
    mean_tpr /= len(FOUR_CLASSES)
    ax.plot(all_fpr, mean_tpr, "k-", lw=2.5,
            label=f"Macro-average   AUC = {auc(all_fpr, mean_tpr):.4f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlim([-0.01, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"4-Class One-vs-Rest ROC Curves\n"
                 f"(Best model: {MODEL_LABELS.get(best_key, best_key)})")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    save("M04_four_class_classwise_roc.png")


# ══════════════════════════════════════════════════════════════════════════════
# M05  4-class SHAP importance
# ══════════════════════════════════════════════════════════════════════════════

def m05_four_class_shap() -> None:
    print("\n[M05] 4-class SHAP importance comparison...")
    available = [(mk, load_shap("4class", mk)) for mk in MODEL_ORDER]
    available = [(mk, df) for mk, df in available if df is not None]
    if not available:
        print("  [SKIP] No 4-class SHAP files found."); return

    n = len(available)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 6 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    fig.suptitle("4-Class Classification — SHAP Feature Importance (All Models)\n"
                 "(OULAD: Distinction / Fail / Pass / Withdrawn)", fontsize=13, y=1.01)

    for i, (mk, shap_df) in enumerate(available):
        ax = axes[i]
        feat_col = next((c for c in shap_df.columns if "feature" in c.lower()), None)
        imp_col  = next((c for c in shap_df.columns
                         if "importance" in c.lower() or "shap" in c.lower()), None)
        if feat_col is None or imp_col is None:
            ax.set_visible(False); continue

        shap_df[imp_col] = pd.to_numeric(shap_df[imp_col], errors="coerce")
        plot_df = (shap_df[[feat_col, imp_col]].dropna()
                   .sort_values(imp_col, ascending=False).head(15)
                   .sort_values(imp_col))
        ax.barh(plot_df[feat_col], plot_df[imp_col],
                color=COLORS.get(mk, "#888888"), alpha=0.85)
        ax.set_title(f"{panel_label(i)}  {MODEL_LABELS.get(mk, mk)}", fontsize=10)
        ax.set_xlabel("Mean |SHAP|")
        if i % ncols == 0: ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.3)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    save("M05_four_class_shap_importance_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE  (merged CSV + console print)
# ══════════════════════════════════════════════════════════════════════════════

def save_summary_table(metrics: pd.DataFrame) -> None:
    print("\n[TABLE] Saving merged summary table...")
    if metrics.empty:
        return

    # Save full merged CSV
    out_csv = ROOT / "results" / "all_models_comparison.csv"
    metrics.to_csv(out_csv, index=False)
    print(f"  [SAVED] {out_csv.relative_to(ROOT)}")

    # Console print
    print("\n" + "=" * 105)
    print(f"{'DATASET':<22} {'MODEL':<16} {'ACC':>7} {'F1-MAC':>7} "
          f"{'BACC':>7} {'KAPPA':>7} {'MCC':>7} {'ROC-AUC':>8}")
    print("=" * 105)
    for _, row in metrics.sort_values(
        ["dataset", "accuracy"], ascending=[True, False]
    ).iterrows():
        star = " ★" if row["accuracy"] >= 0.95 else (
               " ↑" if row["accuracy"] >= 0.90 else "")
        print(
            f"{row['dataset']:<22} {row['model']:<16} "
            f"{row['accuracy']:>7.4f} {row['f1_macro']:>7.4f} "
            f"{row['balanced_accuracy']:>7.4f} {row['cohen_kappa']:>7.4f} "
            f"{row['mcc']:>7.4f} {row.get('roc_auc', float('nan')):>8.4f}"
            f"{star}"
        )
    print("=" * 105)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  UPDATE MAIN RESULTS — ALL MODELS COMPARISON")
    print("=" * 60)
    print(f"  Output → {OUT_DIR}")

    metrics = load_all_metrics()
    if metrics.empty:
        print("\n[ERROR] No metrics found. Run high_accuracy_pipeline.py "
              "and/or extended_models_pipeline.py first.")
        return

    print(f"\n  Models found: {sorted(metrics['model_key'].unique())}")
    print(f"  Datasets    : {sorted(metrics['dataset'].unique())}\n")

    # Binary figures
    b01_binary_model_performance(metrics)
    b02_binary_confusion_matrices()
    b03_binary_roc()
    b04_binary_pr()
    b05_binary_shap()

    # 4-class figures
    m01_four_class_model_performance(metrics)
    m02_four_class_confusion_matrices()
    m03_four_class_macro_roc()
    m04_four_class_classwise_roc(metrics)
    m05_four_class_shap()

    # Save merged table
    save_summary_table(metrics)

    generated = sorted(OUT_DIR.glob("*.png"))
    print(f"\n{'=' * 60}")
    print(f"  DONE — {len(generated)} figures in figures/main_results/")
    print(f"{'=' * 60}")
    for p in generated:
        print(f"  ✓ {p.name}")


if __name__ == "__main__":
    main()
