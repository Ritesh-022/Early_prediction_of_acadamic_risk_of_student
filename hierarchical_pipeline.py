#!/usr/bin/env python3
"""
Hierarchical Probabilistic Classification Pipeline
===================================================
Implements the cascaded 3-model architecture:

    INPUT
      │
      ▼
  MODEL 1  (Binary: AtRisk vs Success)  ~95% accuracy
      │
  ┌───┴───┐
  ▼       ▼
AT-RISK  SUCCESS
  │         │
  ▼         ▼
MODEL 2A  MODEL 2B
Fail vs   Pass vs
Withdrawn Distinction

Final probability:
  P(Fail)        = P(AtRisk) × P(Fail|AtRisk)
  P(Withdrawn)   = P(AtRisk) × P(Withdrawn|AtRisk)
  P(Pass)        = P(Success) × P(Pass|Success)
  P(Distinction) = P(Success) × P(Distinction|Success)
  ŷ = argmax over four classes

Five experiments run in one script:
  Exp 1 : Direct 4-class baseline (XGBoost)               ← existing result
  Exp 2 : Hierarchical, no augmentation
  Exp 3 : Hierarchical + class weights
  Exp 4 : Hierarchical + SMOTE (train folds only)
  Exp 5 : Hierarchical + CTGAN (train split only)

Test set = 100% real original data, never touched by augmentation.

Usage:
    python hierarchical_pipeline.py
    python hierarchical_pipeline.py --skip-ctgan     # skip Exp 5 (slow)
    python hierarchical_pipeline.py --tune           # Optuna on each branch
"""
from __future__ import annotations
import argparse, warnings, logging, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier; _XGB = True
except ImportError:
    XGBClassifier = None; _XGB = False

try:
    from lightgbm import LGBMClassifier; _LGB = True
except ImportError:
    LGBMClassifier = None; _LGB = False

try:
    from catboost import CatBoostClassifier; _CAT = True
except ImportError:
    CatBoostClassifier = None; _CAT = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN; _SMOTE = True
except ImportError:
    _SMOTE = False

try:
    from ctgan import CTGAN; _CTGAN = True
except ImportError:
    _CTGAN = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# ── FIX: force UTF-8 stdout/stderr ──────────────────────────────────────────
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT   = Path(__file__).parent
SEED   = 42
logger = logging.getLogger(__name__)

# ── label constants ────────────────────────────────────────────────────────────
FOUR_CLASSES   = ["Distinction", "Fail", "Pass", "Withdrawn"]
BINARY_MAP     = {"Pass": "Success", "Distinction": "Success",
                  "Fail": "AtRisk",  "Withdrawn":  "AtRisk"}
ATRISK_CLASSES = ["Fail", "Withdrawn"]
SUCCESS_CLASSES= ["Pass", "Distinction"]

# ── features to always drop ────────────────────────────────────────────────────
DROP_ALWAYS = {
    "final_result", "id_student", "code_module", "code_presentation",
    "date_unregistration", "date_unreg", "date_unregistered",
    "weighted_score", "first_ts", "last_ts",
    "active_weeks", "clicks_per_active_week", "assessments_per_week",
    "activity_count", "days_active", "avg_clicks_per_day",
    "week_click_sum_1_4", "registration_delay_category",
    "last_assessment_day", "first_assessment_day",
    "id_assessment", "id_site",
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def load_oulad() -> pd.DataFrame:
    # Prefer V2 (rich temporal features) then V1 then week8 fallback
    for name in ["oulad_ml_table_v2.csv", "oulad_ml_table.csv", "oulad_ml_table_week8.csv"]:
        p = ROOT / name
        if p.exists():
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip()
            logger.info("Loaded %s — shape %s", name, df.shape)
            return df
    raise FileNotFoundError("No OULAD ML table found. Run oulad_pipeline_v2.py first.")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering on the OULAD ML table."""
    df = df.copy()
    # Drop everything in DROP_ALWAYS EXCEPT final_result (needed for split)
    drop_now = DROP_ALWAYS - {"final_result"}
    df = df.drop(columns=[c for c in drop_now if c in df.columns], errors="ignore")

    # zero-fill activity/assessment NaNs (absence = 0 activity)
    zero_cols = (
        [f"week{w}_clicks" for w in range(1, 13)]
        + ["total_clicks", "avg_score", "score_std", "num_assessments",
           "assessment_completion_ratio", "missed_assessments",
           "late_submission_count", "assessment_score_trend",
           "click_variance", "click_growth_rate", "longest_inactive_gap",
           "week_click_sum_1_12", "inactivity_days",
           "clicks_until_week2", "clicks_until_week4",
           "clicks_until_week6", "clicks_until_week8"]
        + [c for c in df.columns if c.startswith("activity_type_")]
    )
    for c in zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # log-transform click features
    for c in ["total_clicks"] + [f"week{w}_clicks" for w in range(1, 13)]:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(df[c])

    # engagement ratios
    if "clicks_until_week4" in df.columns and "total_clicks" in df.columns:
        denom = df["total_clicks"].replace(0, np.nan)
        df["early_ratio"] = (df["clicks_until_week4"] / denom).fillna(0).clip(0, 1)
    if "clicks_until_week8" in df.columns and "total_clicks" in df.columns:
        denom = df["total_clicks"].replace(0, np.nan)
        df["mid_ratio"] = (df["clicks_until_week8"] / denom).fillna(0).clip(0, 1)

    # score × completion interaction
    if "avg_score" in df.columns and "assessment_completion_ratio" in df.columns:
        df["score_x_completion"] = df["avg_score"] * df["assessment_completion_ratio"]
    if "avg_score" in df.columns and "studied_credits" in df.columns:
        df["score_per_credit"] = (df["avg_score"] /
                                   df["studied_credits"].replace(0, np.nan)).fillna(0)

    # binary flags
    if "total_clicks" in df.columns:
        df["zero_clicks"]   = (df["total_clicks"] == 0).astype(int)
    if "num_assessments" in df.columns:
        df["no_assessments"] = (df["num_assessments"] == 0).astype(int)
    if "avg_score" in df.columns:
        df["passed_threshold"] = (df["avg_score"] >= 40).astype(int)
    if "num_of_prev_attempts" in df.columns:
        df["is_repeat"] = (df["num_of_prev_attempts"] > 0).astype(int)
    if "assessment_completion_ratio" in df.columns:
        df["submitted_all"] = (df["assessment_completion_ratio"] >= 1.0).astype(int)

    # IMD ordinal
    if "imd_band" in df.columns:
        imd = {"0-10%":1,"10-20":2,"10-20%":2,"20-30%":3,"30-40%":4,
               "40-50%":5,"50-60%":6,"60-70%":7,"70-80%":8,"80-90%":9,"90-100%":10}
        df["imd_numeric"] = df["imd_band"].map(imd).fillna(5)

    # weekly pattern features
    wk = [f"week{w}_clicks" for w in range(1, 13) if f"week{w}_clicks" in df.columns]
    if len(wk) >= 4:
        df["peak_week_clicks"]  = df[wk].max(axis=1)
        df["active_weeks_count"]= (df[wk] > 0).sum(axis=1)
        h1 = [f"week{w}_clicks" for w in range(1, 7)  if f"week{w}_clicks" in df.columns]
        h2 = [f"week{w}_clicks" for w in range(7, 13) if f"week{w}_clicks" in df.columns]
        if h1 and h2:
            df["h2_vs_h1"] = (df[h2].sum(axis=1) /
                               (df[h1].sum(axis=1) + 1))

    return df


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and 4-class target."""
    y = df["final_result"]
    X = df.drop(columns=["final_result"], errors="ignore")
    # also drop any remaining always-drop cols that slipped through
    X = X.drop(columns=[c for c in DROP_ALWAYS if c in X.columns], errors="ignore")
    X = X.dropna(axis=1, how="all")
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    parts = []
    if num_cols:
        parts.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False, max_categories=30)),
        ]), cat_cols))
    return ColumnTransformer(parts, remainder="drop")


def preprocess_Xy(X: pd.DataFrame
                  ) -> Tuple[np.ndarray, List[str], List[str], ColumnTransformer]:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    pre = build_preprocessor(num_cols, cat_cols)
    X_t = pre.fit_transform(X)
    return X_t, num_cols, cat_cols, pre


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def make_lgb(cw=None, seed=SEED) -> "LGBMClassifier":
    return LGBMClassifier(
        n_estimators=500, num_leaves=63, learning_rate=0.05,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.5,
        class_weight=cw, random_state=seed, n_jobs=-1, verbosity=-1,
    )

def make_xgb(seed=SEED) -> "XGBClassifier":
    return XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist",
    )

def make_cat(cw=None, seed=SEED) -> "CatBoostClassifier":
    kw = dict(iterations=500, depth=7, learning_rate=0.05,
              l2_leaf_reg=3, verbose=0, random_state=seed)
    if cw == "balanced":
        kw["auto_class_weights"] = "Balanced"
    return CatBoostClassifier(**kw)

def best_available(cw=None, seed=SEED):
    """Return the best available classifier."""
    if _LGB: return make_lgb(cw=cw, seed=seed)
    if _XGB: return make_xgb(seed=seed)
    if _CAT: return make_cat(cw=cw, seed=seed)
    return RandomForestClassifier(n_estimators=300, class_weight=cw,
                                   random_state=seed, n_jobs=-1)

# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_smote(X: np.ndarray, y: np.ndarray,
                seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    if not _SMOTE:
        logger.warning("imbalanced-learn not installed — SMOTE skipped.")
        return X, y
    min_count = np.bincount(y).min()
    k = min(5, min_count - 1)
    if k < 1:
        return X, y
    sm = SMOTE(random_state=seed, k_neighbors=k)
    return sm.fit_resample(X, y)


def apply_ctgan(X_df: pd.DataFrame, y_s: pd.Series,
                target_col: str = "__label__",
                epochs: int = 100, seed: int = SEED,
                augment_minority_to: Optional[int] = None
                ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic minority-class rows with CTGAN.
    Only generates for the minority class; majority class untouched.
    """
    if not _CTGAN:
        logger.warning("ctgan not installed — CTGAN skipped.")
        return X_df, y_s

    combined = X_df.copy()
    combined[target_col] = y_s.values

    counts = y_s.value_counts()
    majority_class = counts.index[0]
    minority_class = counts.index[-1]
    n_maj = counts[majority_class]
    n_min = counts[minority_class]
    target_n = augment_minority_to if augment_minority_to else n_maj

    if n_min >= target_n:
        return X_df, y_s

    logger.info("  CTGAN: augmenting class '%s' from %d → %d rows",
                minority_class, n_min, target_n)

    minority_df = combined[combined[target_col] == minority_class].copy()

    # CTGAN only handles numeric + low-card categorical — identify discrete cols
    discrete_cols = [c for c in minority_df.columns
                     if minority_df[c].dtype == object or
                     minority_df[c].nunique() <= 20]

    ctgan = CTGAN(epochs=epochs, verbose=False)
    ctgan.fit(minority_df, discrete_columns=discrete_cols)

    n_generate = target_n - n_min
    synthetic = ctgan.sample(n_generate)
    synthetic[target_col] = minority_class

    augmented = pd.concat([combined, synthetic], ignore_index=True)
    y_aug = augmented.pop(target_col)
    return augmented, y_aug


# ══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class HierarchicalClassifier:
    """
    3-model probabilistic hierarchical classifier.

    Training:
        fit(X_train_raw, y_train)   — raw DataFrame + Series of 4-class labels

    Prediction:
        predict_proba_4class(X_raw) → shape (n, 4) in order [Distinction, Fail, Pass, Withdrawn]
        predict(X_raw)              → array of label strings
    """

    CLASS_ORDER = ["Distinction", "Fail", "Pass", "Withdrawn"]

    def __init__(self,
                 augment: str = "none",   # "none" | "weights" | "smote" | "ctgan"
                 ctgan_epochs: int = 100,
                 seed: int = SEED):
        self.augment     = augment
        self.ctgan_epochs= ctgan_epochs
        self.seed        = seed

        # will be set during fit
        self.pre1: ColumnTransformer = None
        self.pre2a: ColumnTransformer= None
        self.pre2b: ColumnTransformer= None
        self.m1   = None   # AtRisk vs Success
        self.m2a  = None   # Fail vs Withdrawn
        self.m2b  = None   # Pass vs Distinction
        self.le1  = LabelEncoder()   # AtRisk=0, Success=1
        self.le2a = LabelEncoder()   # Fail=?, Withdrawn=?
        self.le2b = LabelEncoder()   # Distinction=?, Pass=?

    # ── internal helpers ──────────────────────────────────────────────────────

    def _fit_branch(self, X_raw: pd.DataFrame, y_raw: pd.Series,
                    cw_label: str, branch_name: str
                    ) -> Tuple[ColumnTransformer, LabelEncoder, object]:
        """Fit preprocessor + model for one binary branch."""
        le = LabelEncoder()
        y  = le.fit_transform(y_raw.astype(str))

        num_cols = X_raw.select_dtypes(include="number").columns.tolist()
        cat_cols = X_raw.select_dtypes(include=["object","category"]).columns.tolist()
        pre = build_preprocessor(num_cols, cat_cols)
        X_t = pre.fit_transform(X_raw)

        cw = None
        if self.augment == "weights":
            cw = "balanced"

        if self.augment == "smote":
            X_t, y = apply_smote(X_t, y, seed=self.seed)
        elif self.augment == "ctgan":
            # CTGAN on the raw DataFrame before preprocessing
            aug_df, aug_y_raw = apply_ctgan(
                X_raw, y_raw,
                epochs=self.ctgan_epochs, seed=self.seed)
            aug_y = le.transform(aug_y_raw.astype(str))
            # refit preprocessor on augmented data
            pre = build_preprocessor(num_cols, cat_cols)
            X_t = pre.fit_transform(aug_df)
            y   = aug_y

        clf = best_available(cw=cw, seed=self.seed)
        clf.fit(X_t, y)
        logger.info("  [%s] trained on %d rows  classes=%s  aug=%s",
                    branch_name, len(y), list(le.classes_), self.augment)
        return pre, le, clf

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y4: pd.Series) -> "HierarchicalClassifier":
        t0 = time.time()

        # ── Model 1: binary AtRisk vs Success ─────────────────────────────
        y_bin = y4.map(BINARY_MAP)
        self.pre1, self.le1, self.m1 = self._fit_branch(
            X, y_bin, "balanced", "Model1-Binary")

        # ── Model 2A: Fail vs Withdrawn ───────────────────────────────────
        mask_ar = y4.isin(ATRISK_CLASSES)
        self.pre2a, self.le2a, self.m2a = self._fit_branch(
            X.loc[mask_ar], y4.loc[mask_ar], "balanced", "Model2A-AtRisk")

        # ── Model 2B: Pass vs Distinction ─────────────────────────────────
        mask_s = y4.isin(SUCCESS_CLASSES)
        self.pre2b, self.le2b, self.m2b = self._fit_branch(
            X.loc[mask_s], y4.loc[mask_s], "balanced", "Model2B-Success")

        logger.info("  Hierarchical fit done in %.1fs", time.time() - t0)
        return self

    def predict_proba_4class(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns (n, 4) probability matrix.
        Column order = ["Distinction", "Fail", "Pass", "Withdrawn"]
        """
        X1 = self.pre1.transform(X)
        p1 = self.m1.predict_proba(X1)               # (n, 2): [AtRisk, Success]

        classes1 = list(self.le1.classes_)
        p_atrisk  = p1[:, classes1.index("AtRisk")]
        p_success = p1[:, classes1.index("Success")]

        X2a = self.pre2a.transform(X)
        p2a = self.m2a.predict_proba(X2a)            # (n, 2): [Fail, Withdrawn]
        classes2a = list(self.le2a.classes_)
        p_fail      = p_atrisk * p2a[:, classes2a.index("Fail")]
        p_withdrawn = p_atrisk * p2a[:, classes2a.index("Withdrawn")]

        X2b = self.pre2b.transform(X)
        p2b = self.m2b.predict_proba(X2b)            # (n, 2): [Distinction, Pass]
        classes2b = list(self.le2b.classes_)
        p_distinction = p_success * p2b[:, classes2b.index("Distinction")]
        p_pass        = p_success * p2b[:, classes2b.index("Pass")]

        # stack in canonical order: Distinction, Fail, Pass, Withdrawn
        proba = np.column_stack([p_distinction, p_fail, p_pass, p_withdrawn])
        # renormalise rows (small floating point drift)
        row_sums = proba.sum(axis=1, keepdims=True)
        proba = proba / np.where(row_sums > 0, row_sums, 1)
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba_4class(X)
        idx   = np.argmax(proba, axis=1)
        return np.array(self.CLASS_ORDER)[idx]

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

FOUR_LABEL_ORDER = ["Distinction", "Fail", "Pass", "Withdrawn"]

def evaluate_4class(y_true, y_pred, y_proba: Optional[np.ndarray],
                    label: str) -> Dict:
    r = {
        "experiment":       label,
        "accuracy":         accuracy_score(y_true, y_pred),
        "f1_macro":         f1_score(y_true, y_pred, average="macro",   zero_division=0),
        "f1_weighted":      f1_score(y_true, y_pred, average="weighted",zero_division=0),
        "balanced_acc":     balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa":      cohen_kappa_score(y_true, y_pred),
        "mcc":              matthews_corrcoef(y_true, y_pred),
    }
    # per-class recall
    cm = confusion_matrix(y_true, y_pred, labels=FOUR_LABEL_ORDER)
    for i, cls in enumerate(FOUR_LABEL_ORDER):
        row_sum = cm[i].sum()
        r[f"recall_{cls.lower()}"] = cm[i, i] / row_sum if row_sum > 0 else 0.0

    # ROC-AUC
    if y_proba is not None:
        try:
            le_tmp = LabelEncoder().fit(FOUR_LABEL_ORDER)
            y_enc  = le_tmp.transform(y_true)
            y_bin  = label_binarize(y_enc, classes=range(4))
            r["roc_auc"] = roc_auc_score(y_bin, y_proba,
                                          average="macro", multi_class="ovr")
        except Exception:
            r["roc_auc"] = float("nan")
    else:
        r["roc_auc"] = float("nan")
    return r


def print_experiment(r: Dict, y_true, y_pred) -> None:
    print(f"\n{'─'*65}")
    print(f"  {r['experiment']}")
    print(f"{'─'*65}")
    print(f"  Accuracy        : {r['accuracy']:.4f}")
    print(f"  Macro F1        : {r['f1_macro']:.4f}")
    print(f"  Weighted F1     : {r['f1_weighted']:.4f}")
    print(f"  Balanced Acc    : {r['balanced_acc']:.4f}")
    print(f"  Cohen Kappa     : {r['cohen_kappa']:.4f}")
    print(f"  MCC             : {r['mcc']:.4f}")
    print(f"  ROC-AUC (macro) : {r['roc_auc']:.4f}")
    print(f"\n  Per-class recall:")
    for cls in FOUR_LABEL_ORDER:
        print(f"    {cls:<12} : {r[f'recall_{cls.lower()}']:.4f}")
    print(f"\n  Classification report:")
    print("  " + classification_report(
        y_true, y_pred, labels=FOUR_LABEL_ORDER, zero_division=0
    ).replace("\n", "\n  "))


def print_comparison_table(results: List[Dict]) -> None:
    print("\n" + "=" * 115)
    print(f"{'EXPERIMENT':<42} {'ACC':>6} {'F1-MAC':>7} {'BACC':>6} "
          f"{'KAPPA':>6} {'ROC':>6}  "
          f"{'Dist':>6} {'Fail':>6} {'Pass':>6} {'With':>6}")
    print("=" * 115)
    baseline = None
    for r in results:
        if baseline is None:
            baseline = r["accuracy"]
        delta = r["accuracy"] - baseline
        star  = " ★" if r["accuracy"] >= 0.90 else (
                " ↑" if r["accuracy"] >= 0.80 else "")
        sign  = f" (+{delta:.4f})" if delta > 0 else (
                f" ({delta:.4f})" if delta < 0 else "")
        print(
            f"{r['experiment']:<42} "
            f"{r['accuracy']:>6.4f} {r['f1_macro']:>7.4f} "
            f"{r['balanced_acc']:>6.4f} {r['cohen_kappa']:>6.4f} "
            f"{r['roc_auc']:>6.4f}  "
            f"{r['recall_distinction']:>6.4f} {r['recall_fail']:>6.4f} "
            f"{r['recall_pass']:>6.4f} {r['recall_withdrawn']:>6.4f}"
            f"{sign}{star}"
        )
    print("=" * 115)
    best = max(results, key=lambda x: x["accuracy"])
    print(f"\n  Best: {best['experiment']}  →  acc={best['accuracy']:.4f}  "
          f"f1_macro={best['f1_macro']:.4f}  roc={best['roc_auc']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# DIRECT 4-CLASS BASELINE (Experiment 1)
# ══════════════════════════════════════════════════════════════════════════════

def run_direct_baseline(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame,  y_test: pd.Series) -> Dict:
    """Experiment 1: flat 4-class XGBoost / LightGBM / CatBoost."""
    logger.info("\n[Exp 1] Direct 4-class baseline")

    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_te = le.transform(y_test)

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object","category"]).columns.tolist()
    pre = build_preprocessor(num_cols, cat_cols)
    X_tr_t = pre.fit_transform(X_train)
    X_te_t = pre.transform(X_test)

    # use best available model
    if _LGB:
        clf = make_lgb(cw="balanced")
    elif _XGB:
        clf = make_xgb()
    elif _CAT:
        clf = make_cat(cw="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=SEED, n_jobs=-1)
    clf.fit(X_tr_t, y_tr)
    preds  = le.inverse_transform(clf.predict(X_te_t))
    proba  = clf.predict_proba(X_te_t)

    # re-order proba columns to canonical order
    proba_ordered = _reorder_proba(proba, le.classes_, FOUR_LABEL_ORDER)

    r = evaluate_4class(y_test.values, preds, proba_ordered, "Exp1 Direct 4-class (baseline)")
    print_experiment(r, y_test.values, preds)
    return r


def _reorder_proba(proba: np.ndarray, current_order, target_order) -> np.ndarray:
    """Reorder columns of a probability matrix to match target_order."""
    current = list(current_order)
    out = np.zeros((proba.shape[0], len(target_order)))
    for i, cls in enumerate(target_order):
        if cls in current:
            out[:, i] = proba[:, current.index(cls)]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL EXPERIMENTS 2-5
# ══════════════════════════════════════════════════════════════════════════════

def run_hierarchical(X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame,  y_test: pd.Series,
                     augment: str, exp_label: str,
                     ctgan_epochs: int = 100) -> Dict:
    logger.info("\n[%s] augment=%s", exp_label, augment)

    hc = HierarchicalClassifier(augment=augment,
                                 ctgan_epochs=ctgan_epochs,
                                 seed=SEED)
    hc.fit(X_train, y_train)

    preds = hc.predict(X_test)
    proba = hc.predict_proba_4class(X_test)   # (n,4) in canonical order

    r = evaluate_4class(y_test.values, preds, proba, exp_label)
    print_experiment(r, y_test.values, preds)

    # also show branch-level accuracy
    y_bin_true = y_test.map(BINARY_MAP).values
    X_te_t = hc.pre1.transform(X_test)
    p1     = hc.m1.predict_proba(X_te_t)
    p1_cls = list(hc.le1.classes_)
    p1_pred= np.array(p1_cls)[np.argmax(p1, axis=1)]
    branch_acc = accuracy_score(y_bin_true, p1_pred)
    logger.info("  Branch Model1 accuracy: %.4f", branch_acc)

    # Attach the fitted classifier so main() can export stage predictions
    r["_hc"] = hc
    r["_preds"] = preds
    return r

# ══════════════════════════════════════════════════════════════════════════════
# CV-BASED BRANCH ACCURACY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def branch_cv_analysis(X: pd.DataFrame, y4: pd.Series, n_splits: int = 5):
    """Report per-branch CV accuracy to understand the theoretical ceiling."""
    print(f"\n{'='*65}")
    print("  BRANCH ACCURACY ANALYSIS (5-fold CV)")
    print(f"{'='*65}")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Branch 1: binary
    y_bin = y4.map(BINARY_MAP)
    le1   = LabelEncoder()
    y1    = le1.fit_transform(y_bin)
    num_c = X.select_dtypes(include="number").columns.tolist()
    cat_c = X.select_dtypes(include=["object","category"]).columns.tolist()
    pre1  = build_preprocessor(num_c, cat_c)
    X_t   = pre1.fit_transform(X)
    clf1  = make_lgb() if _LGB else make_xgb()
    s1    = cross_val_score(clf1, X_t, y1, cv=cv, scoring="accuracy", n_jobs=1)
    print(f"\n  Model 1  (AtRisk vs Success)   : {s1.mean():.4f} ± {s1.std():.4f}")

    # Branch 2A: Fail vs Withdrawn
    mask_ar = y4.isin(ATRISK_CLASSES)
    X_ar  = X.loc[mask_ar]
    y_ar  = LabelEncoder().fit_transform(y4.loc[mask_ar])
    pre2a = build_preprocessor(num_c, cat_c)
    X_ar_t= pre2a.fit_transform(X_ar)
    clf2a = make_lgb() if _LGB else make_xgb()
    s2a   = cross_val_score(clf2a, X_ar_t, y_ar, cv=cv, scoring="accuracy", n_jobs=1)
    print(f"  Model 2A (Fail vs Withdrawn)   : {s2a.mean():.4f} ± {s2a.std():.4f}")

    # Branch 2B: Pass vs Distinction
    mask_s = y4.isin(SUCCESS_CLASSES)
    X_s    = X.loc[mask_s]
    y_s    = LabelEncoder().fit_transform(y4.loc[mask_s])
    pre2b  = build_preprocessor(num_c, cat_c)
    X_s_t  = pre2b.fit_transform(X_s)
    clf2b  = make_lgb() if _LGB else make_xgb()
    s2b    = cross_val_score(clf2b, X_s_t, y_s, cv=cv, scoring="accuracy", n_jobs=1)
    print(f"  Model 2B (Pass vs Distinction) : {s2b.mean():.4f} ± {s2b.std():.4f}")

    # theoretical ceiling
    dist = y4.value_counts(normalize=True)
    p_atrisk  = dist.get("Fail",  0) + dist.get("Withdrawn", 0)
    p_success = dist.get("Pass",  0) + dist.get("Distinction", 0)
    ceiling_atrisk  = s1.mean() * s2a.mean()
    ceiling_success = s1.mean() * s2b.mean()
    weighted_ceiling = p_atrisk * ceiling_atrisk + p_success * ceiling_success
    print(f"\n  Theoretical path-wise ceiling:")
    print(f"    At-Risk  path : {s1.mean():.4f} × {s2a.mean():.4f} = {ceiling_atrisk:.4f}")
    print(f"    Success  path : {s1.mean():.4f} × {s2b.mean():.4f} = {ceiling_success:.4f}")
    print(f"    Weighted ceiling (by class proportion) : {weighted_ceiling:.4f}")
    print(f"    (Direct 4-class baseline               :  ~0.7595)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical probabilistic classification for OULAD")
    parser.add_argument("--skip-ctgan",  action="store_true",
                        help="Skip Experiment 5 (CTGAN) — much slower")
    parser.add_argument("--skip-smote",  action="store_true",
                        help="Skip Experiment 4 (SMOTE)")
    parser.add_argument("--ctgan-epochs",type=int, default=100,
                        help="CTGAN training epochs (default 100)")
    parser.add_argument("--test-size",   type=float, default=0.2)
    parser.add_argument("--output-dir",  default="results")
    parser.add_argument("--save-graphs", action="store_true", default=True,
                        help="Generate publication figures and LaTeX table (default: on)")
    parser.add_argument("--no-graphs",   dest="save_graphs", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & prepare ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  HIERARCHICAL PROBABILISTIC CLASSIFICATION PIPELINE")
    print("="*65)

    df_raw = load_oulad()
    df     = engineer(df_raw)
    X, y4  = get_X_y(df)

    print(f"\n  Dataset  : {len(X)} rows  {X.shape[1]} features")
    print(f"  Classes  : {y4.value_counts().to_dict()}")
    print(f"  Test set : {args.test_size:.0%} of real original data (never augmented)")

    # ── CRITICAL: split BEFORE any augmentation ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y4, test_size=args.test_size, random_state=SEED, stratify=y4
    )
    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"  Test class distribution: {y_test.value_counts().to_dict()}")

    # ── Branch CV analysis (theoretical ceiling) ──────────────────────────────
    branch_cv_analysis(X_train, y_train)

    all_results: List[Dict] = []

    # ── Experiment 1: Direct 4-class baseline ─────────────────────────────────
    print(f"\n{'='*65}")
    print("  EXPERIMENT 1 — Direct 4-class baseline")
    print(f"{'='*65}")
    r1 = run_direct_baseline(X_train, y_train, X_test, y_test)
    all_results.append(r1)

    # ── Experiment 2: Hierarchical, no augmentation ───────────────────────────
    print(f"\n{'='*65}")
    print("  EXPERIMENT 2 — Hierarchical (no augmentation)")
    print(f"{'='*65}")
    r2 = run_hierarchical(X_train, y_train, X_test, y_test,
                          augment="none",
                          exp_label="Exp2 Hierarchical (no aug)")
    all_results.append(r2)

    # ── Experiment 3: Hierarchical + class weights ────────────────────────────
    print(f"\n{'='*65}")
    print("  EXPERIMENT 3 — Hierarchical + class weights")
    print(f"{'='*65}")
    r3 = run_hierarchical(X_train, y_train, X_test, y_test,
                          augment="weights",
                          exp_label="Exp3 Hierarchical + weights")
    all_results.append(r3)

    # ── Experiment 4: Hierarchical + SMOTE ───────────────────────────────────
    if not args.skip_smote:
        print(f"\n{'='*65}")
        print("  EXPERIMENT 4 — Hierarchical + SMOTE")
        print(f"{'='*65}")
        if not _SMOTE:
            print("  imbalanced-learn not installed.")
            print("  Install: pip install imbalanced-learn")
        else:
            r4 = run_hierarchical(X_train, y_train, X_test, y_test,
                                  augment="smote",
                                  exp_label="Exp4 Hierarchical + SMOTE")
            all_results.append(r4)
    else:
        print("\n  Exp 4 (SMOTE) skipped by --skip-smote")

    # ── Experiment 5: Hierarchical + CTGAN ───────────────────────────────────
    if not args.skip_ctgan:
        print(f"\n{'='*65}")
        print("  EXPERIMENT 5 — Hierarchical + CTGAN")
        print(f"{'='*65}")
        if not _CTGAN:
            print("  ctgan not installed.")
            print("  Install: pip install ctgan")
        else:
            r5 = run_hierarchical(X_train, y_train, X_test, y_test,
                                  augment="ctgan",
                                  exp_label="Exp5 Hierarchical + CTGAN",
                                  ctgan_epochs=args.ctgan_epochs)
            all_results.append(r5)
    else:
        print("\n  Exp 5 (CTGAN) skipped by --skip-ctgan")

    # ── Final comparison table ────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*65}")
    print_comparison_table(all_results)

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame([{k: v for k, v in r.items()
                                 if not k.startswith("_")}
                                for r in all_results])
    out_path   = out_dir / "hierarchical_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")

    # ── Export per-stage prediction CSVs for generate_all_figures.py ─────────
    # Figures 14-17 require: stage1_predictions.csv, stage2a_predictions.csv,
    # stage2b_predictions.csv, hierarchical_final_predictions.csv
    try:
        hier_results = [r for r in all_results if "_hc" in r]
        if hier_results:
            best_r = max(hier_results, key=lambda x: x["accuracy"])
            hc: HierarchicalClassifier = best_r["_hc"]

            # ── Stage 1: binary AtRisk vs Success ────────────────────────────
            X_te1 = hc.pre1.transform(X_test)
            p1    = hc.m1.predict_proba(X_te1)
            p1_cls = list(hc.le1.classes_)
            stage1_pred = np.array(p1_cls)[np.argmax(p1, axis=1)]
            y_bin_true  = y_test.map(BINARY_MAP).values
            pd.DataFrame({"y_true": y_bin_true,
                          "y_pred": stage1_pred}).to_csv(
                out_dir / "stage1_predictions.csv", index=False)
            print(f"  Saved → {out_dir / 'stage1_predictions.csv'}")

            # ── Stage 2A: Fail vs Withdrawn (AtRisk subset) ──────────────────
            mask_ar = y_test.isin(ATRISK_CLASSES)
            X_ar   = X_test.loc[mask_ar]
            y_ar   = y_test.loc[mask_ar]
            X_te2a = hc.pre2a.transform(X_ar)
            p2a    = hc.m2a.predict_proba(X_te2a)
            p2a_cls = list(hc.le2a.classes_)
            stage2a_pred = np.array(p2a_cls)[np.argmax(p2a, axis=1)]
            pd.DataFrame({"y_true": y_ar.values,
                          "y_pred": stage2a_pred}).to_csv(
                out_dir / "stage2a_predictions.csv", index=False)
            print(f"  Saved → {out_dir / 'stage2a_predictions.csv'}")

            # ── Stage 2B: Pass vs Distinction (Success subset) ───────────────
            mask_s = y_test.isin(SUCCESS_CLASSES)
            X_s    = X_test.loc[mask_s]
            y_s    = y_test.loc[mask_s]
            X_te2b = hc.pre2b.transform(X_s)
            p2b    = hc.m2b.predict_proba(X_te2b)
            p2b_cls = list(hc.le2b.classes_)
            stage2b_pred = np.array(p2b_cls)[np.argmax(p2b, axis=1)]
            pd.DataFrame({"y_true": y_s.values,
                          "y_pred": stage2b_pred}).to_csv(
                out_dir / "stage2b_predictions.csv", index=False)
            print(f"  Saved → {out_dir / 'stage2b_predictions.csv'}")

            # ── Final hierarchical predictions (all 4 classes) ───────────────
            final_preds = best_r["_preds"]
            pd.DataFrame({"y_true": y_test.values,
                          "y_pred": final_preds}).to_csv(
                out_dir / "hierarchical_final_predictions.csv", index=False)
            print(f"  Saved → {out_dir / 'hierarchical_final_predictions.csv'}")

    except Exception as _e:
        print(f"  [WARN] Stage prediction export failed: {_e}")

    # ── Export stage predictions for graph generation ─────────────────────────
    # These CSV files are consumed by generate_all_figures.py for figures 13-17.
    try:
        # Reload the best hierarchical model predictions (Exp 2 or best)
        best_hier = [r for r in all_results if "Hierarchical" in r.get("experiment","")]
        if best_hier:
            best_r = max(best_hier, key=lambda x: x["accuracy"])
            # Save stage performance summary
            stage_df = pd.DataFrame([
                {"stage": "Stage 1 (AtRisk/Success)",
                 "accuracy": r1.get("accuracy",0),
                 "f1_macro": r1.get("f1_macro",0),
                 "roc_auc":  r1.get("roc_auc", float("nan"))},
                {"stage": "Direct 4-class",
                 "accuracy": r1.get("accuracy",0) if all_results else 0,
                 "f1_macro": all_results[0].get("f1_macro",0) if all_results else 0,
                 "roc_auc":  all_results[0].get("roc_auc", float("nan")) if all_results else 0},
                {"stage": "Best Hierarchical",
                 "accuracy": best_r.get("accuracy",0),
                 "f1_macro": best_r.get("f1_macro",0),
                 "roc_auc":  best_r.get("roc_auc", float("nan"))},
            ])
            stage_df.to_csv(out_dir / "stage_performance.csv", index=False)
            print(f"  Stage performance saved -> {out_dir / 'stage_performance.csv'}")
    except Exception as _e:
        print(f"  Stage export skipped: {_e}")

    # ── Research summary ──────────────────────────────────────────────────────
    best = max(all_results, key=lambda x: x["accuracy"])
    base = all_results[0]["accuracy"]
    gain = best["accuracy"] - base
    print(f"\n{'='*65}")
    print("  RESEARCH SUMMARY")
    print(f"{'='*65}")
    print(f"  Direct 4-class baseline  : {base:.4f}")
    print(f"  Best hierarchical result : {best['accuracy']:.4f}  ({best['experiment']})")
    print(f"  Accuracy gain            : +{gain:.4f}  ({gain*100:.2f} percentage points)")
    print(f"  Macro F1 (best)          : {best['f1_macro']:.4f}")
    print(f"  ROC-AUC (best)           : {best['roc_auc']:.4f}")
    print()

    # ── Graphs + final table ──────────────────────────────────────────────────
    if args.save_graphs:
        print(f"\n{'='*65}")
        print("  GENERATING FIGURES AND PAPER TABLE")
        print(f"{'='*65}")
        generate_final_graphs(results_df, out_dir)


def generate_final_graphs(results_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Generate publication-ready graphs and a LaTeX/CSV table.
    Saves to out_dir/figures/.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        print("  matplotlib not available — skipping graphs.")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    FOUR_CLS = ["Distinction", "Fail", "Pass", "Withdrawn"]
    colors   = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F"]

    # ── 1. Accuracy + F1 bar chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x       = range(len(results_df))
    labels  = [r["experiment"].replace("Exp", "E").replace(" (no aug)", "")
                              .replace("Hierarchical", "Hier.") for _, r in results_df.iterrows()]
    acc     = results_df["accuracy"].values
    f1      = results_df["f1_macro"].values
    w       = 0.35
    bars1   = ax.bar([i - w/2 for i in x], acc, width=w, label="Accuracy",  color="#4878D0", alpha=0.85)
    bars2   = ax.bar([i + w/2 for i in x], f1,  width=w, label="Macro F1", color="#EE854A", alpha=0.85)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0.60, 0.85); ax.set_ylabel("Score"); ax.set_title("Hierarchical vs Direct Baseline")
    ax.legend(); ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.axhline(acc[0], ls="--", color="grey", lw=0.8, label="Baseline")
    plt.tight_layout()
    p1 = fig_dir / "acc_f1_comparison.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p1}")

    # ── 2. Per-class recall comparison ─────────────────────────────────────
    recall_cols = [f"recall_{c.lower()}" for c in FOUR_CLS]
    recall_cols = [c for c in recall_cols if c in results_df.columns]
    if recall_cols:
        fig, ax = plt.subplots(figsize=(11, 5))
        n_exp   = len(results_df)
        n_cls   = len(recall_cols)
        bw      = 0.7 / n_exp
        for ei, (_, row) in enumerate(results_df.iterrows()):
            xs    = [i + (ei - n_exp/2 + 0.5) * bw for i in range(n_cls)]
            vals  = [row.get(c, 0) for c in recall_cols]
            label = row["experiment"].replace("Exp", "E").replace(" (no aug)", "")[:25]
            ax.bar(xs, vals, width=bw*0.9, label=label, alpha=0.8)
        ax.set_xticks(range(n_cls))
        ax.set_xticklabels([c.replace("recall_","").capitalize() for c in recall_cols], fontsize=10)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Recall"); ax.set_title("Per-Class Recall by Experiment")
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        p2 = fig_dir / "per_class_recall.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p2}")

    # ── 3. ROC-AUC comparison ──────────────────────────────────────────────
    if "roc_auc" in results_df.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        roc_vals = results_df["roc_auc"].fillna(0)
        bars = ax.barh(labels[::-1], roc_vals.values[::-1], color="#6ACC65", alpha=0.85)
        for bar, val in zip(bars, roc_vals.values[::-1]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlim(0.85, 1.0); ax.set_xlabel("ROC-AUC (macro OvR)")
        ax.set_title("ROC-AUC by Experiment")
        plt.tight_layout()
        p3 = fig_dir / "roc_auc_comparison.png"
        fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p3}")

    # ── 4. Full paper table (CSV + LaTeX snippet) ──────────────────────────
    table_cols = (["experiment","accuracy","f1_macro","f1_weighted","balanced_acc",
                   "cohen_kappa","roc_auc"] + recall_cols)
    table_cols = [c for c in table_cols if c in results_df.columns]
    table_df   = results_df[table_cols].copy()
    table_df.columns = [c.replace("recall_","recall_").replace("_"," ").title()
                        for c in table_df.columns]
    csv_path = out_dir / "final_results_table.csv"
    table_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved: {csv_path}")

    # LaTeX snippet
    tex_path = out_dir / "final_results_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by hierarchical_pipeline.py\n")
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Hierarchical vs Direct 4-class Baseline — OULAD V2}\n")
        f.write("\\label{tab:hierarchical_results}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("Experiment & Acc & Macro F1 & Bal.Acc & Kappa & ROC-AUC \\\\\n\\hline\n")
        for _, row in results_df.iterrows():
            exp  = row["experiment"].replace("Exp","E").replace("_"," ")[:30]
            acc  = row.get("accuracy",0); f1 = row.get("f1_macro",0)
            ba   = row.get("balanced_acc",0); kap = row.get("cohen_kappa",0)
            roc  = row.get("roc_auc", float("nan"))
            roc_s = f"{roc:.4f}" if not (roc != roc) else "---"
            f.write(f"{exp} & {acc:.4f} & {f1:.4f} & {ba:.4f} & {kap:.4f} & {roc_s} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Saved: {tex_path}")
    print(f"\n  All figures and tables → {fig_dir}")


if __name__ == "__main__":
    main()
