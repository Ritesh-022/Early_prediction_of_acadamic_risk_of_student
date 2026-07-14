#!/usr/bin/env python3
"""
High-Accuracy Student Performance Pipeline
==========================================
Targets 90-95% accuracy across all datasets by using:
  1. Smart target binarisation  (Pass/Distinction vs Fail/Withdrawn)
  2. Deep OULAD feature engineering
  3. XGBoost / LightGBM / CatBoost + Optuna tuning
  4. Stacking ensemble
  5. Cross-dataset evidence pooling
  6. SHAP explainability

Run:
    python high_accuracy_pipeline.py
    python high_accuracy_pipeline.py --tune           # slower, higher accuracy
    python high_accuracy_pipeline.py --dataset oulad  # OULAD only
"""
from __future__ import annotations
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── optional imports ──────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    XGBClassifier = None; _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    LGBMClassifier = None; _HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    CatBoostClassifier = None; _HAS_CAT = False

try:
    import shap as _shap; _HAS_SHAP = True
except ImportError:
    _shap = None; _HAS_SHAP = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    optuna = None; _HAS_OPTUNA = False

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
)
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import joblib

# ── FIX: force UTF-8 stdout/stderr to prevent cp1252 UnicodeEncodeError ──────
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 — TEMPORAL LEAKAGE AUDIT
# ══════════════════════════════════════════════════════════════════════════════
# The 95.05% OULAD binary result uses oulad_ml_table.csv (full-semester table).
# Every feature in that table covers the ENTIRE course duration, making this a
# FINAL-OUTCOME PREDICTION — NOT an early-warning result.
# Features confirmed as END-OF-COURSE (leakage for early-warning):
#   - avg_score, score_std, assessment_score_trend   → all assessments submitted
#   - assessment_completion_ratio, missed_assessments → whole-course submission rate
#   - score_x_completion                             → derived from above two
#   - inactivity_days, last_ts                       → requires knowing the end date
#   - num_assessments, total_assessments             → whole-course count
#   - assessment_span_days                           → whole-course span
#   - longest_inactive_gap, click_variance           → whole-course temporal range
#   - week_click_sum_1_12                            → sum of ALL 12 weeks
#
# The 95.05% result is VALID as a "final-outcome prediction" (full-semester).
# It CANNOT be reported as a Week-4 or Week-8 early-warning result.
# For early-warning, use oulad_ml_table_week8.csv with --drop-late flag.
#
# Features safe for early-warning (available at Week 8 / Day 56):
#   - gender, age_band, highest_education, imd_band, disability
#   - num_of_prev_attempts, studied_credits, registration_early_days
#   - week1..week8 clicks, clicks_until_week2/4/6/8
#   - first_ts, first_assessment_day (if before cutoff)
#   - avg_score UP TO cutoff (requires week8 ML table)
# ══════════════════════════════════════════════════════════════════════════════
# Features that are only available at end-of-course.
# Including these = valid final-outcome model, NOT early-warning.
LATE_FEATURES = {
    "avg_score", "score_std", "assessment_score_trend",
    "assessment_completion_ratio", "missed_assessments",
    "total_assessments", "num_assessments",
    "score_x_completion", "score_per_credit",
    "inactivity_days", "last_ts", "last_assessment_day",
    "assessment_span_days", "longest_inactive_gap",
    "click_variance", "click_growth_rate",
    "week_click_sum_1_12", "late_submission_count",
    "assessment_score_trend", "avg_score", "score_std",
    "passed_assessments", "submitted_all",
    "weighted_score", "first_assessment_day",
}

# Safe features available at Week 4 / Week 8 cutoff
EARLY_FEATURES = {
    "gender", "region", "highest_education", "imd_band", "age_band",
    "num_of_prev_attempts", "studied_credits", "disability",
    "registration_early_days", "imd_numeric", "is_repeat_student",
    "zero_clicks",
}

# ── constants ──────────────────────────────────────────────────────────────────

OULAD_DROP = {
    "date_unregistration", "date_unreg", "date_unregistered", "weighted_score",
    "active_weeks", "clicks_per_active_week", "assessments_per_week",
    "activity_count", "days_active", "avg_clicks_per_day",
    "registration_delay_category", "id_student", "id_assessment", "id_site",
    "first_ts", "last_ts", "last_assessment_day", "first_assessment_day",
    "code_module", "code_presentation",
}

# Binary label map: "success" vs "risk"
BINARY_MAP = {
    "Pass": "Success",
    "Distinction": "Success",
    "Fail": "AtRisk",
    "Withdrawn": "AtRisk",
    "Graduate": "Success",
    "Dropout": "AtRisk",
    "Enrolled": "AtRisk",  # enrolled = incomplete = at risk
}

# ── reading ────────────────────────────────────────────────────────────────────

def _read(path: Path) -> pd.DataFrame:
    for sep in [None, ",", ";", "\t"]:
        try:
            engine = "python" if sep is None else None
            df = pd.read_csv(path, sep=sep, engine=engine, low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, low_memory=False)

# ── OULAD feature engineering ──────────────────────────────────────────────────

def engineer_oulad(df: pd.DataFrame, binary: bool = False) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in OULAD_DROP if c in df.columns])

    # ── Fill missing activity/assessment columns with 0 (absence = signal) ──
    zero_fill = (
        [f"week{w}_clicks" for w in range(1, 13)]
        + ["total_clicks", "avg_score", "score_std", "num_assessments",
           "assessment_completion_ratio", "missed_assessments",
           "late_submission_count", "assessment_score_trend",
           "click_variance", "click_growth_rate", "longest_inactive_gap",
           "week_click_sum_1_12", "inactivity_days",
           "clicks_until_week2", "clicks_until_week4",
           "clicks_until_week6", "clicks_until_week8"]
        + [c for c in df.columns if "activity_type_" in c]
    )
    for col in zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── Log-transform heavy-tailed click/count features ──
    for col in ["total_clicks"] + [f"week{w}_clicks" for w in range(1, 13)]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # ── Engagement ratios ──
    if "clicks_until_week4" in df.columns and "total_clicks" in df.columns:
        denom = df["total_clicks"].replace(0, np.nan)
        df["early_engagement_ratio"] = (df["clicks_until_week4"] / denom).fillna(0).clip(0, 1)
    if "clicks_until_week8" in df.columns and "total_clicks" in df.columns:
        denom = df["total_clicks"].replace(0, np.nan)
        df["mid_engagement_ratio"] = (df["clicks_until_week8"] / denom).fillna(0).clip(0, 1)

    # ── Score features ──
    if "avg_score" in df.columns and "studied_credits" in df.columns:
        df["score_per_credit"] = (df["avg_score"] / df["studied_credits"].replace(0, np.nan)).fillna(0)
    if "avg_score" in df.columns and "assessment_completion_ratio" in df.columns:
        df["score_x_completion"] = df["avg_score"] * df["assessment_completion_ratio"]
    if "avg_score" in df.columns:
        df["passed_assessments"] = (df["avg_score"] >= 40).astype(int)
    if "assessment_completion_ratio" in df.columns:
        df["submitted_all"] = (df["assessment_completion_ratio"] >= 1.0).astype(int)

    # ── IMD (deprivation) ordinal encoding ──
    if "imd_band" in df.columns:
        imd_order = {
            "0-10%": 1, "10-20": 2, "10-20%": 2, "20-30%": 3,
            "30-40%": 4, "40-50%": 5, "50-60%": 6, "60-70%": 7,
            "70-80%": 8, "80-90%": 9, "90-100%": 10,
        }
        df["imd_numeric"] = df["imd_band"].map(imd_order).fillna(5)

    # ── Flags ──
    if "total_clicks" in df.columns:
        df["zero_clicks"] = (df["total_clicks"] == 0).astype(int)
    if "num_of_prev_attempts" in df.columns:
        df["is_repeat_student"] = (df["num_of_prev_attempts"] > 0).astype(int)
    if "num_assessments" in df.columns:
        df["no_assessments"] = (df["num_assessments"] == 0).astype(int)

    # ── Week-activity pattern features ──
    week_cols = [f"week{w}_clicks" for w in range(1, 13) if f"week{w}_clicks" in df.columns]
    if len(week_cols) >= 4:
        df["peak_week_clicks"] = df[week_cols].max(axis=1)
        df["active_weeks_count"] = (df[week_cols] > 0).sum(axis=1)
        first_half = [f"week{w}_clicks" for w in range(1, 7) if f"week{w}_clicks" in df.columns]
        second_half = [f"week{w}_clicks" for w in range(7, 13) if f"week{w}_clicks" in df.columns]
        if first_half and second_half:
            df["h2_vs_h1_ratio"] = (
                df[second_half].sum(axis=1) /
                (df[first_half].sum(axis=1) + 1)
            )

    # ── Binary target ──
    if binary and "final_result" in df.columns:
        df["final_result"] = df["final_result"].map(BINARY_MAP).fillna("AtRisk")

    return df

# ── Dropout feature engineering ────────────────────────────────────────────────

def engineer_dropout(df: pd.DataFrame, binary: bool = False) -> pd.DataFrame:
    df = df.copy()
    for sem in ["1st", "2nd"]:
        enrolled = f"Curricular units {sem} sem (enrolled)"
        approved = f"Curricular units {sem} sem (approved)"
        grade    = f"Curricular units {sem} sem (grade)"
        evaluations = f"Curricular units {sem} sem (evaluations)"
        if enrolled in df.columns and approved in df.columns:
            denom = pd.to_numeric(df[enrolled], errors="coerce").replace(0, np.nan)
            df[f"approval_rate_{sem}"] = pd.to_numeric(df[approved], errors="coerce") / denom
            df[f"approval_rate_{sem}"] = df[f"approval_rate_{sem}"].fillna(0).clip(0, 1)
        if evaluations in df.columns and enrolled in df.columns:
            denom = pd.to_numeric(df[enrolled], errors="coerce").replace(0, np.nan)
            df[f"eval_rate_{sem}"] = pd.to_numeric(df[evaluations], errors="coerce") / denom
            df[f"eval_rate_{sem}"] = df[f"eval_rate_{sem}"].fillna(0).clip(0, 3)
        if grade in df.columns:
            df[f"log_grade_{sem}"] = np.log1p(pd.to_numeric(df[grade], errors="coerce").fillna(0))
    g1_col = "Curricular units 1st sem (grade)"
    g2_col = "Curricular units 2nd sem (grade)"
    if g1_col in df.columns and g2_col in df.columns:
        g1 = pd.to_numeric(df[g1_col], errors="coerce").fillna(0)
        g2 = pd.to_numeric(df[g2_col], errors="coerce").fillna(0)
        df["grade_improvement"] = g2 - g1
        df["avg_semester_grade"] = (g1 + g2) / 2
        df["grade_consistency"] = np.abs(g2 - g1)
        df["both_passing"] = ((g1 > 10) & (g2 > 10)).astype(int)
    if "Debtor" in df.columns and "Tuition fees up to date" in df.columns:
        df["financial_risk"] = (
            (pd.to_numeric(df["Debtor"], errors="coerce").fillna(0) == 1) |
            (pd.to_numeric(df["Tuition fees up to date"], errors="coerce").fillna(1) == 0)
        ).astype(int)
    if "Age at enrollment" in df.columns:
        df["age_numeric"] = pd.to_numeric(df["Age at enrollment"], errors="coerce").fillna(20)
        df["is_mature_student"] = (df["age_numeric"] > 23).astype(int)
    if "Admission grade" in df.columns:
        df["log_admission_grade"] = np.log1p(pd.to_numeric(df["Admission grade"], errors="coerce").fillna(0))
    if binary and "final_result" in df.columns:
        df["final_result"] = df["final_result"].map(BINARY_MAP).fillna("AtRisk")
    return df


# ── xAPI feature engineering ───────────────────────────────────────────────────

def engineer_xapi(df: pd.DataFrame, binary: bool = False) -> pd.DataFrame:
    df = df.copy()
    num_cols = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    present = [c for c in num_cols if c in df.columns]
    if present:
        df["total_engagement"] = df[present].sum(axis=1)
        df["log_total_engagement"] = np.log1p(df["total_engagement"])
        df["engagement_diversity"] = (df[present] > 0).sum(axis=1)
        df["passive_ratio"] = (
            df[["AnnouncementsView", "VisITedResources"]].sum(axis=1)
            / (df["total_engagement"] + 1)
        )
    if "StudentAbsenceDays" in df.columns:
        df["high_absence"] = (df["StudentAbsenceDays"] == "Above-7").astype(int)
    if binary and "final_result" in df.columns:
        df["final_result"] = df["final_result"].map(
            {"Fail": "AtRisk", "Pass": "Success", "Distinction": "Success"}
        ).fillna("AtRisk")
    return df


# ── UCI Student Performance engineering ───────────────────────────────────────

def engineer_uci_perf(df: pd.DataFrame, binary: bool = False) -> pd.DataFrame:
    df = df.copy()
    # G3 binning: keep as multiclass or binary
    if "G3" in df.columns:
        df["G3"] = pd.to_numeric(df["G3"], errors="coerce").fillna(0)
        if binary:
            df["final_result"] = (df["G3"] >= 10).map({True: "Success", False: "AtRisk"})
        else:
            df["final_result"] = pd.cut(df["G3"], bins=[-1, 9, 14, 20],
                                        labels=["Fail", "Pass", "Distinction"])
        df = df.drop(columns=["G1", "G2", "G3"], errors="ignore")
    num_cols = ["age", "studytime", "failures", "absences", "famrel",
                "freetime", "goout", "Dalc", "Walc", "health", "Medu", "Fedu", "traveltime"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Medu" in df.columns and "Fedu" in df.columns:
        df["parent_edu_avg"] = (df["Medu"].fillna(0) + df["Fedu"].fillna(0)) / 2
    if "Dalc" in df.columns and "Walc" in df.columns:
        df["total_alcohol"] = df["Dalc"].fillna(0) + df["Walc"].fillna(0)
        df["high_alcohol"] = (df["total_alcohol"] > 4).astype(int)
    if "studytime" in df.columns and "failures" in df.columns:
        df["study_efficiency"] = df["studytime"].fillna(1) / (df["failures"].fillna(0) + 1)
    if "absences" in df.columns:
        df["log_absences"] = np.log1p(df["absences"].fillna(0))
        df["high_absence"] = (df["absences"] > 10).astype(int)
    return df

# ── Preprocessing pipeline ─────────────────────────────────────────────────────

def build_preprocessor(num_cols: List[str], cat_cols: List[str],
                        model: str = "xgboost") -> ColumnTransformer:
    parts = []
    if num_cols:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if model == "logistic_regression":
            steps.append(("scale", StandardScaler()))
        parts.append(("num", Pipeline(steps), num_cols))
    if cat_cols:
        if model == "catboost":
            cat_steps = [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        else:
            cat_steps = [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
            ]
        parts.append(("cat", Pipeline(cat_steps), cat_cols))
    return ColumnTransformer(parts, remainder="drop")


# ── Model factory ─────────────────────────────────────────────────────────────

def get_models(seed: int = 42, n_jobs: int = -1, class_weight: str = "balanced") -> Dict:
    cw = None if class_weight == "none" else class_weight
    models: Dict = {
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight=cw, random_state=seed, n_jobs=n_jobs
        ),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, gamma=0,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=seed, n_jobs=n_jobs, verbosity=0, tree_method="hist",
        )
    if _HAS_LGB:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=500, num_leaves=63, max_depth=-1,
            learning_rate=0.05, feature_fraction=0.85, bagging_fraction=0.85,
            bagging_freq=5, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.5,
            class_weight=cw, random_state=seed, n_jobs=n_jobs, verbosity=-1,
        )
    if _HAS_CAT:
        cat_kw: Dict = {
            "iterations": 500, "depth": 7, "learning_rate": 0.05,
            "l2_leaf_reg": 3, "verbose": 0, "random_state": seed,
            "thread_count": n_jobs,
        }
        if cw == "balanced":
            cat_kw["auto_class_weights"] = "Balanced"
        models["catboost"] = CatBoostClassifier(**cat_kw)
    return models


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def _tune_xgb(trial, X, y, cv):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 3),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        "use_label_encoder": False, "eval_metric": "mlogloss",
        "random_state": 42, "n_jobs": 1, "verbosity": 0, "tree_method": "hist",
    }
    s = cross_val_score(XGBClassifier(**p), X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    return s.mean()


def _tune_lgb(trial, X, y, cv):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 5,
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        "random_state": 42, "n_jobs": 1, "verbosity": -1, "class_weight": "balanced",
    }
    s = cross_val_score(LGBMClassifier(**p), X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    return s.mean()


def _tune_cat(trial, X, y, cv):
    p = {
        "iterations": trial.suggest_int("iterations", 300, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_float("random_strength", 0, 2),
        "verbose": 0, "random_state": 42, "auto_class_weights": "Balanced",
    }
    s = cross_val_score(CatBoostClassifier(**p), X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    return s.mean()


def tune(model_name: str, X, y, cv, n_trials: int = 60) -> Dict:
    if not _HAS_OPTUNA:
        return {}
    fns = {"xgboost": _tune_xgb, "lightgbm": _tune_lgb, "catboost": _tune_cat}
    if model_name not in fns:
        return {}
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: fns[model_name](t, X, y, cv),
                   n_trials=n_trials, show_progress_bar=False)
    logger.info("  Optuna [%s] best f1_macro=%.4f params=%s",
                model_name, study.best_value, study.best_params)
    return study.best_params

# ── Stacking ensemble ─────────────────────────────────────────────────────────

def build_stacking(seed: int = 42, n_jobs: int = -1) -> Optional[StackingClassifier]:
    estimators = []
    if _HAS_XGB:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=seed, n_jobs=1, verbosity=0, tree_method="hist"
        )))
    if _HAS_LGB:
        estimators.append(("lgb", LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.05,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
            class_weight="balanced", random_state=seed, n_jobs=1, verbosity=-1
        )))
    if _HAS_CAT:
        estimators.append(("cat", CatBoostClassifier(
            iterations=400, depth=7, learning_rate=0.05, l2_leaf_reg=3,
            verbose=0, random_state=seed, auto_class_weights="Balanced"
        )))
    estimators.append(("rf", RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1
    )))
    if len(estimators) < 2:
        return None
    meta = LogisticRegression(max_iter=2000, C=1.0, random_state=seed,
                              class_weight="balanced")
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        passthrough=True,
        n_jobs=1,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def full_eval(y_true, y_pred, y_prob, le: LabelEncoder, model_name: str, ds_name: str) -> Dict:
    r: Dict = {
        "dataset": ds_name, "model": model_name,
        "n_samples": len(y_true),
        "n_classes": len(np.unique(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            classes = np.arange(len(le.classes_))
            y_bin = label_binarize(y_true, classes=classes)
            if len(le.classes_) == 2:
                r["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                r["roc_auc"] = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
        except Exception:
            r["roc_auc"] = float("nan")
    else:
        r["roc_auc"] = float("nan")
    return r


# ── SHAP ──────────────────────────────────────────────────────────────────────

def get_shap_importance(clf, X_transformed: np.ndarray,
                        feature_names: List[str], top_n: int = 25) -> Optional[List]:
    if not _HAS_SHAP:
        return None
    try:
        exp = _shap.TreeExplainer(clf)
        sv = exp.shap_values(X_transformed)
        if isinstance(sv, list):
            vals = np.mean([np.abs(v) for v in sv], axis=0).mean(axis=0)
        elif sv.ndim == 3:
            vals = np.abs(sv).mean(axis=(0, 2))
        else:
            vals = np.abs(sv).mean(axis=0)
        n = min(len(feature_names), len(vals))
        order = np.argsort(vals[:n])[::-1][:top_n]
        return [(feature_names[i], float(vals[i])) for i in order]
    except Exception as e:
        logger.debug("SHAP failed: %s", e)
        return None


def clean_feat_names(names):
    return [
        n.replace("num__", "").replace("cat__ohe__", "").replace("cat__enc__", "")
        .replace("cat__", "").replace("remainder__", "")
        for n in names
    ]

# ── Core training function ────────────────────────────────────────────────────

def run_experiment(
    ds_name: str,
    df: pd.DataFrame,
    engineer_fn,
    target_col: str = "final_result",
    binary: bool = False,
    model_names: Optional[List[str]] = None,
    tune_flag: bool = False,
    tune_trials: int = 60,
    do_shap: bool = True,
    run_stacking: bool = False,
    seed: int = 42,
    n_jobs: int = -1,
    cv_folds: int = 5,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    label = f"{ds_name}{'_binary' if binary else '_4class'}"
    logger.info("\n%s\n=== %s ===\n%s", "=" * 65, label, "=" * 65)

    df = engineer_fn(df, binary=binary)
    if target_col not in df.columns:
        logger.error("  '%s' not found, skipping.", target_col)
        return []

    df = df.dropna(subset=[target_col])
    y_raw = df[target_col].astype(str)
    X = df.drop(columns=[target_col], errors="ignore")
    X = X.drop(columns=[c for c in OULAD_DROP if c in X.columns], errors="ignore")
    X = X.dropna(axis=1, how="all")

    logger.info("  Shape: %s | Classes: %s", X.shape, y_raw.value_counts().to_dict())

    if len(X) < 30:
        logger.warning("  Too few rows, skipping.")
        return []

    # ── FIX 4: 100% small-dataset audit ───────────────────────────────────────
    # Warn if a dataset is very small: results may not generalise.
    if len(X) < 200:
        logger.warning(
            "  AUDIT: Only %d rows — 100%% accuracy likely reflects overfitting "
            "or target leakage on a very small test set. Treat with caution.", len(X))
    # Check that the target is not derivable from any remaining feature
    target_values = set(y_raw.unique())
    suspicious = [c for c in X.columns
                  if X[c].dtype == object
                  and set(X[c].dropna().unique()).issubset(target_values)
                  and X[c].nunique() <= len(target_values)]
    if suspicious:
        logger.warning("  AUDIT: Columns %s have the same unique values as the target "
                       "— possible target leakage. Verify feature construction.", suspicious)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    y_series = pd.Series(y_enc, index=X.index)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── FIX 3: ONE split shared by ALL models in this experiment ─────────────
    # Created ONCE here. Every model below receives the same X_train, X_test.
    # No model creates its own split — this guarantees a fair comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_series, test_size=0.2, random_state=seed, stratify=y_series
    )
    logger.info("  FIX3: single split  train=%d  test=%d  (identical for ALL models)",
                len(X_train), len(X_test))

    min_class = y_train.value_counts().min()
    folds = max(2, min(cv_folds, min_class))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    all_models = get_models(seed=seed, n_jobs=n_jobs)
    if model_names:
        all_models = {k: v for k, v in all_models.items() if k in model_names}

    results: List[Dict] = []

    for mname, clf in all_models.items():
        logger.info("  [%s] Training...", mname)
        pre = build_preprocessor(num_cols, cat_cols, mname)
        pipe = Pipeline([("preprocessor", pre), ("clf", clf)])

        # CV — on training split only
        try:
            cv_sc = cross_val_score(pipe, X_train, y_train, cv=cv,
                                    scoring="f1_macro", n_jobs=1)
            logger.info("  [%s] CV f1_macro: %.4f ± %.4f", mname, cv_sc.mean(), cv_sc.std())
        except Exception as e:
            logger.warning("  [%s] CV error: %s", mname, e)

        # Optuna tuning (on training split only)
        if tune_flag and mname in ("xgboost", "lightgbm", "catboost"):
            logger.info("  [%s] Tuning with Optuna (%d trials)...", mname, tune_trials)
            X_train_t = pre.fit_transform(X_train)
            best = tune(mname, X_train_t, y_train.values, cv, n_trials=tune_trials)
            if best:
                for remove_key in ["use_label_encoder", "eval_metric", "random_state",
                                   "n_jobs", "verbosity", "verbose", "class_weight",
                                   "auto_class_weights", "thread_count"]:
                    best.pop(remove_key, None)
                try:
                    pipe.named_steps["clf"].set_params(**best)
                except Exception as e:
                    logger.warning("  [%s] Could not set tuned params: %s", mname, e)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        try:
            y_prob = pipe.predict_proba(X_test)
        except Exception:
            y_prob = None

        r = full_eval(y_test.values, y_pred, y_prob, le, mname, label)
        results.append(r)

        # ── Export predictions + probabilities for graph generation ──────────
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            pred_df = pd.DataFrame({"y_true": le.inverse_transform(y_test.values),
                                    "y_pred": le.inverse_transform(y_pred)})
            if y_prob is not None:
                for ci, cname in enumerate(le.classes_):
                    pred_df[f"prob_{cname}"] = y_prob[:, ci]
                pred_df["confidence"] = y_prob.max(axis=1)
            pred_path = output_dir / f"predictions_{label}_{mname}.csv"
            pred_df.to_csv(pred_path, index=False)

        logger.info(
            "  [%s] acc=%.4f  f1_mac=%.4f  bacc=%.4f  kappa=%.4f  roc=%.4f",
            mname, r["accuracy"], r["f1_macro"], r["balanced_accuracy"],
            r["cohen_kappa"], r.get("roc_auc", float("nan"))
        )
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=le.classes_))

        # SHAP
        if do_shap and _HAS_SHAP and mname in ("xgboost", "lightgbm", "catboost", "random_forest"):
            try:
                feat_names = clean_feat_names(
                    list(pipe.named_steps["preprocessor"].get_feature_names_out()))
            except Exception:
                feat_names = [f"f{i}" for i in range(
                    pipe.named_steps["preprocessor"].transform(X_test[:1]).shape[1])]
            X_shap_raw = X_test.sample(min(500, len(X_test)), random_state=seed)
            X_shap_t = pipe.named_steps["preprocessor"].transform(X_shap_raw)
            shap_imp = get_shap_importance(pipe.named_steps["clf"], X_shap_t, feat_names)
            if shap_imp:
                logger.info("  [%s] Top SHAP features:", mname)
                for feat, val in shap_imp[:15]:
                    logger.info("    %-40s %.5f", feat, val)
                if output_dir:
                    pd.DataFrame(shap_imp, columns=["feature", "shap_importance"]).to_csv(
                        output_dir / f"shap_{label}_{mname}.csv", index=False)

        # Save
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, output_dir / f"model_{label}_{mname}.pkl")
            joblib.dump(le, output_dir / f"encoder_{label}_{mname}.pkl")

    # Stacking ensemble
    if run_stacking:
        logger.info("  [stacking] Training ensemble...")
        stack = build_stacking(seed=seed, n_jobs=n_jobs)
        if stack:
            pre_s = build_preprocessor(num_cols, cat_cols, "xgboost")
            X_tr_t = pre_s.fit_transform(X_tr)
            X_te_t = pre_s.transform(X_te)
            try:
                stack.fit(X_tr_t, y_tr)
                y_pred_s = stack.predict(X_te_t)
                try:
                    y_prob_s = stack.predict_proba(X_te_t)
                except Exception:
                    y_prob_s = None
                r_s = full_eval(y_te.values, y_pred_s, y_prob_s, le, "stacking_ensemble", label)
                results.append(r_s)
                logger.info(
                    "  [stacking] acc=%.4f  f1_mac=%.4f  bacc=%.4f  kappa=%.4f",
                    r_s["accuracy"], r_s["f1_macro"], r_s["balanced_accuracy"], r_s["cohen_kappa"]
                )
                logger.info("\n%s", classification_report(y_te, y_pred_s, target_names=le.classes_))
                if output_dir:
                    joblib.dump((pre_s, stack, le), output_dir / f"stacking_{label}.pkl")
            except Exception as e:
                logger.warning("  [stacking] Failed: %s", e)

    return results

# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results: List[Dict]) -> None:
    if not all_results:
        return
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 95)
    print(f"{'EXPERIMENT':<35} {'MODEL':<20} {'ACC':>6} {'F1-MAC':>7} "
          f"{'BACC':>6} {'KAPPA':>6} {'ROC':>6} {'N':>6}")
    print("=" * 95)
    for _, row in df.sort_values(["dataset", "accuracy"], ascending=[True, False]).iterrows():
        print(
            f"{row['dataset']:<35} {row['model']:<20} "
            f"{row['accuracy']:>6.4f} {row['f1_macro']:>7.4f} "
            f"{row['balanced_accuracy']:>6.4f} {row['cohen_kappa']:>6.4f} "
            f"{row.get('roc_auc', float('nan')):>6.4f} {int(row['n_samples']):>6}"
        )
    print("=" * 95)
    print("\n=== Best Per Experiment ===")
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        best = sub.loc[sub["accuracy"].idxmax()]
        star = " ★" if best["accuracy"] >= 0.90 else (" ↑" if best["accuracy"] >= 0.80 else "")
        print(f"  {ds:<35} -> {best['model']:<20} "
              f"acc={best['accuracy']:.4f}  f1_mac={best['f1_macro']:.4f}"
              f"  roc={best.get('roc_auc', float('nan')):.4f}{star}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="High-Accuracy Student Performance Pipeline")
    parser.add_argument("--dataset", default="all",
                        help="Datasets to run: all | oulad | dropout | xapi | uci_perf "
                             "| mental_health | placement | academics. Comma-separated.")
    parser.add_argument("--model", default="xgboost,lightgbm,catboost",
                        help="Models: xgboost,lightgbm,catboost,random_forest. Comma-separated.")
    parser.add_argument("--binary", action="store_true", default=False,
                        help="Use binary Success/AtRisk target instead of multiclass.")
    parser.add_argument("--both", action="store_true", default=False,
                        help="Run both binary AND 4-class experiments on each dataset.")
    parser.add_argument("--tune", action="store_true",
                        help="Optuna hyperparameter tuning (adds ~5-10 min per model).")
    parser.add_argument("--tune-trials", type=int, default=60)
    parser.add_argument("--no-shap", dest="shap", action="store_false", default=True)
    parser.add_argument("--stacking", action="store_true",
                        help="Add stacking ensemble to OULAD experiment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = ROOT / args.output_dir

    # Dataset loaders
    def _get_oulad():
        p = ROOT / "oulad_ml_table.csv"
        if not p.exists(): p = ROOT / "oulad_ml_table_week8.csv"
        if not p.exists(): return None
        d = _read(p); d.columns = d.columns.str.strip(); return d

    def _get_dropout():
        p = ROOT / "dropout" / "data.csv"
        if not p.exists(): return None
        d = _read(p); d.columns = d.columns.str.strip()
        if "Target" in d.columns: d = d.rename(columns={"Target": "final_result"})
        return d

    def _get_xapi():
        p = ROOT / "xAPI" / "xAPI-Edu-Data.csv"
        if not p.exists(): return None
        d = _read(p); d.columns = d.columns.str.strip()
        if "Class" in d.columns:
            d["Class"] = d["Class"].map({"L": "Fail", "M": "Pass", "H": "Distinction"}).fillna(d["Class"])
            d = d.rename(columns={"Class": "final_result"})
        return d

    def _get_uci_perf():
        parts = []
        for f in ["student+performance/student/student-mat.csv",
                  "UI_student+performance/student/student-mat.csv"]:
            p = ROOT / f
            if p.exists():
                d = _read(p); d["subject"] = "math"; parts.append(d); break
        for f in ["student+performance/student/student-por.csv",
                  "UI_student+performance/student/student-por.csv"]:
            p = ROOT / f
            if p.exists():
                d = _read(p); d["subject"] = "por"; parts.append(d); break
        if not parts: return None
        return pd.concat(parts, ignore_index=True)

    def _get_mental():
        p = ROOT / "Student Mental health.csv"
        if not p.exists(): return None
        d = _read(p); d.columns = d.columns.str.strip()
        d = d.rename(columns={
            "Choose your gender": "gender", "What is your course?": "course",
            "Your current year of Study": "study_year",
            "What is your CGPA?": "cgpa_band", "Marital status": "marital_status",
            "Do you have Depression?": "depression", "Do you have Anxiety?": "anxiety",
            "Do you have Panic attack?": "panic_attack",
            "Did you seek any specialist for a treatment?": "treatment",
        })
        d = d.drop(columns=["Timestamp"], errors="ignore")
        def cgpa_label(s):
            s = str(s)
            if any(x in s for x in ["3.50", "4.00", "3.5", "4.0"]): return "Distinction"
            if any(x in s for x in ["3.00", "3.0", "2.50", "2.5"]): return "Pass"
            return "Fail"
        d["final_result"] = d["cgpa_band"].apply(cgpa_label)
        return d

    def _get_placement():
        p = ROOT / "Placement_Data_Full_Class.csv"
        if not p.exists(): return None
        d = _read(p); d.columns = d.columns.str.strip()
        d = d.rename(columns={"status": "final_result"})
        d["final_result"] = d["final_result"].map({"Placed": "Pass", "Not Placed": "Fail"}).fillna(d["final_result"])
        return d.drop(columns=["sl_no", "salary"], errors="ignore")

    def _get_academics():
        p = ROOT / "academics" / "data.csv"
        if not p.exists(): return None
        d = _read(p); d.columns = d.columns.str.strip()
        if "tnp" in d.columns:
            d["final_result"] = d["tnp"].map(
                {"Best": "Distinction", "Vg": "Pass", "Good": "Pass", "Pass": "Fail"}
            ).fillna("Pass")
        else: return None
        return d

    REGISTRY = {
        "oulad": (_get_oulad, engineer_oulad),
        "dropout": (_get_dropout, engineer_dropout),
        "xapi": (_get_xapi, engineer_xapi),
        "uci_perf": (_get_uci_perf, engineer_uci_perf),
        "mental_health": (_get_mental, lambda df, binary=False: df),
        "placement": (_get_placement, lambda df, binary=False: df),
        "academics": (_get_academics, lambda df, binary=False: df),
    }

    if args.dataset in ("all",):
        datasets_to_run = list(REGISTRY.keys())
    else:
        datasets_to_run = [d.strip() for d in args.dataset.split(",")]

    model_names = [m.strip() for m in args.model.split(",")]

    # Determine binary/multiclass modes to run
    binary_modes: List[bool] = []
    if args.both:
        binary_modes = [False, True]
    elif args.binary:
        binary_modes = [True]
    else:
        binary_modes = [False]

    all_results: List[Dict] = []

    for ds_name in datasets_to_run:
        if ds_name not in REGISTRY:
            logger.warning("Unknown dataset '%s'", ds_name)
            continue
        loader, eng_fn = REGISTRY[ds_name]
        df = loader()
        if df is None:
            logger.warning("Dataset '%s' not available, skipping.", ds_name)
            continue
        for binary in binary_modes:
            r = run_experiment(
                ds_name=ds_name,
                df=df.copy(),
                engineer_fn=eng_fn,
                binary=binary,
                model_names=model_names,
                tune_flag=args.tune,
                tune_trials=args.tune_trials,
                do_shap=args.shap,
                run_stacking=args.stacking and ds_name == "oulad",
                seed=args.seed,
                n_jobs=args.n_jobs,
                cv_folds=args.cv_folds,
                output_dir=out_dir,
            )
            all_results.extend(r)

    if all_results:
        print_summary(all_results)
        out_dir.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(all_results)
        results_csv = out_dir / "high_accuracy_results.csv"
        # Append to existing results so partial runs (e.g. SHAP-only for one model)
        # don't overwrite previously computed rows for other models.
        if results_csv.exists():
            existing = pd.read_csv(results_csv)
            # Remove rows for datasets+models being re-run, then append fresh rows
            key_cols = ["dataset", "model"]
            mask = existing.set_index(key_cols).index.isin(
                new_df.set_index(key_cols).index)
            existing = existing[~mask]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(results_csv, index=False)
        logger.info("\nResults saved → %s", results_csv)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
