#!/usr/bin/env python3
"""
Unified Multi-Dataset Student Performance Pipeline
===================================================
Combines OULAD + UCI Dropout + xAPI + UCI Student Performance + Mental Health
into a single high-accuracy ML pipeline targeting 90-95% accuracy.

Datasets used:
  - OULAD           : 32,593 students, VLE clickstream + assessments
  - UCI Dropout     : 4,424 students, academic + socioeconomic
  - xAPI            : 480 students, LMS behaviour
  - UCI Perf (mat)  : 395 students, Portuguese Math grades
  - UCI Perf (por)  : 649 students, Portuguese Language grades
  - Student Academics: 131 students, Indian university
  - Mental Health   : 101 students, mental health questionnaire
  - Placement       : 215 students, placement outcomes

Strategy:
  Each dataset is trained on its own strong signal (within-dataset).
  OULAD gets the deepest treatment. Then all datasets contribute to
  an ensemble. XGBoost / LightGBM / CatBoost + Optuna tuning + SHAP.

Usage:
    python unified_pipeline.py [--dataset all|oulad|dropout|xapi|uci_perf|compare]
                               [--tune] [--shap] [--output-dir results/]
"""
from __future__ import annotations
import argparse
import logging
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Optional imports
# ──────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    XGBClassifier = None
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    LGBMClassifier = None
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    CatBoostClassifier = None
    _HAS_CAT = False

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _shap = None
    _HAS_SHAP = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    optuna = None
    _HAS_OPTUNA = False

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).parent

# ──────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────

def _read(path: Path, **kw) -> pd.DataFrame:
    """Auto-detect delimiter and read CSV."""
    for sep in [None, ",", ";", "\t", "|"]:
        try:
            engine = "python" if sep is None else None
            df = pd.read_csv(path, sep=sep, engine=engine, low_memory=False, **kw)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False, **kw)


def load_oulad() -> Optional[pd.DataFrame]:
    """Load pre-built OULAD ML table (built by oulad_pipeline.py)."""
    path = ROOT / "oulad_ml_table.csv"
    if not path.exists():
        # Try week-8 version as fallback
        path = ROOT / "oulad_ml_table_week8.csv"
    if not path.exists():
        logger.warning("OULAD ML table not found. Run oulad_pipeline.py first.")
        return None
    df = _read(path)
    df.columns = df.columns.str.strip()
    if "final_result" not in df.columns:
        logger.warning("OULAD: 'final_result' column not found.")
        return None
    return df


def load_dropout() -> Optional[pd.DataFrame]:
    """UCI Predict Students' Dropout and Academic Success (id=697)."""
    path = ROOT / "dropout" / "data.csv"
    if not path.exists():
        logger.warning("Dropout dataset not found at %s", path)
        return None
    df = _read(path)
    df.columns = df.columns.str.strip()
    if "Target" not in df.columns:
        # Try last column
        df = df.rename(columns={df.columns[-1]: "Target"})
    df = df.rename(columns={"Target": "final_result"})
    return df


def load_xapi() -> Optional[pd.DataFrame]:
    """xAPI Educational Data (LMS behaviour)."""
    path = ROOT / "xAPI" / "xAPI-Edu-Data.csv"
    if not path.exists():
        logger.warning("xAPI dataset not found.")
        return None
    df = _read(path)
    df.columns = df.columns.str.strip()
    if "Class" not in df.columns:
        logger.warning("xAPI: 'Class' column not found.")
        return None
    df = df.rename(columns={"Class": "final_result"})
    # Map L/M/H → categorical pass/fail-like labels for alignment
    df["final_result"] = df["final_result"].map({"L": "Fail", "M": "Pass", "H": "Distinction"}).fillna(df["final_result"])
    return df


def load_uci_performance() -> Optional[pd.DataFrame]:
    """UCI Student Performance — Math + Portuguese combined."""
    paths = [
        ROOT / "student+performance" / "student" / "student-mat.csv",
        ROOT / "UI_student+performance" / "student" / "student-mat.csv",
    ]
    por_paths = [
        ROOT / "student+performance" / "student" / "student-por.csv",
        ROOT / "UI_student+performance" / "student" / "student-por.csv",
    ]
    dfs = []
    for p in paths:
        if p.exists():
            d = _read(p)
            d["subject"] = "math"
            dfs.append(d)
            break
    for p in por_paths:
        if p.exists():
            d = _read(p)
            d["subject"] = "portuguese"
            dfs.append(d)
            break
    if not dfs:
        logger.warning("UCI Student Performance CSVs not found.")
        return None
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    if "G3" not in df.columns:
        logger.warning("UCI Perf: G3 column not found.")
        return None
    # Binarise G3: 0-9 = Fail, 10-14 = Pass, 15-20 = Distinction
    df["G3"] = pd.to_numeric(df["G3"], errors="coerce").fillna(0)
    def grade_to_label(g):
        if g < 10: return "Fail"
        if g < 15: return "Pass"
        return "Distinction"
    df["final_result"] = df["G3"].apply(grade_to_label)
    df = df.drop(columns=["G1", "G2", "G3"], errors="ignore")
    return df


def load_mental_health() -> Optional[pd.DataFrame]:
    """Kaggle Student Mental Health survey."""
    path = ROOT / "Student Mental health.csv"
    if not path.exists():
        return None
    df = _read(path)
    df.columns = df.columns.str.strip()
    # Rename to shorter names
    rename = {
        "Choose your gender": "gender",
        "What is your course?": "course",
        "Your current year of Study": "study_year",
        "What is your CGPA?": "cgpa_band",
        "Marital status": "marital_status",
        "Do you have Depression?": "depression",
        "Do you have Anxiety?": "anxiety",
        "Do you have Panic attack?": "panic_attack",
        "Did you seek any specialist for a treatment?": "treatment",
    }
    df = df.rename(columns=rename)
    df = df.drop(columns=["Timestamp"], errors="ignore")
    # Derive target from CGPA: 3.5+ = Distinction, 3.0-3.49 = Pass, below 3.0 = Fail
    def cgpa_to_label(s):
        s = str(s).strip()
        if "3.50" in s or "4.00" in s or "3.5" in s or "4.0" in s:
            return "Distinction"
        if "3.00" in s or "3.0" in s or "2.50" in s or "2.5" in s:
            return "Pass"
        return "Fail"
    df["final_result"] = df["cgpa_band"].apply(cgpa_to_label)
    return df


def load_placement() -> Optional[pd.DataFrame]:
    """Campus Placement dataset."""
    path = ROOT / "Placement_Data_Full_Class.csv"
    if not path.exists():
        return None
    df = _read(path)
    df.columns = df.columns.str.strip()
    if "status" not in df.columns:
        return None
    df = df.rename(columns={"status": "final_result"})
    df["final_result"] = df["final_result"].map(
        {"Placed": "Pass", "Not Placed": "Fail"}
    ).fillna(df["final_result"])
    df = df.drop(columns=["sl_no", "salary"], errors="ignore")
    return df


def load_academics() -> Optional[pd.DataFrame]:
    """UCI Student Academics Performance."""
    path = ROOT / "academics" / "data.csv"
    if not path.exists():
        return None
    df = _read(path)
    df.columns = df.columns.str.strip()
    # 'atd' = attendance (Good/Average/Poor), map to target
    if "atd" in df.columns:
        df["final_result"] = df["atd"].map(
            {"Good": "Pass", "Average": "Pass", "Poor": "Fail"}
        ).fillna("Pass")
    elif "tnp" in df.columns:
        df["final_result"] = df["tnp"].map(
            {"Best": "Distinction", "Vg": "Pass", "Good": "Pass", "Pass": "Fail"}
        ).fillna("Pass")
    else:
        return None
    return df

# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────

OULAD_LATE_FEATURES = {
    "assessment_completion_ratio", "last_ts", "last_assessment_day",
    "assessment_span_days", "inactivity_days", "missed_assessments",
    "total_assessments", "week_click_sum_1_12", "click_growth_rate",
    "click_variance", "longest_inactive_gap", "avg_score", "score_std",
    "assessment_score_trend", "num_assessments", "late_submission_count",
}

OULAD_DROP_ALWAYS = {
    "date_unregistration", "date_unreg", "date_unregistered",
    "weighted_score", "active_weeks", "clicks_per_active_week",
    "assessments_per_week", "activity_count", "days_active",
    "avg_clicks_per_day", "week_click_sum_1_4", "registration_delay_category",
    "id_student", "id_assessment", "id_site", "first_ts",
}


def engineer_oulad(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering specifically for OULAD."""
    df = df.copy()

    # Drop always-leaky or always-redundant columns
    df = df.drop(columns=[c for c in OULAD_DROP_ALWAYS if c in df.columns])

    # Fill numeric NAs with 0 for VLE/assessment features (no activity = 0)
    vle_cols = [c for c in df.columns if "click" in c or "week" in c]
    assess_cols = ["avg_score", "score_std", "num_assessments",
                   "assessment_completion_ratio", "missed_assessments",
                   "late_submission_count", "assessment_score_trend"]
    for col in vle_cols + assess_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Engagement ratio: clicks in first 4 weeks / total clicks
    if "clicks_until_week4" in df.columns and "total_clicks" in df.columns:
        denom = df["total_clicks"].replace(0, np.nan)
        df["early_engagement_ratio"] = df["clicks_until_week4"] / denom
        df["early_engagement_ratio"] = df["early_engagement_ratio"].fillna(0).clip(0, 1)

    # Late engagement ratio
    if "clicks_until_week8" in df.columns and "total_clicks" in df.columns:
        denom = df["total_clicks"].replace(0, np.nan)
        df["mid_engagement_ratio"] = df["clicks_until_week8"] / denom
        df["mid_engagement_ratio"] = df["mid_engagement_ratio"].fillna(0).clip(0, 1)

    # Score-to-credits ratio
    if "avg_score" in df.columns and "studied_credits" in df.columns:
        df["score_per_credit"] = df["avg_score"] / df["studied_credits"].replace(0, np.nan)
        df["score_per_credit"] = df["score_per_credit"].fillna(0)

    # Log-transform heavy-tailed click features
    for col in ["total_clicks"] + [f"week{w}_clicks" for w in range(1, 13)]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].fillna(0))

    # prev_attempts risk flag
    if "num_of_prev_attempts" in df.columns:
        df["is_repeat_student"] = (df["num_of_prev_attempts"] > 0).astype(int)

    # IMD deprivation encoding (ordinal: higher number = less deprived)
    if "imd_band" in df.columns:
        imd_order = {
            "0-10%": 1, "10-20": 2, "10-20%": 2, "20-30%": 3,
            "30-40%": 4, "40-50%": 5, "50-60%": 6, "60-70%": 7,
            "70-80%": 8, "80-90%": 9, "90-100%": 10,
        }
        df["imd_numeric"] = df["imd_band"].map(imd_order).fillna(5)

    # Assessment submission completeness
    if "assessment_completion_ratio" in df.columns:
        df["submitted_all"] = (df["assessment_completion_ratio"] >= 1.0).astype(int)

    # Zero-click flag (disengaged students)
    if "total_clicks" in df.columns:
        df["zero_clicks"] = (df["total_clicks"] == 0).astype(int)

    # Drop raw last_ts (future-leaking position marker)
    df = df.drop(columns=["last_ts", "last_assessment_day", "first_assessment_day"], errors="ignore")

    return df


def engineer_dropout(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for UCI Dropout dataset."""
    df = df.copy()
    # Academic performance ratios
    for sem in ["1st", "2nd"]:
        enrolled_col = f"Curricular units {sem} sem (enrolled)"
        approved_col = f"Curricular units {sem} sem (approved)"
        grade_col = f"Curricular units {sem} sem (grade)"
        if enrolled_col in df.columns and approved_col in df.columns:
            denom = df[enrolled_col].replace(0, np.nan)
            df[f"approval_rate_{sem}"] = df[approved_col] / denom
            df[f"approval_rate_{sem}"] = df[f"approval_rate_{sem}"].fillna(0).clip(0, 1)
        if grade_col in df.columns:
            df[f"log_grade_{sem}"] = np.log1p(pd.to_numeric(df[grade_col], errors="coerce").fillna(0))
    # Semester comparison
    if "Curricular units 1st sem (grade)" in df.columns and "Curricular units 2nd sem (grade)" in df.columns:
        g1 = pd.to_numeric(df["Curricular units 1st sem (grade)"], errors="coerce").fillna(0)
        g2 = pd.to_numeric(df["Curricular units 2nd sem (grade)"], errors="coerce").fillna(0)
        df["grade_improvement"] = g2 - g1
        df["grade_consistency"] = (g2 - g1).abs()
    # Financial risk: debtor + no tuition
    if "Debtor" in df.columns and "Tuition fees up to date" in df.columns:
        df["financial_risk"] = ((df["Debtor"] == 1) | (df["Tuition fees up to date"] == 0)).astype(int)
    # Age bands
    if "Age at enrollment" in df.columns:
        df["age_numeric"] = pd.to_numeric(df["Age at enrollment"], errors="coerce").fillna(20)
        df["is_mature_student"] = (df["age_numeric"] > 23).astype(int)
    return df


def engineer_xapi(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for xAPI dataset."""
    df = df.copy()
    # Engagement score
    num_cols = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    present = [c for c in num_cols if c in df.columns]
    if present:
        df["total_engagement"] = df[present].sum(axis=1)
        df["engagement_diversity"] = (df[present] > 0).sum(axis=1)
    # Absence encoding
    if "StudentAbsenceDays" in df.columns:
        df["high_absence"] = (df["StudentAbsenceDays"] == "Above-7").astype(int)
    return df


def engineer_uci_perf(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for UCI Student Performance."""
    df = df.copy()
    num_cols = ["age", "studytime", "failures", "absences",
                "famrel", "freetime", "goout", "Dalc", "Walc", "health",
                "Medu", "Fedu", "traveltime"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Parental education average
    if "Medu" in df.columns and "Fedu" in df.columns:
        df["parent_edu_avg"] = (df["Medu"].fillna(0) + df["Fedu"].fillna(0)) / 2
    # Alcohol risk
    if "Dalc" in df.columns and "Walc" in df.columns:
        df["total_alcohol"] = df["Dalc"].fillna(0) + df["Walc"].fillna(0)
        df["high_alcohol"] = (df["total_alcohol"] > 4).astype(int)
    # Study efficiency
    if "studytime" in df.columns and "failures" in df.columns:
        df["study_efficiency"] = df["studytime"].fillna(1) / (df["failures"].fillna(0) + 1)
    return df

# ──────────────────────────────────────────────
# Preprocessing utilities
# ──────────────────────────────────────────────

def build_preprocessor(numeric_cols: List[str], cat_cols: List[str],
                        model_name: str = "xgboost") -> ColumnTransformer:
    transformers = []
    num_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if model_name in ("logistic_regression",):
        num_steps.append(("scale", StandardScaler()))
    if numeric_cols:
        transformers.append(("num", Pipeline(num_steps), numeric_cols))
    if cat_cols:
        if model_name in ("catboost",):
            # CatBoost handles categoricals natively; encode as ordinal integers
            transformers.append((
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                cat_cols,
            ))
        else:
            transformers.append((
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
                ]),
                cat_cols,
            ))
    return ColumnTransformer(transformers, remainder="drop")


def prepare_dataset(df: pd.DataFrame, target_col: str = "final_result",
                    drop_late: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean and split X / y."""
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates()

    # Drop leakage + identifier columns
    drop_always = list(OULAD_DROP_ALWAYS) + [
        "code_module", "code_presentation",
        "salary", "sl_no", "Timestamp",
    ]
    df = df.drop(columns=[c for c in drop_always if c in df.columns], errors="ignore")

    if drop_late:
        df = df.drop(columns=[c for c in OULAD_LATE_FEATURES if c in df.columns], errors="ignore")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col].dropna()
    X = df.drop(columns=[target_col]).loc[y.index]
    return X, y


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    return le.fit_transform(y.astype(str)), le

# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────

def get_models(seed: int = 42, n_jobs: int = -1, class_weight: str = "balanced") -> Dict:
    cw = class_weight if class_weight != "none" else None
    models: Dict = {}
    models["random_forest"] = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight=cw, random_state=seed, n_jobs=n_jobs
    )
    models["logistic_regression"] = LogisticRegression(
        max_iter=2000, class_weight=cw, random_state=seed, C=0.5
    )
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=seed, n_jobs=n_jobs, verbosity=0,
            tree_method="hist",
        )
    if _HAS_LGB:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=400, max_depth=-1, num_leaves=63,
            learning_rate=0.05, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=5,
            class_weight=cw, random_state=seed, n_jobs=n_jobs, verbosity=-1,
        )
    if _HAS_CAT:
        cat_kw: Dict = {"iterations": 400, "depth": 7, "learning_rate": 0.05,
                        "verbose": 0, "random_state": seed, "thread_count": n_jobs}
        if cw == "balanced":
            cat_kw["auto_class_weights"] = "Balanced"
        models["catboost"] = CatBoostClassifier(**cat_kw)
    return models


# ──────────────────────────────────────────────
# Optuna tuning helpers
# ──────────────────────────────────────────────

def _optuna_objective_xgb(trial, X_tr, y_tr, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        "use_label_encoder": False, "eval_metric": "mlogloss",
        "random_state": 42, "n_jobs": -1, "verbosity": 0, "tree_method": "hist",
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=1)
    return scores.mean()


def _optuna_objective_lgb(trial, X_tr, y_tr, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        "random_state": 42, "n_jobs": -1, "verbosity": -1,
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=1)
    return scores.mean()


def _optuna_objective_cat(trial, X_tr, y_tr, cv):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_float("random_strength", 0, 2),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"]),
        "verbose": 0, "random_state": 42, "auto_class_weights": "Balanced",
    }
    model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=1)
    return scores.mean()


def tune_with_optuna(model_name: str, X_tr, y_tr, cv, n_trials: int = 50) -> Dict:
    """Run Optuna hyperparameter search. Returns best params dict."""
    if not _HAS_OPTUNA:
        logger.info("Optuna not available, skipping tuning.")
        return {}
    study = optuna.create_study(direction="maximize")
    obj_map = {
        "xgboost": _optuna_objective_xgb,
        "lightgbm": _optuna_objective_lgb,
        "catboost": _optuna_objective_cat,
    }
    if model_name not in obj_map:
        return {}
    obj = lambda trial: obj_map[model_name](trial, X_tr, y_tr, cv)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    logger.info("  Optuna best %s f1_macro=%.4f", model_name, study.best_value)
    return study.best_params

# ──────────────────────────────────────────────
# Evaluation & SHAP
# ──────────────────────────────────────────────

def evaluate(pipe, X_test, y_test, le: LabelEncoder, model_name: str, dataset_name: str) -> Dict:
    preds = pipe.predict(X_test)
    results = {
        "dataset": dataset_name,
        "model": model_name,
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "cohen_kappa": cohen_kappa_score(y_test, preds),
        "mcc": matthews_corrcoef(y_test, preds),
    }
    # ROC-AUC (OVR, macro)
    try:
        probas = pipe.predict_proba(X_test)
        classes = np.unique(y_test)
        if len(classes) == 2:
            results["roc_auc"] = roc_auc_score(y_test, probas[:, 1])
        else:
            y_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))
            results["roc_auc"] = roc_auc_score(y_bin, probas, average="macro", multi_class="ovr")
    except Exception:
        results["roc_auc"] = float("nan")
    return results


def compute_shap(pipe, X_sample: pd.DataFrame, feature_names: List[str], top_n: int = 20) -> Optional[List]:
    if not _HAS_SHAP:
        return None
    clf = pipe.named_steps["clf"]
    try:
        X_t = pipe.named_steps["preprocessor"].transform(X_sample)
    except Exception:
        return None
    try:
        explainer = _shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_t)
        if isinstance(sv, list):
            vals = np.mean([np.abs(v) for v in sv], axis=0)
        elif sv.ndim == 3:
            vals = np.abs(sv).mean(axis=(0, 2))
        else:
            vals = np.abs(sv).mean(axis=0)
        names = feature_names if len(feature_names) == vals.shape[0] else [f"f{i}" for i in range(vals.shape[0])]
        order = np.argsort(vals)[::-1][:top_n]
        return [(names[i], float(vals[i])) for i in order]
    except Exception:
        return None


def print_results_table(all_results: List[Dict]) -> None:
    print("\n" + "=" * 80)
    print(f"{'DATASET':<25} {'MODEL':<18} {'ACC':>6} {'F1-MAC':>7} {'BACC':>6} {'KAPPA':>6} {'ROC':>6}")
    print("=" * 80)
    for r in sorted(all_results, key=lambda x: (-x["accuracy"], x["dataset"])):
        print(
            f"{r['dataset']:<25} {r['model']:<18} "
            f"{r['accuracy']:>6.3f} {r['f1_macro']:>7.3f} {r['balanced_accuracy']:>6.3f} "
            f"{r['cohen_kappa']:>6.3f} {r.get('roc_auc', float('nan')):>6.3f}"
        )
    print("=" * 80)

# ──────────────────────────────────────────────
# Stacking ensemble (for OULAD)
# ──────────────────────────────────────────────

def build_stacking_ensemble(seed: int = 42, n_jobs: int = -1) -> Optional[object]:
    """Build a stacking ensemble from XGB + LGB + CatBoost → LogReg meta."""
    estimators = []
    if _HAS_XGB:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=seed, n_jobs=n_jobs, verbosity=0, tree_method="hist"
        )))
    if _HAS_LGB:
        estimators.append(("lgb", LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            class_weight="balanced", random_state=seed, n_jobs=n_jobs, verbosity=-1
        )))
    if _HAS_CAT:
        estimators.append(("cat", CatBoostClassifier(
            iterations=400, depth=7, learning_rate=0.05,
            verbose=0, random_state=seed, auto_class_weights="Balanced",
            thread_count=n_jobs
        )))
    if len(estimators) < 2:
        return None
    final = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        passthrough=False,
        n_jobs=1,  # avoid nested parallelism issues
    )
    return stack


# ──────────────────────────────────────────────
# Per-dataset training runner
# ──────────────────────────────────────────────

def run_dataset(
    name: str,
    df: pd.DataFrame,
    engineer_fn,
    target_col: str = "final_result",
    models_to_run: Optional[List[str]] = None,
    tune: bool = False,
    run_shap: bool = True,
    seed: int = 42,
    n_jobs: int = -1,
    n_cv_folds: int = 5,
    output_dir: Optional[Path] = None,
    run_stacking: bool = False,
    tune_n_trials: int = 50,
) -> List[Dict]:
    """Full train/eval loop for one dataset."""
    logger.info("\n%s\n=== Dataset: %s ===\n%s", "="*60, name, "="*60)

    # Engineer features
    df = engineer_fn(df)

    # Prepare X / y
    try:
        X, y = prepare_dataset(df, target_col)
    except ValueError as e:
        logger.error("  Skipping %s: %s", name, e)
        return []

    logger.info("  Shape: %s  |  Classes: %s", X.shape, y.value_counts().to_dict())

    if len(X) < 20:
        logger.warning("  Too few rows (%d), skipping.", len(X))
        return []

    y_enc, le = encode_labels(y)
    y_enc_series = pd.Series(y_enc, index=X.index)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc_series, test_size=0.2, random_state=seed,
        stratify=y_enc_series
    )

    cv = StratifiedKFold(n_splits=min(n_cv_folds, y_train.value_counts().min()),
                         shuffle=True, random_state=seed)

    all_models = get_models(seed=seed, n_jobs=n_jobs)
    selected = {k: v for k, v in all_models.items()
                if models_to_run is None or k in models_to_run}

    all_results = []

    for mname, clf in selected.items():
        logger.info("  Training %s ...", mname)
        preprocessor = build_preprocessor(numeric_cols, cat_cols, mname)
        pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

        # CV score
        try:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                        scoring="f1_macro", n_jobs=1)
            logger.info("  CV f1_macro: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
        except Exception as e:
            logger.warning("  CV failed: %s", e)

        # Optional Optuna tuning
        if tune and mname in ("xgboost", "lightgbm", "catboost") and _HAS_OPTUNA:
            logger.info("  Tuning %s with Optuna (%d trials)...", mname, tune_n_trials)
            best_params = tune_with_optuna(mname, X_train, y_train, cv, n_trials=tune_n_trials)
            if best_params:
                pipe.named_steps["clf"].set_params(**best_params)

        pipe.fit(X_train, y_train)
        r = evaluate(pipe, X_test, y_test, le, mname, name)
        all_results.append(r)

        logger.info(
            "  TEST  acc=%.4f  f1_macro=%.4f  bacc=%.4f  kappa=%.4f  roc_auc=%.4f",
            r["accuracy"], r["f1_macro"], r["balanced_accuracy"],
            r["cohen_kappa"], r.get("roc_auc", float("nan"))
        )
        logger.info("\n%s", classification_report(
            y_test, pipe.predict(X_test), target_names=le.classes_
        ))

        # SHAP
        if run_shap and _HAS_SHAP and mname in ("xgboost", "lightgbm", "catboost", "random_forest"):
            try:
                feat_names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
            except Exception:
                feat_names = [f"f{i}" for i in range(
                    pipe.named_steps["preprocessor"].transform(X_test[:1]).shape[1])]
            feat_names = [f.replace("num__", "").replace("cat__", "").replace("ohe__", "") for f in feat_names]
            X_shap = X_test.sample(min(500, len(X_test)), random_state=seed)
            shap_imp = compute_shap(pipe, X_shap, feat_names)
            if shap_imp:
                logger.info("  Top SHAP features:")
                for feat, val in shap_imp[:15]:
                    logger.info("    %-35s %.6f", feat, val)
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(shap_imp, columns=["feature", "shap_importance"]).to_csv(
                        output_dir / f"shap_{name}_{mname}.csv", index=False)

        # Save model
        if output_dir:
            import joblib
            output_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, output_dir / f"model_{name}_{mname}.pkl")
            joblib.dump(le, output_dir / f"encoder_{name}_{mname}.pkl")
            logger.info("  Saved model to %s", output_dir / f"model_{name}_{mname}.pkl")

    # Stacking ensemble
    if run_stacking and len(selected) >= 2:
        logger.info("  Training stacking ensemble ...")
        stack = build_stacking_ensemble(seed=seed, n_jobs=n_jobs)
        if stack is not None:
            preprocessor = build_preprocessor(numeric_cols, cat_cols, "xgboost")
            X_train_t = preprocessor.fit_transform(X_train, y_train)
            X_test_t = preprocessor.transform(X_test)
            try:
                stack.fit(X_train_t, y_train)
                preds = stack.predict(X_test_t)
                r_stack = {
                    "dataset": name, "model": "stacking_ensemble",
                    "accuracy": accuracy_score(y_test, preds),
                    "f1_macro": f1_score(y_test, preds, average="macro"),
                    "f1_weighted": f1_score(y_test, preds, average="weighted"),
                    "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                    "cohen_kappa": cohen_kappa_score(y_test, preds),
                    "mcc": matthews_corrcoef(y_test, preds),
                    "roc_auc": float("nan"),
                }
                all_results.append(r_stack)
                logger.info(
                    "  STACK acc=%.4f  f1_macro=%.4f  bacc=%.4f",
                    r_stack["accuracy"], r_stack["f1_macro"], r_stack["balanced_accuracy"]
                )
                if output_dir:
                    import joblib
                    joblib.dump((preprocessor, stack), output_dir / f"stacking_{name}.pkl")
            except Exception as e:
                logger.warning("  Stacking failed: %s", e)

    return all_results

# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

DATASET_REGISTRY = {
    "oulad": (load_oulad, engineer_oulad, "final_result"),
    "dropout": (load_dropout, engineer_dropout, "final_result"),
    "xapi": (load_xapi, engineer_xapi, "final_result"),
    "uci_perf": (load_uci_performance, engineer_uci_perf, "final_result"),
    "mental_health": (load_mental_health, lambda df: df, "final_result"),
    "placement": (load_placement, lambda df: df, "final_result"),
    "academics": (load_academics, lambda df: df, "final_result"),
}


def main():
    parser = argparse.ArgumentParser(description="Unified Student Performance ML Pipeline")
    parser.add_argument("--dataset", default="all",
                        help="Which dataset(s): all | oulad | dropout | xapi | uci_perf | "
                             "mental_health | placement | academics | compare. "
                             "Comma-separated list also accepted.")
    parser.add_argument("--model", default="all",
                        help="Model(s): all | xgboost | lightgbm | catboost | random_forest | "
                             "logistic_regression. Comma-separated.")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning (requires optuna)")
    parser.add_argument("--tune-trials", type=int, default=50,
                        help="Number of Optuna trials per model")
    parser.add_argument("--shap", action="store_true", default=True,
                        help="Compute SHAP explanations (default: on)")
    parser.add_argument("--no-shap", dest="shap", action="store_false")
    parser.add_argument("--stacking", action="store_true",
                        help="Add stacking ensemble on OULAD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="results",
                        help="Directory to save models, SHAP CSVs, results")
    parser.add_argument("--drop-late", action="store_true",
                        help="Drop late/end-of-course features from OULAD for early-warning mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = ROOT / args.output_dir

    # Parse dataset selection
    if args.dataset in ("all", "compare"):
        datasets_to_run = list(DATASET_REGISTRY.keys())
    else:
        datasets_to_run = [d.strip() for d in args.dataset.split(",")]

    # Parse model selection
    if args.model in ("all", "compare"):
        models_to_run = None  # all
    else:
        models_to_run = [m.strip() for m in args.model.split(",")]

    all_results: List[Dict] = []

    for ds_name in datasets_to_run:
        if ds_name not in DATASET_REGISTRY:
            logger.warning("Unknown dataset '%s', skipping.", ds_name)
            continue
        loader, engineer_fn, target_col = DATASET_REGISTRY[ds_name]
        df = loader()
        if df is None:
            logger.warning("Dataset '%s' could not be loaded, skipping.", ds_name)
            continue

        # For OULAD apply drop_late flag
        is_oulad = ds_name == "oulad"
        effective_engineer = engineer_fn
        if is_oulad and args.drop_late:
            def effective_engineer(df, _fn=engineer_fn):
                d = _fn(df)
                return d.drop(columns=[c for c in OULAD_LATE_FEATURES if c in d.columns], errors="ignore")

        results = run_dataset(
            name=ds_name,
            df=df,
            engineer_fn=effective_engineer,
            target_col=target_col,
            models_to_run=models_to_run,
            tune=args.tune,
            run_shap=args.shap,
            seed=args.seed,
            n_jobs=args.n_jobs,
            n_cv_folds=args.cv_folds,
            output_dir=out_dir,
            run_stacking=args.stacking and is_oulad,
            tune_n_trials=args.tune_trials,
        )
        all_results.extend(results)

    if all_results:
        print_results_table(all_results)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(all_results)
        results_path = out_dir / "all_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info("\nResults saved to %s", results_path)

        # Print best per dataset
        print("\n=== Best Result Per Dataset ===")
        for ds in results_df["dataset"].unique():
            sub = results_df[results_df["dataset"] == ds]
            best = sub.loc[sub["accuracy"].idxmax()]
            print(f"  {ds:<20} → {best['model']:<18} "
                  f"acc={best['accuracy']:.4f}  f1_macro={best['f1_macro']:.4f}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
