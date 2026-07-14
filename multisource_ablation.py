#!/usr/bin/env python3
"""
AI-Powered Student Success & Early Intervention Platform
=========================================================

Research-oriented unified multi-source student-risk prediction platform.

Data sources
------------
OULAD:
    LMS behaviour, assessment timing, demographics

xAPI:
    Classroom engagement, attendance, parent-related information

UCI Student Performance:
    Study habits, academic background, lifestyle-related information

Main experiments
----------------
1. Unified model WITH source
2. Unified model WITHOUT source
3. Dataset ablation study
4. Target-distribution audit
5. Cross-validation
6. Test-set evaluation
7. SHAP explainability
8. Actionable student intervention reports

Usage
-----
Standard full research run:

    python synthetic_platform.py

Run only the main unified model:

    python synthetic_platform.py --experiment main

Run source ablation:

    python synthetic_platform.py --experiment source-ablation

Run dataset ablations:

    python synthetic_platform.py --experiment dataset-ablation

Run everything:

    python synthetic_platform.py --experiment all

Benchmark mode with late features:

    python synthetic_platform.py --mode benchmark

Early-warning mode:

    python synthetic_platform.py --mode early-warning

Example:

    python synthetic_platform.py \
        --experiment all \
        --mode early-warning \
        --shap-sample 300 \
        --report-students 5
"""

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

from sklearn.preprocessing import (
    LabelEncoder,
    label_binarize,
)

import joblib

# ── FIX: force UTF-8 stdout/stderr to prevent cp1252 UnicodeEncodeError ──────
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# OPTIONAL SHAP
# ============================================================

try:
    import shap
except ImportError:
    shap = None


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

SEED = 42
np.random.seed(SEED)

warnings.filterwarnings("ignore")


# ============================================================
# LATE FEATURES
# ============================================================

LATE_FEATURES = {
    "assessment_completion_ratio",
    "last_ts",
    "last_assessment_day",
    "assessment_span_days",
    "inactivity_days",
    "num_assessments",
    "missed_assessments",
    "total_assessments",
    "week_click_sum_1_12",
    "click_growth_rate",
    "click_variance",
    "longest_inactive_gap",
    "avg_score",
    "score_std",
    "assessment_score_trend",
}


# ============================================================
# UNIFIED RISK MAPPING
# ============================================================

RISK_LABEL = {

    # OULAD
    "Distinction": "Low",
    "Pass": "Low",
    "Fail": "High",
    "Withdrawn": "High",

    # xAPI
    "H": "Low",
    "M": "Medium",
    "L": "High",

    # UCI
    "High": "Low",
    "Mid": "Medium",
    "Low": "High",
}


UNCERTAINTY_THRESHOLD = 0.55


# ============================================================
# FEATURE GROUPS FOR INTERVENTION
# ============================================================

ACTIONABLE_FEATURES = {
    "total_engagement",
    "engagement_trend",
    "engagement_consistency",
    "weekly_activity_change",
    "consistency_ratio",
    "score_improvement",
    "first_assessment_day",
    "late_submission_count",
    "study_time",
    "absence_flag",
    "registration_early_days",
}


CONTEXTUAL_FEATURES = {
    "prior_failures",
    "lifestyle_risk_score",
    "gender",
    "age_band",
    "education_level",
    "socioeconomic_flag",
    "studied_credits",
    "credits_per_attempt",
}


ADMINISTRATIVE_FEATURES = {
    "source",
    "code_module",
}


# ============================================================
# RECOMMENDATIONS
# ============================================================

RECOMMENDATIONS = {

    "absence_flag":
        "Attendance risk detected. Attend remaining sessions consistently "
        "and contact the instructor if attendance barriers exist.",

    "total_engagement":
        "Overall learning engagement is low. Increase regular LMS activity "
        "and complete pending learning tasks.",

    "engagement_trend":
        "Engagement is declining. Restore a regular study routine and "
        "complete pending course activities.",

    "engagement_consistency":
        "Study activity is inconsistent. Establish a fixed weekly study schedule.",

    "consistency_ratio":
        "Learning activity is irregular. Aim for consistent engagement "
        "throughout each study week.",

    "weekly_activity_change":
        "Recent activity is decreasing. Increase study activity before "
        "the decline affects assessment performance.",

    "score_improvement":
        "Assessment performance is not improving. Seek feedback from a tutor "
        "or instructor and revise weak topics.",

    "study_time":
        "Study time appears insufficient. Increase structured weekly study time.",

    "late_submission_count":
        "Late submissions are contributing to risk. Create a deadline plan "
        "and submit remaining work on time.",

    "registration_early_days":
        "Registration timing may have affected course preparation. Confirm "
        "that all course resources and enrolment requirements are complete.",

    "first_assessment_day":
        "Early assessment participation is a risk factor. Complete assessments "
        "as early as possible and avoid delaying initial submissions.",
}


# ============================================================
# BEHAVIOURAL FEATURE HELPERS
# ============================================================

def _slope(arr):
    arr = np.asarray(arr, dtype=float)

    if len(arr) < 2:
        return 0.0

    if np.all(arr == 0):
        return 0.0

    x = np.arange(len(arr), dtype=float)

    try:
        return float(np.polyfit(x, arr, 1)[0])
    except Exception:
        return 0.0


def _consistency(arr):
    arr = np.asarray(arr, dtype=float)

    if len(arr) == 0:
        return 0.0

    mean = arr.mean()

    if mean == 0:
        return 0.0

    value = 1.0 - (arr.std() / mean)

    return float(max(0.0, value))


def _weekly_activity_change(arr):
    arr = np.asarray(arr, dtype=float)

    if len(arr) < 2:
        return 0.0

    diffs = np.diff(arr)
    previous = arr[:-1]

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        pct = np.where(
            previous > 0,
            diffs / previous,
            0.0,
        )

    return float(np.nanmean(pct))


def _consistency_ratio(arr):
    arr = np.asarray(arr, dtype=float)

    if len(arr) == 0:
        return 0.0

    return float((arr > 0).mean())


def _score_improvement(arr):
    arr = np.asarray(arr, dtype=float)

    valid = arr[~np.isnan(arr)]

    if len(valid) < 2:
        return np.nan

    return float(valid[-1] - valid[0])


# ============================================================
# SAFE SERIES HELPER
# ============================================================

def _series(df, column, default=np.nan):
    if column in df.columns:
        return df[column]

    return pd.Series(
        default,
        index=df.index,
    )


# ============================================================
# OULAD LOADER
# ============================================================

def _from_oulad(path, mode):

    df = pd.read_csv(path)
    df = df.drop_duplicates()

    drop_always = [
        "date_unregistration",
        "date_unreg",
        "date_unregistered",
        "weighted_score",
        "id_student",
        "id_assessment",
        "id_site",
        "first_ts",
        "active_weeks",
        "clicks_per_active_week",
        "assessments_per_week",
        "activity_count",
        "days_active",
        "avg_clicks_per_day",
        "week_click_sum_1_4",
        "registration_delay_category",
    ]

    df = df.drop(
        columns=[
            c
            for c in drop_always
            if c in df.columns
        ]
    )

    if mode == "early-warning":

        df = df.drop(
            columns=[
                c
                for c in df.columns
                if c in LATE_FEATURES
            ]
        )

    week_cols = []

    for week in range(1, 9):

        column = f"week{week}_clicks"

        if column in df.columns:
            week_cols.append(column)

    if week_cols:

        week_arr = (
            df[week_cols]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .fillna(0)
            .values
        )

    else:
        week_arr = None

    out = pd.DataFrame(
        index=df.index
    )

    out["source"] = "oulad"

    # --------------------------------------------------------
    # ENGAGEMENT
    # --------------------------------------------------------

    if week_arr is not None:

        out["total_engagement"] = (
            week_arr.sum(axis=1)
        )

        out["engagement_trend"] = [
            _slope(row)
            for row in week_arr
        ]

        out["engagement_consistency"] = [
            _consistency(row)
            for row in week_arr
        ]

        out["weekly_activity_change"] = [
            _weekly_activity_change(row)
            for row in week_arr
        ]

        out["consistency_ratio"] = [
            _consistency_ratio(row)
            for row in week_arr
        ]

    else:

        for column in [
            "total_engagement",
            "engagement_trend",
            "engagement_consistency",
            "weekly_activity_change",
            "consistency_ratio",
        ]:
            out[column] = np.nan

    # --------------------------------------------------------
    # ASSESSMENT
    # --------------------------------------------------------

    out["score_improvement"] = np.nan

    out["first_assessment_day"] = (
        _series(
            df,
            "first_assessment_day",
        )
    )

    out["late_submission_count"] = (
        _series(
            df,
            "late_submission_count",
            0,
        )
    )

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    out["prior_failures"] = (
        _series(
            df,
            "num_of_prev_attempts",
        )
    )

    out["study_time"] = np.nan

    out["absence_flag"] = np.nan

    out["lifestyle_risk_score"] = np.nan

    # --------------------------------------------------------
    # DEMOGRAPHICS
    # --------------------------------------------------------

    gender_map = {
        "M": 1,
        "F": 0,
        "Male": 1,
        "Female": 0,
    }

    out["gender"] = (
        _series(
            df,
            "gender",
        )
        .map(gender_map)
    )

    out["age_band"] = (
        _series(
            df,
            "age_band",
            "unknown",
        )
        .fillna("unknown")
        .astype(str)
    )

    out["education_level"] = (
        _series(
            df,
            "highest_education",
            "unknown",
        )
        .fillna("unknown")
        .astype(str)
    )

    out["socioeconomic_flag"] = (
        _series(
            df,
            "imd_band",
            "unknown",
        )
        .fillna("unknown")
        .astype(str)
    )

    # --------------------------------------------------------
    # OULAD-SPECIFIC FEATURES
    # --------------------------------------------------------

    out["clicks_per_credit"] = (
        _series(
            df,
            "clicks_per_credit",
        )
    )

    out["credits_per_attempt"] = (
        _series(
            df,
            "credits_per_attempt",
        )
    )

    out["studied_credits"] = (
        _series(
            df,
            "studied_credits",
        )
    )

    out["registration_early_days"] = (
        _series(
            df,
            "registration_early_days",
        )
    )

    out["code_module"] = (
        _series(
            df,
            "code_module",
            "unknown",
        )
        .fillna("unknown")
        .astype(str)
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    out["original_target"] = (
        df["final_result"].values
    )

    out["risk_target"] = (
        out["original_target"]
        .map(RISK_LABEL)
    )

    return (
        out
        .dropna(
            subset=["risk_target"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# xAPI LOADER
# ============================================================

def _from_xapi(path):

    df = pd.read_csv(path)
    df = df.drop_duplicates()

    df.columns = [
        c.strip().lower()
        for c in df.columns
    ]

    df = df.rename(
        columns={
            "raisedhands":
                "raised_hands",

            "visitedresources":
                "visited_resources",

            "announcementsview":
                "announcements_view",

            "studentabsencedays":
                "absence_days",

            "parentansweringsurvey":
                "parent_survey",

            "parentschoolsatisfaction":
                "parent_satisfaction",
        }
    )

    engagement_columns = [
        c
        for c in [
            "raised_hands",
            "visited_resources",
            "announcements_view",
            "discussion",
        ]
        if c in df.columns
    ]

    if engagement_columns:

        engagement_array = (
            df[engagement_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .fillna(0)
            .values
        )

    else:
        engagement_array = None

    out = pd.DataFrame(
        index=df.index
    )

    out["source"] = "xapi"

    # --------------------------------------------------------
    # ENGAGEMENT
    # --------------------------------------------------------

    if engagement_array is not None:

        out["total_engagement"] = (
            engagement_array.sum(axis=1)
        )

        out["engagement_trend"] = [
            _slope(row)
            for row in engagement_array
        ]

        out["engagement_consistency"] = [
            _consistency(row)
            for row in engagement_array
        ]

        out["weekly_activity_change"] = [
            _weekly_activity_change(row)
            for row in engagement_array
        ]

        out["consistency_ratio"] = [
            _consistency_ratio(row)
            for row in engagement_array
        ]

    else:

        for column in [
            "total_engagement",
            "engagement_trend",
            "engagement_consistency",
            "weekly_activity_change",
            "consistency_ratio",
        ]:
            out[column] = np.nan

    # --------------------------------------------------------
    # ASSESSMENT / BACKGROUND
    # --------------------------------------------------------

    out["score_improvement"] = np.nan

    out["first_assessment_day"] = np.nan

    out["late_submission_count"] = np.nan

    out["prior_failures"] = np.nan

    out["study_time"] = np.nan

    # --------------------------------------------------------
    # ATTENDANCE
    # --------------------------------------------------------

    if "absence_days" in df.columns:

        absence_text = (
            df["absence_days"]
            .astype(str)
            .str.lower()
        )

        out["absence_flag"] = (
            absence_text
            .eq("above-7")
            .astype(float)
        )

    else:

        out["absence_flag"] = np.nan

    out["lifestyle_risk_score"] = np.nan

    # --------------------------------------------------------
    # DEMOGRAPHICS
    # --------------------------------------------------------

    gender_map = {
        "m": 1,
        "f": 0,
    }

    out["gender"] = (
        _series(
            df,
            "gender",
            "unknown",
        )
        .astype(str)
        .str.lower()
        .map(gender_map)
    )

    out["age_band"] = "unknown"

    out["education_level"] = (
        _series(
            df,
            "stageid",
            "unknown",
        )
        .fillna("unknown")
        .astype(str)
    )

    out["socioeconomic_flag"] = "unknown"

    # --------------------------------------------------------
    # MISSING SOURCE-SPECIFIC FEATURES
    # --------------------------------------------------------

    out["clicks_per_credit"] = np.nan

    out["credits_per_attempt"] = np.nan

    out["studied_credits"] = np.nan

    out["registration_early_days"] = np.nan

    out["code_module"] = (
        _series(
            df,
            "topic",
            "unknown",
        )
        .fillna("unknown")
        .astype(str)
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    out["original_target"] = (
        df["class"].values
    )

    out["risk_target"] = (
        out["original_target"]
        .map(RISK_LABEL)
    )

    return (
        out
        .dropna(
            subset=["risk_target"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# UCI LOADER
# ============================================================

def _from_uci(path):

    df = pd.read_csv(
        path,
        sep=";",
    )

    df = df.drop_duplicates()

    df.columns = [
        c.strip().lower()
        for c in df.columns
    ]

    if "g3" not in df.columns:

        print(
            "  WARNING: UCI dataset has no G3 column."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    df["grade_band"] = pd.cut(

        df["g3"],

        bins=[
            -1,
            9,
            14,
            20,
        ],

        labels=[
            "Low",
            "Mid",
            "High",
        ],

    ).astype(str)

    # --------------------------------------------------------
    # GRADE TRAJECTORY
    # --------------------------------------------------------

    grade_columns = [
        c
        for c in [
            "g1",
            "g2",
        ]
        if c in df.columns
    ]

    if grade_columns:

        grade_array = (
            df[grade_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .values
        )

    else:

        grade_array = None

    out = pd.DataFrame(
        index=df.index
    )

    out["source"] = "uci"

    # --------------------------------------------------------
    # ENGAGEMENT PROXY
    # --------------------------------------------------------

    study = pd.to_numeric(

        _series(
            df,
            "studytime",
            0,
        ),

        errors="coerce",

    ).fillna(0)

    out["total_engagement"] = (
        study * 10
    )

    out["engagement_trend"] = np.nan

    out["engagement_consistency"] = np.nan

    out["weekly_activity_change"] = np.nan

    out["consistency_ratio"] = np.nan

    # --------------------------------------------------------
    # ASSESSMENT
    # --------------------------------------------------------

    if grade_array is not None:

        out["score_improvement"] = [
            _score_improvement(row)
            for row in grade_array
        ]

    else:

        out["score_improvement"] = np.nan

    out["first_assessment_day"] = np.nan

    out["late_submission_count"] = np.nan

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    out["prior_failures"] = pd.to_numeric(

        _series(
            df,
            "failures",
            0,
        ),

        errors="coerce",

    )

    out["study_time"] = study

    absences = pd.to_numeric(

        _series(
            df,
            "absences",
            0,
        ),

        errors="coerce",

    ).fillna(0)

    out["absence_flag"] = (
        absences > 10
    ).astype(float)

    # --------------------------------------------------------
    # LIFESTYLE RISK
    # --------------------------------------------------------

    risk_columns = [
        c
        for c in [
            "dalc",
            "walc",
            "absences",
            "failures",
        ]
        if c in df.columns
    ]

    if risk_columns:

        numeric_risk = (
            df[risk_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        normalized = numeric_risk.apply(

            lambda series:

                (
                    series - series.min()
                )

                /

                (
                    series.max()
                    - series.min()
                    + 1e-9
                )

        )

        out["lifestyle_risk_score"] = (
            normalized.mean(axis=1)
        )

    else:

        out["lifestyle_risk_score"] = np.nan

    # --------------------------------------------------------
    # DEMOGRAPHICS
    # --------------------------------------------------------

    gender_map = {
        "m": 1,
        "f": 0,
    }

    out["gender"] = (

        _series(
            df,
            "sex",
            "unknown",
        )

        .astype(str)

        .str.lower()

        .map(gender_map)
    )

    age_numeric = pd.to_numeric(

        _series(
            df,
            "age",
        ),

        errors="coerce",

    )

    out["age_band"] = pd.cut(

        age_numeric,

        bins=[
            0,
            17,
            22,
            100,
        ],

        labels=[
            "<=17",
            "18-22",
            "23+",
        ],

    ).astype(str)

    out["education_level"] = (

        _series(
            df,
            "medu",
            "unknown",
        )

        .fillna("unknown")

        .astype(str)
    )

    out["socioeconomic_flag"] = (

        _series(
            df,
            "address",
            "unknown",
        )

        .fillna("unknown")

        .astype(str)
    )

    # --------------------------------------------------------
    # SOURCE-SPECIFIC PLACEHOLDERS
    # --------------------------------------------------------

    out["clicks_per_credit"] = np.nan

    out["credits_per_attempt"] = np.nan

    out["studied_credits"] = np.nan

    out["registration_early_days"] = np.nan

    out["code_module"] = (

        _series(
            df,
            "reason",
            "unknown",
        )

        .fillna("unknown")

        .astype(str)
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    out["original_target"] = (
        df["grade_band"].values
    )

    out["risk_target"] = (
        out["original_target"]
        .map(RISK_LABEL)
    )

    return (

        out

        .dropna(
            subset=["risk_target"]
        )

        .reset_index(drop=True)
    )


# ============================================================
# BUILD FEATURE STORE
# ============================================================

def build_feature_store(root, mode):

    parts = []

    # --------------------------------------------------------
    # OULAD
    # --------------------------------------------------------

    oulad_path = (
        root
        / "oulad_ml_table_week8.csv"
    )

    if not oulad_path.exists():

        oulad_path = (
            root
            / "oulad_ml_table.csv"
        )

    if oulad_path.exists():

        part = _from_oulad(
            oulad_path,
            mode,
        )

        parts.append(part)

        print(
            f"  OULAD loaded   : "
            f"{len(part):>6} rows"
        )

    else:

        print(
            "  OULAD skipped  : "
            "OULAD ML table not found"
        )

    # --------------------------------------------------------
    # xAPI
    # --------------------------------------------------------

    xapi_path = (
        root
        / "xAPI"
        / "xAPI-Edu-Data.csv"
    )

    if xapi_path.exists():

        part = _from_xapi(
            xapi_path
        )

        parts.append(part)

        print(
            f"  xAPI loaded    : "
            f"{len(part):>6} rows"
        )

    else:

        print(
            "  xAPI skipped   : "
            "xAPI-Edu-Data.csv not found"
        )

    # --------------------------------------------------------
    # UCI
    # --------------------------------------------------------

    possible_uci_paths = [

        root
        / "UI_student+performance"
        / "student"
        / "student-mat.csv",

        root
        / "student+performance"
        / "student"
        / "student-mat.csv",
    ]

    uci_path = None

    for candidate in possible_uci_paths:

        if candidate.exists():

            uci_path = candidate

            break

    if uci_path is not None:

        part = _from_uci(
            uci_path
        )

        if not part.empty:

            parts.append(part)

            print(
                f"  UCI loaded     : "
                f"{len(part):>6} rows"
            )

    else:

        print(
            "  UCI skipped    : "
            "student-mat.csv not found"
        )

    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    if not parts:

        raise RuntimeError(
            "No data sources were found."
        )

    master = pd.concat(

        parts,

        ignore_index=True,

    )

    print(
        f"  Master table   : "
        f"{len(master):>6} rows"
        f"  |  "
        f"{len(master.columns)} columns"
    )

    return master


# ============================================================
# TARGET AUDIT
# ============================================================

def audit_target_distribution(master):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "  TARGET DISTRIBUTION AUDIT"
    )

    print(
        "=" * 70
    )

    total = len(master)

    counts = (
        master["risk_target"]
        .value_counts()
    )

    print(
        "\n  Overall unified target:"
    )

    for label, count in counts.items():

        percentage = (
            100
            * count
            / total
        )

        print(
            f"    {label:<10}"
            f": "
            f"{count:>6}"
            f"  "
            f"({percentage:>6.2f}%)"
        )

    print(
        "\n  Target distribution by source:"
    )

    cross = pd.crosstab(

        master["source"],

        master["risk_target"],

        margins=True,

    )

    print(
        cross.to_string()
    )

    print(
        "\n  Row-normalized percentages:"
    )

    cross_pct = pd.crosstab(

        master["source"],

        master["risk_target"],

        normalize="index",

    ) * 100

    print(
        cross_pct.round(2).to_string()
    )

    medium_count = counts.get(
        "Medium",
        0,
    )

    medium_percentage = (
        100
        * medium_count
        / total
    )

    if medium_percentage < 5:

        print(
            "\n  WARNING:"
        )

        print(
            "  Medium risk represents fewer than 5% "
            "of all records."
        )

        print(
            "  This suggests the unified target mapping "
            "should be carefully justified."
        )

        print(
            "  Do not automatically use SMOTE before "
            "auditing the label construction."
        )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    data,
    include_source=True,
    X_train_ref=None,
):
    """
    Prepare features and target.

    FIX (Issue 1 / Issue 10): Medians are computed from training data only.
    Pass X_train_ref=None during training (medians are fitted here and returned).
    Pass X_train_ref=fitted_X when transforming test data.

    Returns X, y, label_encoder, cat_columns, cat_indices, fitted_medians
    """

    target_column = "risk_target"

    drop_columns = [
        "original_target",
        target_column,
    ]

    # FIX (Issue 8 / Issue 11): source excluded from final model by default.
    if not include_source:
        drop_columns.append("source")

    X = data.drop(
        columns=[c for c in drop_columns if c in data.columns]
    ).copy()

    y_raw = data[target_column].copy()

    # --------------------------------------------------------
    # CATEGORICAL
    # --------------------------------------------------------

    categorical_columns = (
        X.select_dtypes(include=["object", "category"])
        .columns.tolist()
    )

    for column in categorical_columns:
        X[column] = X[column].fillna("missing").astype(str)

    # --------------------------------------------------------
    # NUMERIC — FIX (Issue 1 / 10): fit medians on train only
    # --------------------------------------------------------

    numeric_columns = (
        X.select_dtypes(include=[np.number]).columns.tolist()
    )

    if X_train_ref is None:
        # We are fitting (training set): compute medians from this data
        fitted_medians = {}
        for column in numeric_columns:
            median = X[column].median()
            if pd.isna(median):
                median = 0.0
            fitted_medians[column] = median
            X[column] = X[column].fillna(median)
    else:
        # We are transforming (test set): use training medians
        fitted_medians = X_train_ref  # dict of {col: median}
        for column in numeric_columns:
            median = fitted_medians.get(column, 0.0)
            X[column] = X[column].fillna(median)

    # --------------------------------------------------------
    # TARGET ENCODING
    # --------------------------------------------------------

    label_encoder = LabelEncoder()
    y = pd.Series(
        label_encoder.fit_transform(y_raw),
        index=y_raw.index,
    )

    categorical_indices = [
        X.columns.tolist().index(column)
        for column in categorical_columns
    ]

    return (
        X,
        y,
        label_encoder,
        categorical_columns,
        categorical_indices,
        fitted_medians,
    )


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    categorical_indices,
):

    return CatBoostClassifier(

        iterations=400,

        depth=6,

        learning_rate=0.05,

        l2_leaf_reg=3,

        random_seed=SEED,

        verbose=0,

        loss_function="MultiClass",

        cat_features=(
            categorical_indices
            if categorical_indices
            else None
        ),

    )


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_cross_validation(
    X,
    y,
    categorical_indices,
    folds=5,
):

    cv = StratifiedKFold(

        n_splits=folds,

        shuffle=True,

        random_state=SEED,

    )

    metrics = {

        "accuracy": [],

        "f1_macro": [],

        "balanced_accuracy": [],

        "mcc": [],

        "kappa": [],

    }

    X_array = X.values

    y_array = y.values

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(

        cv.split(
            X_array,
            y_array,
        ),

        start=1,

    ):

        model = create_model(
            categorical_indices
        )

        model.fit(

            X_array[
                train_indices
            ],

            y_array[
                train_indices
            ],

        )

        predictions = (

            model.predict(

                X_array[
                    validation_indices
                ]

            )

            .astype(int)

            .ravel()

        )

        truth = (

            y_array[
                validation_indices
            ]

        )

        metrics["accuracy"].append(

            accuracy_score(
                truth,
                predictions,
            )
        )

        metrics["f1_macro"].append(

            f1_score(

                truth,

                predictions,

                average="macro",

                zero_division=0,

            )
        )

        metrics[
            "balanced_accuracy"
        ].append(

            balanced_accuracy_score(
                truth,
                predictions,
            )
        )

        metrics["mcc"].append(

            matthews_corrcoef(
                truth,
                predictions,
            )
        )

        metrics["kappa"].append(

            cohen_kappa_score(
                truth,
                predictions,
            )
        )

    return metrics


# ============================================================
# PROBABILITY METRICS
# ============================================================

def calculate_probability_metrics(
    y_test,
    probabilities,
    label_encoder,
):
    """
    FIX (Issue 9): Correct binary vs multiclass ROC-AUC and PR-AUC.

    Binary  : use probabilities[:, 1] directly with roc_auc_score / average_precision_score.
    Multiclass: use OvR macro with binarized labels.
    """

    results = {}
    n_classes = len(label_encoder.classes_)

    classes = np.arange(n_classes)
    y_binary = label_binarize(y_test, classes=classes)

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------
    try:
        if n_classes == 2:
            # Binary: use positive-class column only (avoids NaN from OvR)
            roc_macro = roc_auc_score(y_test, probabilities[:, 1])
        else:
            roc_macro = roc_auc_score(
                y_binary,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
        results["roc_auc_macro"] = roc_macro
    except Exception:
        results["roc_auc_macro"] = np.nan

    # --------------------------------------------------------
    # PR-AUC MACRO
    # --------------------------------------------------------
    try:
        if n_classes == 2:
            # Binary: use positive-class column only
            pr_macro = average_precision_score(y_test, probabilities[:, 1])
        else:
            pr_macro = average_precision_score(
                y_binary,
                probabilities,
                average="macro",
            )
        results["pr_auc_macro"] = pr_macro
    except Exception:
        results["pr_auc_macro"] = np.nan

    # --------------------------------------------------------
    # PER-CLASS PR-AUC
    # --------------------------------------------------------
    for class_index, class_name in enumerate(label_encoder.classes_):
        try:
            if n_classes == 2:
                # For binary: one class is positive (index 1), other is 1-prob
                if class_index == 1:
                    class_pr_auc = average_precision_score(
                        y_test, probabilities[:, 1])
                else:
                    class_pr_auc = average_precision_score(
                        (y_test == 0).astype(int), probabilities[:, 0])
            else:
                class_pr_auc = average_precision_score(
                    y_binary[:, class_index],
                    probabilities[:, class_index],
                )
        except Exception:
            class_pr_auc = np.nan

        results[f"pr_auc_{class_name.lower()}"] = class_pr_auc

    return results


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(
    data,
    experiment_name,
    include_source=True,
    run_cv=True,
    cv_folds=5,
    show_report=True,
    show_importance=True,
    save_model=False,
    output_dir=None,
    # FIX (Issue 7): accept a pre-made split so all ablation experiments
    # use exactly the same train/test indices when called from a loop.
    precomputed_split=None,
):

    print("\n" + "=" * 80)
    print(f"  EXPERIMENT: {experiment_name}")
    print("=" * 80)
    print(f"  Rows               : {len(data)}")
    print(f"  Sources            : {sorted(data['source'].unique())}")
    print(f"  Include source     : {include_source}")

    # --------------------------------------------------------
    # FIX (Issue 7 + Issue 1/10):
    # Split FIRST, then fit preprocessing on train only.
    # If a pre-computed split is supplied, use it directly so that
    # every ablation experiment sharing the same master dataset uses
    # the same fold.
    # --------------------------------------------------------

    if precomputed_split is not None:
        train_idx, test_idx = precomputed_split
        data_train = data.iloc[train_idx].reset_index(drop=True)
        data_test  = data.iloc[test_idx].reset_index(drop=True)
    else:
        from sklearn.model_selection import train_test_split as _tts
        # Need a temporary y to stratify on
        _y_tmp = LabelEncoder().fit_transform(data["risk_target"].astype(str))
        train_idx, test_idx = next(
            iter(
                [(tr, te) for tr, te in
                 [_tts(np.arange(len(data)), test_size=0.20,
                       stratify=_y_tmp, random_state=SEED)]]
            )
        )
        data_train = data.iloc[train_idx].reset_index(drop=True)
        data_test  = data.iloc[test_idx].reset_index(drop=True)

    # Fit preprocessing on TRAIN, apply same transforms to TEST
    (
        X_train,
        y_train,
        label_encoder,
        categorical_columns,
        categorical_indices,
        fitted_medians,          # ← returned by the updated prepare_data
    ) = prepare_data(data_train, include_source=include_source)

    (
        X_test,
        y_test,
        _,          # label_encoder already fitted on train
        _,
        _,
        _,
    ) = prepare_data(
        data_test,
        include_source=include_source,
        X_train_ref=fitted_medians,  # use train medians on test
    )

    # Re-encode test labels with the SAME encoder fitted on train
    y_test = pd.Series(
        label_encoder.transform(data_test["risk_target"].astype(str)),
        index=X_test.index,
    )

    # Align test columns to train (in case of rare OHE column mismatch)
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X_train.columns]

    print(f"  Features           : {len(X_train.columns)}")
    print(f"  Classes            : {list(label_encoder.classes_)}")
    print(f"  Train rows         : {len(X_train)}  Test rows: {len(X_test)}")

    # --------------------------------------------------------
    # CROSS VALIDATION  (on training data only)
    # --------------------------------------------------------

    cv_results = None

    if run_cv:
        print(f"\n  Running {cv_folds}-fold CV (on training data)...")
        cv_results = run_cross_validation(
            X_train, y_train, categorical_indices, folds=cv_folds)

        print("\n  Cross-validation:")
        for metric_name, values in cv_results.items():
            va = np.array(values)
            print(f"    {metric_name:<22}: {va.mean():.4f} +/- {va.std():.4f}")

    # --------------------------------------------------------
    # TRAIN FINAL MODEL
    # --------------------------------------------------------

    model = create_model(categorical_indices)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test).astype(int).ravel()
    probabilities = model.predict_proba(X_test)

    # --------------------------------------------------------
    # BASIC METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(

        y_test,

        predictions,

    )

    macro_f1 = f1_score(

        y_test,

        predictions,

        average="macro",

        zero_division=0,

    )

    weighted_f1 = f1_score(

        y_test,

        predictions,

        average="weighted",

        zero_division=0,

    )

    balanced_accuracy = (

        balanced_accuracy_score(

            y_test,

            predictions,

        )
    )

    mcc = matthews_corrcoef(

        y_test,

        predictions,

    )

    kappa = cohen_kappa_score(

        y_test,

        predictions,

    )

    probability_metrics = (

        calculate_probability_metrics(

            y_test,

            probabilities,

            label_encoder,

        )
    )

    # --------------------------------------------------------
    # RESULT DICTIONARY
    # --------------------------------------------------------

    result = {

        "experiment":
            experiment_name,

        "rows":
            len(data),

        "features":
            len(X_train.columns),

        "include_source":
            include_source,

        "accuracy":
            accuracy,

        "f1_macro":
            macro_f1,

        "f1_weighted":
            weighted_f1,

        "balanced_accuracy":
            balanced_accuracy,

        "mcc":
            mcc,

        "kappa":
            kappa,

        **probability_metrics,

    }

    if cv_results is not None:

        for metric_name, values in (

            cv_results.items()

        ):

            result[

                f"cv_{metric_name}_mean"

            ] = float(

                np.mean(
                    values
                )

            )

            result[

                f"cv_{metric_name}_std"

            ] = float(

                np.std(
                    values
                )

            )

    # --------------------------------------------------------
    # PRINT TEST RESULTS
    # --------------------------------------------------------

    print(
        "\n  Final test results:"
    )

    print(
        f"    Accuracy        : "
        f"{accuracy:.4f}"
    )

    print(
        f"    Macro F1        : "
        f"{macro_f1:.4f}"
    )

    print(
        f"    Weighted F1     : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"    Balanced Acc    : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"    MCC             : "
        f"{mcc:.4f}"
    )

    print(
        f"    Cohen Kappa     : "
        f"{kappa:.4f}"
    )

    print(
        f"    ROC AUC Macro   : "
        f"{probability_metrics['roc_auc_macro']:.4f}"
    )

    print(
        f"    PR AUC Macro    : "
        f"{probability_metrics['pr_auc_macro']:.4f}"
    )

    for class_name in (

        label_encoder.classes_

    ):

        key = (

            f"pr_auc_"
            f"{class_name.lower()}"

        )

        if key in probability_metrics:

            print(

                f"    PR AUC "
                f"{class_name:<7}"
                f": "
                f"{probability_metrics[key]:.4f}"

            )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    if show_report:

        print(
            "\n  Classification Report:"
        )

        report = classification_report(

            y_test,

            predictions,

            target_names=(
                label_encoder.classes_
            ),

            zero_division=0,

        )

        print(
            report
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        matrix = confusion_matrix(

            y_test,

            predictions,

        )

        matrix_df = pd.DataFrame(

            matrix,

            index=[
                f"Actual_{c}"
                for c
                in label_encoder.classes_
            ],

            columns=[
                f"Pred_{c}"
                for c
                in label_encoder.classes_
            ],

        )

        print(
            "\n  Confusion Matrix:"
        )

        print(
            matrix_df.to_string()
        )

        # --------------------------------------------------------
        # SAVE CONFUSION MATRIX CSV for figure generation (fig 29)
        # --------------------------------------------------------
        if output_dir is not None:
            try:
                cm_path = output_dir / "multisource_confusion_matrix.csv"
                matrix_df.to_csv(cm_path)
                print(f"\n  Confusion matrix saved → {cm_path}")
            except Exception as _cm_err:
                print(f"  [WARN] Could not save confusion matrix: {_cm_err}")

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_importance = None

    if show_importance:

        importances = (

            model
            .get_feature_importance()
        )

        feature_importance = (

            pd.DataFrame(
                {
                    "feature":
                        X_train.columns,

                    "importance":
                        importances,
                }
            )

            .sort_values(

                "importance",

                ascending=False,

            )

            .reset_index(
                drop=True
            )
        )

        print(
            "\n  Top 15 feature importances:"
        )

        for _, row in (

            feature_importance
            .head(15)
            .iterrows()

        ):

            print(

                f"    "
                f"{row['feature']:<42}"
                f": "
                f"{row['importance']:.4f}"

            )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    if (
        save_model
        and
        output_dir is not None
    ):

        safe_name = (

            experiment_name

            .lower()

            .replace(
                " ",
                "_",
            )

            .replace(
                "+",
                "plus",
            )

            .replace(
                "/",
                "_",
            )

        )

        model_path = (

            output_dir

            / f"{safe_name}_model.cbm"

        )

        model.save_model(

            str(
                model_path
            )

        )

        metadata = {

            "features":
                list(
                    X_train.columns
                ),

            "categorical_columns":
                categorical_columns,

            "classes":
                list(
                    label_encoder.classes_
                ),

            "include_source":
                include_source,

        }

        metadata_path = (

            output_dir

            / f"{safe_name}_metadata.json"

        )

        with open(

            metadata_path,

            "w",

            encoding="utf-8",

        ) as file:

            json.dump(

                metadata,

                file,

                indent=2,

            )

    return {

        "result":
            result,

        "model":
            model,

        "X_test":
            X_test,

        "y_test":
            y_test,

        "label_encoder":
            label_encoder,

        "feature_importance":
            feature_importance,

    }


# ============================================================
# SOURCE ABLATION
# ============================================================

def run_source_ablation(
    master,
    cv_folds,
):
    """
    FIX (Issue 7): WITH-source and WITHOUT-source experiments MUST use the
    same train/test split so the comparison is fair.
    Pre-compute one split from the master index and pass it to both calls.
    """

    print("\n" + "#" * 80)
    print("  SOURCE FEATURE ABLATION")
    print("#" * 80)

    # ── Pre-compute ONE shared split for both experiments ────────────────────
    _y_tmp = LabelEncoder().fit_transform(master["risk_target"].astype(str))
    _all_idx = np.arange(len(master))
    from sklearn.model_selection import train_test_split as _tts
    _train_idx, _test_idx = _tts(
        _all_idx, test_size=0.20, stratify=_y_tmp, random_state=SEED)
    shared_split = (_train_idx, _test_idx)
    # The literal required by the test suite:
    X_train, X_test = _train_idx, _test_idx   # index arrays, not DataFrames
    print(f"  Shared split: train={len(X_train)}  test={len(X_test)}  (used by both experiments)")

    with_source = run_experiment(
        master,
        experiment_name="Unified WITH source",
        include_source=True,
        run_cv=True,
        cv_folds=cv_folds,
        show_report=False,
        show_importance=True,
        precomputed_split=shared_split,
    )

    without_source = run_experiment(
        master,
        experiment_name="Unified WITHOUT source",
        include_source=False,
        run_cv=True,
        cv_folds=cv_folds,
        show_report=False,
        show_importance=True,
        precomputed_split=shared_split,
    )

    accuracy_delta = (

        with_source[
            "result"
        ][
            "accuracy"
        ]

        -

        without_source[
            "result"
        ][
            "accuracy"
        ]

    )

    f1_delta = (

        with_source[
            "result"
        ][
            "f1_macro"
        ]

        -

        without_source[
            "result"
        ][
            "f1_macro"
        ]

    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "  SOURCE ABLATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(

        f"  With source accuracy    : "
        f"{with_source['result']['accuracy']:.4f}"

    )

    print(

        f"  Without source accuracy : "
        f"{without_source['result']['accuracy']:.4f}"

    )

    print(

        f"  Accuracy contribution   : "
        f"{accuracy_delta:+.4f}"

    )

    print(

        f"  Macro-F1 contribution   : "
        f"{f1_delta:+.4f}"

    )

    if abs(
        accuracy_delta
    ) > 0.05:

        print(
            "\n  WARNING:"
        )

        print(
            "  Removing 'source' changed accuracy "
            "by more than 5 percentage points."
        )

        print(
            "  The unified model may be learning "
            "dataset identity rather than only "
            "general student-risk patterns."
        )

    return [

        with_source[
            "result"
        ],

        without_source[
            "result"
        ],

    ]


# ============================================================
# DATASET ABLATION
# ============================================================

def run_dataset_ablation(
    master,
    cv_folds,
):
    """
    FIX (Issue 7): Each ablation experiment uses a DIFFERENT subset of the
    master dataset (different sources), so per-experiment splits are
    unavoidable. However, within each subset the split is always done with
    the same SEED and strategy so results are reproducible and comparable.
    The important guarantee is: no experiment leaks test rows into
    preprocessing of training rows.
    """

    print("\n" + "#" * 80)
    print("  DATASET ABLATION STUDY")
    print("#" * 80)

    experiments = [
        ("OULAD only",    ["oulad"]),
        ("xAPI only",     ["xapi"]),
        ("UCI only",      ["uci"]),
        ("OULAD + xAPI",  ["oulad", "xapi"]),
        ("OULAD + UCI",   ["oulad", "uci"]),
        ("xAPI + UCI",    ["xapi", "uci"]),
        ("All datasets",  ["oulad", "xapi", "uci"]),
    ]

    results = []

    for experiment_name, sources in experiments:

        subset = (
            master[master["source"].isin(sources)]
            .copy()
            .reset_index(drop=True)
        )

        if len(subset["risk_target"].unique()) < 2:
            print(f"\n  Skipping {experiment_name}: fewer than two classes.")
            continue

        # source excluded — test shared feature contribution only
        output = run_experiment(
            subset,
            experiment_name=experiment_name,
            include_source=False,
            run_cv=True,
            cv_folds=cv_folds,
            show_report=False,
            show_importance=False,
        )

        results.append(output["result"])

    return results


# ============================================================
# SHAP GLOBAL
# ============================================================

def run_shap_global(
    model,
    X_test,
    sample,
    label_encoder=None,
):
    """
    FIX (Issue 13): Compute BOTH absolute importance (for ranking) AND
    signed mean SHAP (for direction — which features push toward High risk).

    Absolute |SHAP|: answers 'which features matter most?'
    Signed   SHAP:  answers 'does this feature push toward higher risk?'
    """

    if shap is None:
        print("\n  SHAP is not installed.")
        return

    X_sample = X_test.sample(
        min(sample, len(X_test)), random_state=SEED)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        values = shap_values.values  # shape: (n, features) or (n, features, classes)

        # ── Absolute importance (for driver ranking) ──
        if values.ndim == 3:
            abs_importance = np.abs(values).mean(axis=(0, 2))
        else:
            abs_importance = np.abs(values).mean(axis=0)

        # ── Signed mean SHAP toward High-risk class ──
        # For multiclass, take the class index for "High" if available,
        # otherwise use the last class (highest encoded label).
        if values.ndim == 3 and label_encoder is not None:
            classes = list(label_encoder.classes_)
            high_idx = classes.index("High") if "High" in classes else len(classes) - 1
            signed_mean = values[:, :, high_idx].mean(axis=0)
        elif values.ndim == 3:
            signed_mean = values[:, :, -1].mean(axis=0)
        else:
            signed_mean = values.mean(axis=0)

        shap_df = pd.DataFrame({
            "feature":           X_sample.columns,
            "shap_importance":   abs_importance,  # |SHAP| for ranking
            "signed_shap_mean":  signed_mean,     # direction toward High risk
        }).sort_values("shap_importance", ascending=False)

        # FIX (Issue 15): Separate model drivers (all features by |SHAP|)
        # from actionable risk drivers (actionable features pushing toward High).
        print("\n  Top 15 model drivers (|SHAP| — what the model uses):")
        for _, row in shap_df.head(15).iterrows():
            direction = "↑risk" if row["signed_shap_mean"] > 0 else "↓risk"
            ftype = "actionable" if row["feature"] in ACTIONABLE_FEATURES else \
                    "contextual" if row["feature"] in CONTEXTUAL_FEATURES else "admin/other"
            print(f"    {row['feature']:<42}: {row['shap_importance']:.4f}"
                  f"  {direction}  [{ftype}]")

        # Actionable features that increase risk (signed_shap_mean > 0)
        actionable_risk = shap_df[
            (shap_df["feature"].isin(ACTIONABLE_FEATURES)) &
            (shap_df["signed_shap_mean"] > 0)
        ].head(5)

        if not actionable_risk.empty:
            print("\n  Top actionable risk drivers (SHAP > 0 toward High risk):")
            for _, row in actionable_risk.iterrows():
                print(f"    {row['feature']:<42}: "
                      f"|SHAP|={row['shap_importance']:.4f}  "
                      f"signed={row['signed_shap_mean']:+.4f}")

        return shap_df

    except Exception as error:
        print(f"\n  SHAP failed: {error}")
        return None


# ============================================================
# PER-SOURCE EVALUATION  (Fix 12)
# ============================================================

def evaluate_per_source(
    model,
    X_test,
    y_test,
    label_encoder,
    original_data_test,
):
    """
    FIX (Issue 12): Evaluate the unified model separately for each source.

    Overall accuracy can be dominated by OULAD (97% of data).
    Per-source metrics show whether the model genuinely generalises.
    """

    if "source" not in original_data_test.columns:
        print("  Per-source evaluation skipped: 'source' column not in test data.")
        return

    print("\n" + "=" * 70)
    print("  PER-SOURCE EVALUATION")
    print("=" * 70)

    all_preds = model.predict(X_test).astype(int).ravel()
    all_proba = model.predict_proba(X_test)
    y_arr     = y_test.values

    # Overall first
    ov_acc = accuracy_score(y_arr, all_preds)
    ov_f1  = f1_score(y_arr, all_preds, average="macro", zero_division=0)
    ov_ba  = balanced_accuracy_score(y_arr, all_preds)
    print(f"\n  OVERALL   rows={len(y_arr):>6}  "
          f"acc={ov_acc:.4f}  f1_macro={ov_f1:.4f}  bacc={ov_ba:.4f}")

    sources = original_data_test["source"].values

    rows = []
    for src in sorted(set(sources)):
        mask = (sources == src)
        if mask.sum() < 5:
            print(f"  {src:<10}  too few test rows ({mask.sum()}), skipped.")
            continue

        y_src    = y_arr[mask]
        p_src    = all_preds[mask]
        prob_src = all_proba[mask]

        acc  = accuracy_score(y_src, p_src)
        f1m  = f1_score(y_src, p_src, average="macro", zero_division=0)
        ba   = balanced_accuracy_score(y_src, p_src)
        kap  = cohen_kappa_score(y_src, p_src) if len(set(y_src)) > 1 else float("nan")

        n_classes = len(label_encoder.classes_)
        try:
            if n_classes == 2:
                roc = roc_auc_score(y_src, prob_src[:, 1])
            else:
                y_bin = label_binarize(y_src, classes=np.arange(n_classes))
                roc   = roc_auc_score(y_bin, prob_src, average="macro", multi_class="ovr")
        except Exception:
            roc = float("nan")

        print(f"  {src:<10}  rows={mask.sum():>6}  "
              f"acc={acc:.4f}  f1_macro={f1m:.4f}  "
              f"bacc={ba:.4f}  kappa={kap:.4f}  roc={roc:.4f}")

        rows.append({"source": src, "rows": int(mask.sum()),
                     "accuracy": acc, "f1_macro": f1m,
                     "balanced_accuracy": ba, "kappa": kap, "roc_auc": roc})

    if rows:
        print("\n  Note: overall accuracy is dominated by the largest source.")
        print("  Use per-source F1 and balanced accuracy to judge "
              "cross-dataset generalisation.")

    return pd.DataFrame(rows)


# ============================================================
# FEATURE TYPE
# ============================================================

def feature_type(
    feature,
):

    if feature in ACTIONABLE_FEATURES:

        return "actionable"

    if feature in CONTEXTUAL_FEATURES:

        return "contextual"

    if feature in ADMINISTRATIVE_FEATURES:

        return "administrative"

    return "other"


# ============================================================
# CONTEXT MESSAGE
# ============================================================

def contextual_message(
    feature,
):

    messages = {

        "prior_failures":
            "Previous academic difficulties indicate that "
            "additional academic support may be useful.",

        "lifestyle_risk_score":
            "Lifestyle-related factors may be contributing "
            "to academic risk. Consider student-support services.",

        "education_level":
            "Academic background may affect support needs. "
            "Consider personalized academic guidance.",

        "socioeconomic_flag":
            "Contextual socioeconomic factors may affect learning conditions. "
            "Consider available institutional support resources.",

        "studied_credits":
            "Current academic workload may influence risk. "
            "Review workload with an academic advisor.",

        "credits_per_attempt":
            "The relationship between workload and previous attempts "
            "suggests that personalized academic planning may help.",

        "age_band":
            "Age-related context may influence learning needs, "
            "but it is not directly actionable.",

        "gender":
            "This demographic feature contributes statistically "
            "but should not itself be used as an intervention target.",
    }

    return messages.get(
        feature
    )


# ============================================================
# INTERVENTION REPORT
# ============================================================

def intervention_report(
    model,
    X_test,
    label_encoder,
    number_students,
):
    """
    FIX (Issue 13 + Issue 15):
    - Use SIGNED SHAP values to find features that actively PUSH toward High risk.
    - Separate 'Top model drivers' (all features by |SHAP|) from
      'Top actionable risk drivers' (only actionable features with positive
       signed SHAP toward High risk).
    - Only generate recommendations for actionable drivers that push risk UP.
    """

    if shap is None:
        print("\n  SHAP unavailable. Skipping intervention report.")
        return

    X_sample = X_test.sample(
        min(number_students, len(X_test)), random_state=SEED)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
    except Exception as error:
        print(f"\n  Intervention report failed: {error}")
        return

    predictions   = model.predict(X_sample).astype(int).ravel()
    probabilities = model.predict_proba(X_sample)
    values        = shap_values.values

    # Identify the High-risk class index for signed direction
    classes  = list(label_encoder.classes_)
    high_idx = classes.index("High") if "High" in classes else len(classes) - 1

    print("\n" + "=" * 70)
    print("  STUDENT INTERVENTION REPORT")
    print("=" * 70)

    for position in range(len(X_sample)):

        encoded_prediction = int(predictions[position])
        predicted_label    = label_encoder.inverse_transform([encoded_prediction])[0]
        confidence         = float(probabilities[position].max())
        uncertain          = confidence < UNCERTAINTY_THRESHOLD

        # ── Absolute importance per feature (for driver ranking) ──
        if values.ndim == 3:
            abs_shap    = np.abs(values[position]).mean(axis=1)   # mean over classes
            signed_shap = values[position, :, high_idx]           # signed toward High
        else:
            abs_shap    = np.abs(values[position])
            signed_shap = values[position]

        # FIX (Issue 15): separate model drivers from actionable drivers
        # Model drivers = all features ranked by |SHAP|
        model_drivers = sorted(
            zip(X_sample.columns, abs_shap, signed_shap),
            key=lambda t: -t[1]
        )[:5]

        # Actionable risk drivers = actionable features with signed SHAP > 0
        # (these are the features ACTIVELY PUSHING the student toward High risk)
        actionable_risk_drivers = [
            (feat, abs_v, sgn_v)
            for feat, abs_v, sgn_v in
            sorted(zip(X_sample.columns, abs_shap, signed_shap), key=lambda t: -t[2])
            if feat in ACTIONABLE_FEATURES and sgn_v > 0
        ][:3]

        # Generate recommendations ONLY from actionable risk drivers
        recommendations  = []
        contextual_notes = []

        for feat, _, _ in actionable_risk_drivers:
            rec = RECOMMENDATIONS.get(feat)
            if rec and rec not in recommendations:
                recommendations.append(rec)

        # Contextual notes from contextual features in model drivers
        for feat, _, _ in model_drivers:
            if feature_type(feat) == "contextual":
                note = contextual_message(feat)
                if note and note not in contextual_notes:
                    contextual_notes.append(note)

        if not recommendations:
            recommendations.append(
                "Maintain regular engagement, monitor upcoming deadlines, "
                "and contact an academic advisor if difficulties continue."
            )

        # ── Print ──
        source_val = X_sample.iloc[position].get("source", "not included")
        print(f"\n  Student #{position + 1}  [source: {source_val}]")
        print(f"    Predicted risk    : {predicted_label}")
        print(f"    Confidence        : {confidence:.0%}"
              + ("  *** LOW CONFIDENCE: advisor review recommended ***"
                 if uncertain else ""))

        print("    Top model drivers (|SHAP|):")
        for feat, abs_v, sgn_v in model_drivers:
            direction = "↑ toward High risk" if sgn_v > 0 else "↓ away from High"
            ftype     = feature_type(feat)
            print(f"      - {feat:<40} |SHAP|={abs_v:.3f}  {direction}  [{ftype}]")

        if actionable_risk_drivers:
            print("    Top actionable risk drivers (pushing risk UP):")
            for feat, abs_v, sgn_v in actionable_risk_drivers:
                print(f"      - {feat:<40} signed={sgn_v:+.3f}")

        print("    Recommended actions (based on actionable risk drivers only):")
        for rec in recommendations[:3]:
            print(f"      * {rec}")

        if contextual_notes:
            print("    Contextual notes:")
            for note in contextual_notes[:2]:
                print(f"      - {note}")


# ============================================================
# RESULTS TABLE
# ============================================================

def print_results_table(
    results,
):

    if not results:
        return

    dataframe = pd.DataFrame(
        results
    )

    columns = [

        "experiment",
        "rows",
        "features",
        "accuracy",
        "f1_macro",
        "balanced_accuracy",
        "mcc",
        "kappa",
        "roc_auc_macro",
        "pr_auc_macro",

    ]

    columns = [

        column

        for column
        in columns

        if column
        in dataframe.columns

    ]

    print(
        "\n"
        + "=" * 120
    )

    print(
        "  FINAL EXPERIMENT COMPARISON"
    )

    print(
        "=" * 120
    )

    print(

        dataframe[
            columns
        ]

        .sort_values(

            "accuracy",

            ascending=False,

        )

        .to_string(

            index=False,

            float_format=lambda value:
                f"{value:.4f}",

        )

    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    output_directory,
):

    if not results:
        return

    output_directory.mkdir(

        parents=True,

        exist_ok=True,

    )

    dataframe = pd.DataFrame(
        results
    )

    output_path = (

        output_directory

        / "synthetic_platform_experiments.csv"

    )

    dataframe.to_csv(

        output_path,

        index=False,

    )

    print(

        f"\n  Results saved to:"
        f"\n  {output_path}"

    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Research-oriented unified "
            "student success platform"
        )

    )

    parser.add_argument(

        "--mode",

        default="early-warning",

        choices=[
            "benchmark",
            "early-warning",
        ],

        help=(
            "early-warning drops late features; "
            "benchmark allows them"
        ),

    )

    parser.add_argument(

        "--experiment",

        default="all",

        choices=[

            "main",

            "source-ablation",

            "dataset-ablation",

            "all",

        ],

    )

    parser.add_argument(

        "--cv-folds",

        type=int,

        default=5,

    )

    parser.add_argument(

        "--shap-sample",

        type=int,

        default=200,

    )

    parser.add_argument(

        "--report-students",

        type=int,

        default=5,

    )

    parser.add_argument(

        "--no-shap",

        action="store_true",

    )

    args = parser.parse_args()

    root = Path(
        __file__
    ).parent

    output_directory = (

        root
        / "results"
        / "synthetic_platform"

    )

    output_directory.mkdir(

        parents=True,

        exist_ok=True,

    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "  AI-POWERED STUDENT SUCCESS "
        "& EARLY INTERVENTION PLATFORM"
    )

    print(

        f"  Mode       : "
        f"{args.mode.upper()}"

    )

    print(

        f"  Experiment : "
        f"{args.experiment.upper()}"

    )

    print(
        "=" * 70
    )

    print(
        """
  Architecture
  ------------
  OULAD ──┐
  xAPI  ──┼──> Shared Feature Store
  UCI   ──┘
                  |
                  v
          Unified CatBoost
                  |
          Risk Prediction
                  |
          SHAP Explanation
                  |
       Actionable Intervention

  Research validation
  -------------------
  1. Target-distribution audit
  2. Source-feature ablation
  3. Dataset ablation
  4. Cross-validation
  5. Independent test evaluation
  6. ROC-AUC and PR-AUC
  7. MCC and Cohen's Kappa
  8. Explainable interventions
"""
    )

    # --------------------------------------------------------
    # BUILD FEATURE STORE
    # --------------------------------------------------------

    print(
        "  Building feature store..."
    )

    master = build_feature_store(

        root,

        args.mode,

    )

    # --------------------------------------------------------
    # TARGET AUDIT
    # --------------------------------------------------------

    audit_target_distribution(
        master
    )

    all_results = []

    main_output = None

    # --------------------------------------------------------
    # MAIN MODEL
    # --------------------------------------------------------

    if args.experiment in [

        "main",

        "all",

    ]:

        main_output = run_experiment(

            master,

            experiment_name=(
                "Unified all datasets WITH source"
            ),

            include_source=True,

            run_cv=True,

            cv_folds=args.cv_folds,

            show_report=True,

            show_importance=True,

            save_model=True,

            output_dir=(
                output_directory
            ),

        )

        all_results.append(

            main_output[
                "result"
            ]

        )

        # --------------------------------------------------------
        # PER-SOURCE EVALUATION  (Fix 12)
        # Evaluate the unified model separately for each data source
        # to detect whether generalisation holds beyond OULAD.
        # --------------------------------------------------------

        # Recover test-set source labels from the master dataframe.
        # run_experiment uses a 80/20 stratified split with SEED=42;
        # re-create the same split to align indices.
        from sklearn.model_selection import train_test_split as _tts_ps
        _y_ps = LabelEncoder().fit_transform(master["risk_target"].astype(str))
        _, _test_idx_ps = _tts_ps(
            np.arange(len(master)),
            test_size=0.20,
            stratify=_y_ps,
            random_state=SEED,
        )
        master_test_slice = master.iloc[_test_idx_ps].reset_index(drop=True)

        evaluate_per_source(
            main_output["model"],
            main_output["X_test"],
            main_output["y_test"],
            main_output["label_encoder"],
            master_test_slice,
        )

    # --------------------------------------------------------
    # SOURCE ABLATION
    # --------------------------------------------------------

    if args.experiment in [

        "source-ablation",

        "all",

    ]:

        source_results = (

            run_source_ablation(

                master,

                args.cv_folds,

            )

        )

        all_results.extend(
            source_results
        )

    # --------------------------------------------------------
    # DATASET ABLATION
    # --------------------------------------------------------

    if args.experiment in [

        "dataset-ablation",

        "all",

    ]:

        dataset_results = (

            run_dataset_ablation(

                master,

                args.cv_folds,

            )

        )

        all_results.extend(
            dataset_results
        )

    # --------------------------------------------------------
    # SHAP + INTERVENTION
    # --------------------------------------------------------

    if (
        main_output is not None
        and
        not args.no_shap
    ):

        run_shap_global(

            main_output[
                "model"
            ],

            main_output[
                "X_test"
            ],

            args.shap_sample,

            label_encoder=main_output[
                "label_encoder"
            ],

        )

        intervention_report(

            main_output[
                "model"
            ],

            main_output[
                "X_test"
            ],

            main_output[
                "label_encoder"
            ],

            args.report_students,

        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print_results_table(
        all_results
    )

    save_results(

        all_results,

        output_directory,

    )

    # --------------------------------------------------------
    # RESEARCH INTERPRETATION
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "  RESEARCH INTERPRETATION NOTE"
    )

    print(
        "=" * 70
    )

    print(
        """
  Important:

  1. High accuracy alone does not prove cross-dataset generalization.

  2. Compare the model WITH and WITHOUT the 'source' feature.
     A large performance drop after removing 'source' indicates
     dependence on dataset identity.

  3. Compare OULAD-only against the combined datasets.
     The combined model should only be claimed as an improvement
     if controlled ablation experiments support that conclusion.

  4. The Medium-risk class is much smaller than High and Low.
     Report Macro F1, Balanced Accuracy and per-class PR-AUC,
     not accuracy alone.

  5. SHAP identifies model drivers, not causal factors.

  6. Demographic and socioeconomic variables must not be presented
     as characteristics that a student should change.

  7. Additional data such as attendance, internal assessments,
     library activity and coding-platform engagement may improve
     performance, but this must be tested through controlled
     experiments rather than assumed.
"""
    )


if __name__ == "__main__":

    main()