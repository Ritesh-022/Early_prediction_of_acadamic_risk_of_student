#!/usr/bin/env python3
"""AI-Powered Student Success & Early Intervention Platform

Architecture
------------
Three public datasets are treated as separate data sources feeding ONE unified
feature store. Each source is mapped to a shared schema, then vertically stacked
into a single master table. One CatBoost model trains on that table.

  OULAD  ──┐
  xAPI   ──┼──> build_feature_store() ──> master_table ──> CatBoost ──> Risk + Report
  UCI    ──┘

Shared schema (all sources contribute what they have; missing = NaN):
  Engagement  : total_engagement, engagement_trend, engagement_consistency,
                weekly_activity_change, consistency_ratio
  Assessment  : score_improvement, first_assessment_day, late_submission_count
  Background  : prior_failures, study_time, absence_flag, lifestyle_risk_score
  Demographics: gender, age_band, education_level, socioeconomic_flag
  Metadata    : source (which dataset), risk_target (unified 3-level label)

Usage:
    python synthetic_platform.py
    python synthetic_platform.py --mode benchmark
    python synthetic_platform.py --shap-sample 300 --report-students 5
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, label_binarize

try:
    import shap
except ImportError:
    shap = None

# ── FIX: force UTF-8 stdout/stderr to prevent cp1252 UnicodeEncodeError ──────
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SEED = 42
np.random.seed(SEED)

LATE_FEATURES = {
    'assessment_completion_ratio', 'last_ts', 'last_assessment_day',
    'assessment_span_days', 'inactivity_days', 'num_assessments',
    'missed_assessments', 'total_assessments', 'week_click_sum_1_12',
    'click_growth_rate', 'click_variance', 'longest_inactive_gap',
    'avg_score', 'score_std', 'assessment_score_trend',
}

# Unified risk label mapping — every source maps its target to Low/Medium/High
#
# FIX 6 — TARGET HARMONIZATION NOTE:
# OULAD has no "Medium" risk class: Pass→Low, Fail→High, Withdrawn→High, Distinction→Low
# xAPI provides Medium (class M). UCI provides Medium (grade Mid).
# This creates a structural imbalance: OULAD (~97% of all rows) never contributes
# Medium samples, so the unified model must be evaluated carefully.
#
# Valid options:
#   Option A (current): Keep Low/Medium/High — report Medium as under-represented WARNING.
#   Option B (binary):  Map all sources to Low/High only — eliminates the imbalance.
#
# WARNING: The Medium-risk class represents <2% of all records (xAPI+UCI only).
# Do NOT use SMOTE to inflate Medium from OULAD rows — that would fabricate labels.
# Always report Macro F1 and per-class PR-AUC, not accuracy alone.
RISK_LABEL = {
    # OULAD final_result
    'Distinction': 'Low', 'Pass': 'Low', 'Fail': 'High', 'Withdrawn': 'High',
    # xAPI Class
    'H': 'Low', 'M': 'Medium', 'L': 'High',
    # UCI grade_band
    'High': 'Low', 'Mid': 'Medium', 'Low': 'High',
}

# Option B: Binary mapping (use with --binary flag)
RISK_LABEL_BINARY = {
    'Distinction': 'Low', 'Pass': 'Low', 'Fail': 'High', 'Withdrawn': 'High',
    'H': 'Low', 'M': 'Low', 'L': 'High',
    'High': 'Low', 'Mid': 'Low', 'Low': 'High',
}

# FIX 15 — ACTIONABLE vs MODEL DRIVERS:
# Not all top-SHAP features should generate recommendations.
# Split into: actionable (student can change) vs contextual (background).
ACTIONABLE_FEATURES = {
    'total_engagement', 'engagement_trend', 'engagement_consistency',
    'weekly_activity_change', 'consistency_ratio',
    'score_improvement', 'first_assessment_day',
    'late_submission_count', 'study_time', 'absence_flag',
    'registration_early_days',
}
CONTEXTUAL_FEATURES = {
    'prior_failures', 'lifestyle_risk_score', 'gender', 'age_band',
    'education_level', 'socioeconomic_flag', 'studied_credits',
    'credits_per_attempt', 'code_module',
}

# Confidence threshold below which a prediction is flagged as uncertain
UNCERTAINTY_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# BEHAVIORAL FEATURE HELPERS
# ---------------------------------------------------------------------------

def _slope(arr: np.ndarray) -> float:
    if len(arr) < 2 or np.all(arr == 0):
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def _consistency(arr: np.ndarray) -> float:
    """1 - CV; higher = more consistent engagement."""
    if len(arr) == 0 or arr.mean() == 0:
        return 0.0
    return float(max(0.0, 1.0 - arr.std() / arr.mean()))


def _weekly_activity_change(arr: np.ndarray) -> float:
    """Mean week-over-week % change in activity."""
    if len(arr) < 2:
        return 0.0
    diffs = np.diff(arr.astype(float))
    prev = arr[:-1].astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.where(prev > 0, diffs / prev, 0.0)
    return float(pct.mean())


def _consistency_ratio(arr: np.ndarray) -> float:
    """Fraction of periods with non-zero activity."""
    if len(arr) == 0:
        return 0.0
    return float((arr > 0).mean())


def _score_improvement(arr: np.ndarray) -> float:
    """Last score minus first score; NaN if fewer than 2 scores."""
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return np.nan
    return float(valid[-1] - valid[0])


# ---------------------------------------------------------------------------
# SOURCE LOADERS — each returns a DataFrame in the shared schema
# ---------------------------------------------------------------------------

def _from_oulad(path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates()

    # Drop leakage / redundant columns
    drop_always = [
        'date_unregistration', 'date_unreg', 'date_unregistered',
        'weighted_score', 'id_student', 'id_assessment', 'id_site',
        'first_ts', 'active_weeks', 'clicks_per_active_week',
        'assessments_per_week', 'activity_count', 'days_active',
        'avg_clicks_per_day', 'week_click_sum_1_4', 'registration_delay_category',
    ]
    df = df.drop(columns=[c for c in drop_always if c in df.columns])
    if mode == 'early-warning':
        df = df.drop(columns=[c for c in df.columns if c in LATE_FEATURES])

    week_cols = [f'week{w}_clicks' for w in range(1, 9) if f'week{w}_clicks' in df.columns]
    week_arr = df[week_cols].fillna(0).values if week_cols else None

    out = pd.DataFrame(index=df.index)
    out['source'] = 'oulad'

    # Engagement
    if week_arr is not None:
        out['total_engagement']        = week_arr.sum(axis=1)
        out['engagement_trend']        = [_slope(r) for r in week_arr]
        out['engagement_consistency']  = [_consistency(r) for r in week_arr]
        out['weekly_activity_change']  = [_weekly_activity_change(r) for r in week_arr]
        out['consistency_ratio']       = [_consistency_ratio(r) for r in week_arr]
    else:
        for c in ['total_engagement', 'engagement_trend', 'engagement_consistency',
                  'weekly_activity_change', 'consistency_ratio']:
            out[c] = np.nan

    # Assessment
    out['score_improvement']    = np.nan   # not available per-student in week-cutoff table
    out['first_assessment_day'] = df.get('first_assessment_day', pd.Series(np.nan, index=df.index))
    out['late_submission_count']= df.get('late_submission_count', pd.Series(0, index=df.index))

    # Background
    out['prior_failures']       = df.get('num_of_prev_attempts', pd.Series(np.nan, index=df.index))
    out['study_time']           = np.nan
    out['absence_flag']         = np.nan
    out['lifestyle_risk_score'] = np.nan

    # Demographics
    gender_map = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0}
    out['gender']            = df.get('gender', pd.Series(dtype=str)).map(gender_map)
    out['age_band']          = df.get('age_band', pd.Series(dtype=str)).astype(str)
    out['education_level']   = df.get('highest_education', pd.Series(dtype=str)).astype(str)
    out['socioeconomic_flag']= df.get('imd_band', pd.Series(dtype=str)).astype(str)

    # Extra OULAD-specific features worth keeping
    out['clicks_per_credit']  = df.get('clicks_per_credit', pd.Series(np.nan, index=df.index))
    out['credits_per_attempt']= df.get('credits_per_attempt', pd.Series(np.nan, index=df.index))
    out['studied_credits']    = df.get('studied_credits', pd.Series(np.nan, index=df.index))
    out['registration_early_days'] = df.get('registration_early_days', pd.Series(np.nan, index=df.index))
    out['code_module']        = df.get('code_module', pd.Series(dtype=str)).astype(str)

    # Target
    out['original_target'] = df['final_result'].values
    out['risk_target']     = out['original_target'].map(RISK_LABEL)
    out.index = df.index
    return out.dropna(subset=['risk_target'])


def _from_xapi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={
        'raisedhands': 'raised_hands',
        'visitedresources': 'visited_resources',
        'announcementsview': 'announcements_view',
        'studentabsencedays': 'absence_days',
        'parentansweringsurvey': 'parent_survey',
        'parentschoolsatisfaction': 'parent_satisfaction',
    })

    eng_cols = [c for c in ['raised_hands', 'visited_resources',
                             'announcements_view', 'discussion'] if c in df.columns]
    eng_arr = df[eng_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values \
              if eng_cols else None

    out = pd.DataFrame(index=df.index)
    out['source'] = 'xapi'

    if eng_arr is not None:
        out['total_engagement']       = eng_arr.sum(axis=1)
        out['engagement_trend']       = [_slope(r) for r in eng_arr]
        out['engagement_consistency'] = [_consistency(r) for r in eng_arr]
        out['weekly_activity_change'] = [_weekly_activity_change(r) for r in eng_arr]
        out['consistency_ratio']      = [_consistency_ratio(r) for r in eng_arr]
    else:
        for c in ['total_engagement', 'engagement_trend', 'engagement_consistency',
                  'weekly_activity_change', 'consistency_ratio']:
            out[c] = np.nan

    out['score_improvement']     = np.nan
    out['first_assessment_day']  = np.nan
    out['late_submission_count'] = np.nan
    out['prior_failures']        = np.nan
    out['study_time']            = np.nan

    if 'absence_days' in df.columns:
        out['absence_flag'] = (df['absence_days'].astype(str).str.lower() == 'above-7').astype(float)
    else:
        out['absence_flag'] = np.nan

    out['lifestyle_risk_score'] = np.nan

    gender_map = {'m': 1, 'f': 0}
    out['gender']            = df.get('gender', pd.Series(dtype=str)).str.lower().map(gender_map)
    out['age_band']          = 'unknown'
    out['education_level']   = df.get('stageid', pd.Series(dtype=str)).astype(str)
    out['socioeconomic_flag']= 'unknown'

    out['clicks_per_credit']       = np.nan
    out['credits_per_attempt']     = np.nan
    out['studied_credits']         = np.nan
    out['registration_early_days'] = np.nan
    out['code_module']             = df.get('topic', pd.Series(dtype=str)).astype(str)

    out['original_target'] = df['class'].values
    out['risk_target']     = out['original_target'].map(RISK_LABEL)
    return out.dropna(subset=['risk_target'])


def _from_uci(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';').drop_duplicates()
    df.columns = [c.strip().lower() for c in df.columns]

    if 'g3' not in df.columns:
        return pd.DataFrame()

    # Grade band target
    df['grade_band'] = pd.cut(df['g3'], bins=[-1, 9, 14, 20],
                               labels=['Low', 'Mid', 'High']).astype(str)

    # Score trajectory from G1, G2
    grade_cols = [c for c in ['g1', 'g2'] if c in df.columns]
    grade_arr = df[grade_cols].apply(pd.to_numeric, errors='coerce').values \
                if grade_cols else None

    out = pd.DataFrame(index=df.index)
    out['source'] = 'uci'

    # Map study time to engagement proxy (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)
    study = pd.to_numeric(df.get('studytime', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
    out['total_engagement']       = study * 10   # scale to comparable range
    out['engagement_trend']       = np.nan
    out['engagement_consistency'] = np.nan
    out['weekly_activity_change'] = np.nan
    out['consistency_ratio']      = np.nan

    if grade_arr is not None:
        out['score_improvement'] = [_score_improvement(r) for r in grade_arr]
    else:
        out['score_improvement'] = np.nan

    out['first_assessment_day']  = np.nan
    out['late_submission_count'] = np.nan
    out['prior_failures']        = pd.to_numeric(df.get('failures', pd.Series(0, index=df.index)), errors='coerce')
    out['study_time']            = study

    absences = pd.to_numeric(df.get('absences', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
    out['absence_flag'] = (absences > 10).astype(float)

    # Lifestyle risk: normalised mean of dalc, walc, absences, failures
    risk_cols = [c for c in ['dalc', 'walc', 'absences', 'failures'] if c in df.columns]
    if risk_cols:
        normed = df[risk_cols].apply(pd.to_numeric, errors='coerce').apply(
            lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9))
        out['lifestyle_risk_score'] = normed.mean(axis=1)
    else:
        out['lifestyle_risk_score'] = np.nan

    gender_map = {'m': 1, 'f': 0}
    out['gender']            = df.get('sex', pd.Series(dtype=str)).str.lower().map(gender_map)
    out['age_band']          = pd.cut(
        pd.to_numeric(df.get('age', pd.Series(dtype=float)), errors='coerce'),
        bins=[0, 17, 22, 100], labels=['<=17', '18-22', '23+']).astype(str)
    out['education_level']   = df.get('medu', pd.Series(dtype=str)).astype(str)
    out['socioeconomic_flag']= df.get('address', pd.Series(dtype=str)).astype(str)

    out['clicks_per_credit']       = np.nan
    out['credits_per_attempt']     = np.nan
    out['studied_credits']         = np.nan
    out['registration_early_days'] = np.nan
    out['code_module']             = df.get('reason', pd.Series(dtype=str)).astype(str)

    # Drop raw grade columns — they would be leakage
    out['original_target'] = df['grade_band'].values
    out['risk_target']     = out['original_target'].map(RISK_LABEL)
    return out.dropna(subset=['risk_target'])


# ---------------------------------------------------------------------------
# UNIFIED FEATURE STORE
# ---------------------------------------------------------------------------

def build_feature_store(root: Path, mode: str) -> pd.DataFrame:
    """Load all available sources and stack into one master feature table."""
    parts = []

    oulad_path = root / 'oulad_ml_table_week8.csv'
    if not oulad_path.exists():
        oulad_path = root / 'oulad_ml_table.csv'
    if oulad_path.exists():
        part = _from_oulad(oulad_path, mode)
        parts.append(part)
        print(f"  OULAD loaded   : {len(part):>6} rows")
    else:
        print("  OULAD skipped  : oulad_ml_table_week8.csv not found")

    xapi_path = root / 'xAPI' / 'xAPI-Edu-Data.csv'
    if xapi_path.exists():
        part = _from_xapi(xapi_path)
        parts.append(part)
        print(f"  xAPI loaded    : {len(part):>6} rows")
    else:
        print("  xAPI skipped   : xAPI-Edu-Data.csv not found")

    uci_path = root / 'UI_student+performance' / 'student' / 'student-mat.csv'
    if uci_path.exists():
        part = _from_uci(uci_path)
        parts.append(part)
        print(f"  UCI loaded     : {len(part):>6} rows")
    else:
        print("  UCI skipped    : student-mat.csv not found")

    if not parts:
        raise RuntimeError("No data sources found.")

    master = pd.concat(parts, ignore_index=True)
    print(f"  Master table   : {len(master):>6} rows  |  {len(master.columns)} columns")
    return master


# ---------------------------------------------------------------------------
# TRAINING — one model on the unified table
# ---------------------------------------------------------------------------

def train_unified(master: pd.DataFrame, shap_sample: int, report_n: int,
                  include_source: bool = False):
    """
    Train the unified CatBoost model.

    FIX 8: include_source=False by default — source is excluded from the
    final predictive model. It is used only for per-source evaluation.
    Use include_source=True only for ablation experiments.

    FIX 7: Train/test split is done BEFORE any numeric imputation.
    Medians are computed from X_train only and applied to X_test.
    """
    target_col = 'risk_target'
    drop_cols  = ['original_target', target_col]
    if not include_source:
        drop_cols.append('source')  # FIX 8: exclude source from features

    y_raw = master[target_col]
    X = master.drop(columns=drop_cols).copy()

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cat_cols:
        X.loc[:, c] = X[c].fillna('missing').astype(str)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)

    cat_indices = [X.columns.tolist().index(c) for c in cat_cols]

    # FIX 6: Target harmonization warning
    class_counts = dict(zip(le.classes_, np.bincount(y)))
    medium_count = class_counts.get('Medium', 0)
    medium_pct   = 100 * medium_count / max(1, len(y))
    print(f"\n{'='*60}")
    print("  UNIFIED MODEL — one CatBoost on the master feature table")
    print(f"  Rows: {len(X)}  |  Features: {len(X.columns)}")
    print(f"  Source included: {include_source}  (FIX 8: False = source excluded)")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Distribution: {class_counts}")
    if medium_pct < 5:
        print(f"\n  WARNING (Fix 6): Medium risk = {medium_count} rows ({medium_pct:.1f}%)")
        print("  Medium comes only from xAPI+UCI. OULAD contributes no Medium rows.")
        print("  Report Macro F1 and per-class PR-AUC — do NOT rely on accuracy alone.")
    print(f"{'='*60}")

    # FIX 7: Split FIRST — compute medians from X_train only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    # Fit medians on train, apply to both
    train_medians = X_train[num_cols].median()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train.loc[:, num_cols] = X_train[num_cols].fillna(train_medians)
    X_test.loc[:,  num_cols] = X_test[num_cols].fillna(train_medians)

    X_train_arr = X_train.values
    X_test_arr  = X_test.values
    y_train_arr = y_train.values
    y_test_arr  = y_test.values

    # 5-fold CV on training data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_acc, fold_f1, fold_ba = [], [], []
    for tr, va in cv.split(X_train_arr, y_train_arr):
        m = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3,
            random_seed=SEED, verbose=0,
            cat_features=cat_indices if cat_indices else None,
        )
        m.fit(X_train_arr[tr], y_train_arr[tr])
        p = m.predict(X_train_arr[va])
        fold_acc.append(accuracy_score(y_train_arr[va], p))
        fold_f1.append(f1_score(y_train_arr[va], p, average='macro', zero_division=0))
        fold_ba.append(balanced_accuracy_score(y_train_arr[va], p))

    print("\n  5-fold CV (training data only):")
    for name, vals in [('accuracy', fold_acc), ('f1_macro', fold_f1), ('balanced_accuracy', fold_ba)]:
        a = np.array(vals)
        print(f"    {name:<22}: {a.mean():.4f} +/- {a.std():.4f}")

    # Final model — trained on X_train only
    model = CatBoostClassifier(
        iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3,
        random_seed=SEED, verbose=0,
        cat_features=cat_indices if cat_indices else None,
    )
    model.fit(X_train_arr, y_train_arr)

    preds  = model.predict(X_test_arr)
    probas = model.predict_proba(X_test_arr)

    print("\n  Final test results:")
    print(f"    Accuracy        : {accuracy_score(y_test_arr, preds):.4f}")
    print(f"    Macro F1        : {f1_score(y_test_arr, preds, average='macro'):.4f}")
    print(f"    Balanced Acc    : {balanced_accuracy_score(y_test_arr, preds):.4f}")

    try:
        classes = np.unique(y_test_arr)
        y_bin = label_binarize(y_test_arr, classes=classes)
        if y_bin.shape[1] > 1:
            auc = roc_auc_score(y_bin, probas[:, :y_bin.shape[1]],
                                average='macro', multi_class='ovr')
            print(f"    ROC AUC (macro) : {auc:.4f}")
    except Exception:
        pass

    print(f"\n  Classification Report:")
    print("    " + classification_report(
        y_test_arr, preds, target_names=le.classes_).replace('\n', '\n    '))

    # Feature importance
    importances = model.get_feature_importance()
    feat_imp = sorted(zip(X_train.columns, importances), key=lambda x: -x[1])
    print("  Top 15 feature importances:")
    for feat, imp in feat_imp[:15]:
        print(f"    {feat:<42}: {imp:.4f}")

    # SHAP global (Fix 13 + Fix 15)
    if shap is not None:
        _shap_global(model, X_test, shap_sample, le)

    # Per-student intervention report (Fix 13 + Fix 15)
    _intervention_report(model, X_test, y_test, le, shap_sample, report_n)


# ---------------------------------------------------------------------------
# SHAP GLOBAL SUMMARY
# ---------------------------------------------------------------------------

def _shap_global(model, X_test: pd.DataFrame, sample: int, le=None):
    """
    FIX 13: Compute BOTH absolute importance (ranking) AND signed mean SHAP
    toward the High-risk class (direction — does feature push toward more risk?).

    FIX 15: Separate output into:
      - Top MODEL DRIVERS (any feature, ranked by |SHAP|)
      - Top ACTIONABLE RISK DRIVERS (only features student can influence,
        that have positive signed SHAP toward High risk)
    """
    X_s = X_test.sample(min(sample, len(X_test)), random_state=SEED)
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer(X_s)
        vals = shap_values.values  # (n_samples, n_features) or (n_samples, n_features, n_classes)

        # ── Absolute importance for ranking (Fix 13) ──────────────────────
        if vals.ndim == 3:
            abs_imp = np.abs(vals).mean(axis=(0, 2))
        else:
            abs_imp = np.abs(vals).mean(axis=0)

        # ── Signed mean SHAP toward High-risk class (Fix 13) ──────────────
        # Positive = pushes toward High risk; Negative = pushes toward Low risk
        if vals.ndim == 3 and le is not None:
            classes = list(le.classes_)
            high_idx = classes.index('High') if 'High' in classes else -1
            signed_mean = vals[:, :, high_idx].mean(axis=0)
        elif vals.ndim == 2:
            signed_mean = vals.mean(axis=0)
        else:
            signed_mean = np.zeros(len(abs_imp))

        features = list(X_s.columns)
        shap_df = sorted(zip(features, abs_imp, signed_mean), key=lambda x: -x[1])

        # ── Top MODEL DRIVERS (Fix 15: all features, by absolute importance) ─
        print("\n  Top 10 MODEL DRIVERS (|SHAP| — what the model relies on most):")
        for feat, abs_val, sign_val in shap_df[:10]:
            direction = 'risk ↑' if sign_val > 0 else 'risk ↓'
            print(f"    {feat:<42}: |SHAP|={abs_val:.4f}  direction={direction}")

        # ── Top ACTIONABLE RISK DRIVERS (Fix 15: student can act on these) ───
        # Only features that are actionable AND push toward higher risk
        actionable_risk = [(f, a, s) for f, a, s in shap_df
                           if f in ACTIONABLE_FEATURES and s > 0]
        if actionable_risk:
            print("\n  Top ACTIONABLE RISK DRIVERS (positive SHAP toward High — intervention targets):")
            for feat, abs_val, sign_val in actionable_risk[:5]:
                print(f"    {feat:<42}: |SHAP|={abs_val:.4f}  signed={sign_val:+.4f}")
        else:
            print("\n  No actionable features currently driving risk upward.")

    except Exception as e:
        print(f"  SHAP failed: {e}")


# ---------------------------------------------------------------------------
# PER-STUDENT INTERVENTION REPORT
# ---------------------------------------------------------------------------

# Recommendations driven by SHAP feature name + predicted risk level
_RECO = {
    'absence':          'Attendance is a key risk driver — attend all remaining sessions.',
    'engagement':       'Engagement is declining — log in daily and complete pending tasks.',
    'click':            'Low LMS activity — review course materials and attempt quizzes.',
    'consistency':      'Inconsistent study pattern — establish a regular weekly schedule.',
    'weekly_activity':  'Activity is dropping week-on-week — increase study hours now.',
    'score_improve':    'Assessment scores are not improving — seek tutor or faculty support.',
    'lifestyle_risk':   'Lifestyle risk factors detected — consider academic counselling.',
    'prior_failures':   'Prior failures on record — enroll in the academic support programme.',
    'study_time':       'Study time is low — target at least 10 hours per week.',
    'late_submission':  'Late submissions detected — submit all pending work on time.',
    'registration':     'Late registration detected — confirm all module enrolments.',
    'first_assessment': 'No early assessment submitted — complete pending assessments now.',
}


def _reco_for_feature(feat: str) -> str:
    for keyword, reco in _RECO.items():
        if keyword in feat.lower():
            return reco
    return f"Review your performance in: {feat}."


def _intervention_report(model, X_test: pd.DataFrame, y_test: pd.Series,
                          le: LabelEncoder, sample: int, n: int):
    """
    FIX 13: Use signed SHAP to determine risk direction.
    Only flag a feature as a risk driver if its signed SHAP pushes toward
    the predicted risk class (positive contribution).

    FIX 15: Separate model drivers (all top features) from actionable
    risk drivers (features student can change that actually increase risk).
    """
    if shap is None:
        print("\n  SHAP not available — skipping intervention report.")
        return

    X_s = X_test.sample(min(n, len(X_test)), random_state=SEED)
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer(X_s)
    except Exception as e:
        print(f"\n  Intervention report skipped: {e}")
        return

    preds  = model.predict(X_s)
    probas = model.predict_proba(X_s)
    vals   = shap_values.values   # (n, features) or (n, features, classes)
    classes = list(le.classes_)

    # Index of High-risk class for signed SHAP direction
    high_idx = classes.index('High') if 'High' in classes else len(classes) - 1

    print(f"\n{'='*60}")
    print("  STUDENT INTERVENTION REPORT")
    print(f"{'='*60}")

    for i, (row_idx, _) in enumerate(X_s.iterrows()):
        pred_enc   = int(np.squeeze(preds[i]))
        pred_label = le.inverse_transform([pred_enc])[0]
        confidence = float(probas[i].max())
        uncertain  = confidence < UNCERTAINTY_THRESHOLD

        # FIX 13: absolute |SHAP| for importance ranking
        if vals.ndim == 3:
            abs_sv   = np.abs(vals[i]).mean(axis=1)    # mean over classes
            # signed SHAP toward the High-risk class
            sign_sv  = vals[i][:, high_idx]
        else:
            abs_sv   = np.abs(vals[i])
            sign_sv  = vals[i]

        feat_abs  = list(zip(X_s.columns, abs_sv,  sign_sv))
        ranked_abs = sorted(feat_abs, key=lambda x: -x[1])

        # FIX 15: top MODEL DRIVERS (absolute, any feature)
        top_model_drivers = ranked_abs[:5]

        # FIX 15: top ACTIONABLE RISK DRIVERS
        # = actionable features where signed SHAP pushes toward High risk
        actionable_drivers = [
            (f, a, s) for f, a, s in ranked_abs
            if f in ACTIONABLE_FEATURES and s > 0
        ][:3]

        # Recommendations — only from actionable risk-increasing features
        recos = []
        seen  = set()
        for feat, _, _ in actionable_drivers:
            r = _reco_for_feature(feat)
            if r and r not in seen:
                recos.append(r)
                seen.add(r)

        # Fallback if nothing actionable is pushing risk up
        if not recos:
            recos.append(
                "Maintain regular engagement, monitor upcoming deadlines, "
                "and contact an academic advisor if difficulties continue.")

        source_val = X_s.iloc[i].get('source', 'not included') \
                     if 'source' in X_s.columns else '?'

        print(f"\n  Student #{i + 1}  [source: {source_val}]")
        print(f"    Predicted risk    : {pred_label}")
        print(f"    Confidence        : {confidence:.0%}"
              + ("  *** LOW — recommend advisor review ***" if uncertain else ""))

        # FIX 15: show model drivers with direction
        print("    Top model drivers (|SHAP| + risk direction):")
        for feat, abs_val, sign_val in top_model_drivers:
            direction = 'risk ↑' if sign_val > 0 else 'risk ↓'
            ftype = 'actionable' if feat in ACTIONABLE_FEATURES else (
                    'contextual' if feat in CONTEXTUAL_FEATURES else 'other')
            print(f"      - {feat:<36} |SHAP|={abs_val:.3f}  {direction}  [{ftype}]")

        # FIX 15: show only actionable risk drivers as intervention targets
        if actionable_drivers:
            print("    Actionable risk drivers (student can address these):")
            for feat, abs_val, sign_val in actionable_drivers:
                print(f"      - {feat:<36} signed={sign_val:+.3f}")

        print("    Recommendations (from actionable risk-increasing features):")
        for r in recos:
            print(f"      * {r}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Unified Student Success Platform')
    parser.add_argument('--mode', default='early-warning',
                        choices=['benchmark', 'early-warning'],
                        help='benchmark: include late features. early-warning: drop them.')
    parser.add_argument('--shap-sample',      type=int, default=200)
    parser.add_argument('--report-students',  type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).parent

    print("\n" + "="*60)
    print("  AI-POWERED STUDENT SUCCESS & EARLY INTERVENTION PLATFORM")
    print(f"  Mode : {args.mode.upper()}")
    print("="*60)
    print("""
  Data sources  -> unified feature store -> ONE CatBoost model
  -----------------------------------------------------------------
  OULAD   : LMS behaviour, assessment timing, demographics
  xAPI    : Classroom engagement, attendance, parent data
  UCI     : Background, study habits, lifestyle risk
  -----------------------------------------------------------------
  Shared schema : engagement, assessment, background,
                  demographics, AI-derived behavioral features
  Unified target: Low / Medium / High risk
""")

    print("  Building feature store...")
    master = build_feature_store(root, args.mode)

    train_unified(master, shap_sample=args.shap_sample, report_n=args.report_students)

    print(f"\n{'='*60}")
    print("  PLATFORM NOTE")
    print(f"{'='*60}")
    print("""
  All three datasets feed ONE model via a shared feature schema.
  In a real university system, each source (LMS, attendance
  portal, SIS) would write to the same feature store, and this
  single model would score every student nightly.

  To reach 90%+ accuracy: add attendance %, internal marks,
  library usage, and coding activity as additional sources.
  The feature store architecture above scales directly to that.
""")


if __name__ == '__main__':
    main()
