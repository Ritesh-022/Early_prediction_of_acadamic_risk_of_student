#!/usr/bin/env python3
"""
Branch Optimizer — Structured 6-Phase Research Plan
=====================================================
Implements the complete plan:

  Phase 1 : Lock baselines
  Phase 2 : Build & save branch datasets with branch-specific features
  Phase 3 : Optimize Model 2B (Pass vs Distinction)
             - Academic feature engineering
             - 6 imbalance experiments (2B-A through 2B-F)
             - Model comparison (XGB / LGB / CAT)
             - Soft-voting ensemble
             - External academic data (Option 4: domain-filtered augmentation)
  Phase 4 : Optimize Model 2A (Fail vs Withdrawn)
             - Withdrawal trajectory features
             - Model comparison
  Phase 5 : Final end-to-end hierarchical evaluation
             Best M1 + Best M2A + Best M2B → untouched test set
  Phase 6 : Compare all experiments in one table

Usage:
    python branch_optimizer.py                  # full run
    python branch_optimizer.py --skip-ext       # skip external data (Phase 3 step 3.3)
    python branch_optimizer.py --phase 3        # run only Phase 3
    python branch_optimizer.py --phase 4        # run only Phase 4
    python branch_optimizer.py --phase 5        # run final E2E eval
"""
from __future__ import annotations
import argparse, warnings, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── optional imports ──────────────────────────────────────────────────────────
try:
    from lightgbm  import LGBMClassifier;  _LGB = True
except ImportError:
    LGBMClassifier = None; _LGB = False
try:
    from xgboost   import XGBClassifier;   _XGB = True
except ImportError:
    XGBClassifier = None; _XGB = False
try:
    from catboost  import CatBoostClassifier; _CAT = True
except ImportError:
    CatBoostClassifier = None; _CAT = False
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine       import SMOTETomek
    _SMOTE = True
except ImportError:
    _SMOTE = False
try:
    import shap as _shap; _SHAP = True
except ImportError:
    _SHAP = False

from sklearn.model_selection   import (StratifiedKFold, cross_val_score,
                                        train_test_split)
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import (OneHotEncoder, LabelEncoder,
                                        StandardScaler)
from sklearn.ensemble          import (RandomForestClassifier,
                                        VotingClassifier)
from sklearn.linear_model      import LogisticRegression
from sklearn.utils             import resample
from sklearn.metrics           import (accuracy_score, f1_score,
                                        balanced_accuracy_score,
                                        cohen_kappa_score,
                                        classification_report,
                                        roc_auc_score, recall_score,
                                        precision_score)
from sklearn.preprocessing     import label_binarize

ROOT = Path(__file__).parent
SEED = 42

# ── column groups ──────────────────────────────────────────────────────────────
DROP_ALWAYS = {
    "final_result","id_student","code_module","code_presentation",
    "date_unregistration","date_unreg","date_unregistered","weighted_score",
    "first_ts","last_ts","active_weeks","clicks_per_active_week",
    "assessments_per_week","activity_count","days_active","avg_clicks_per_day",
    "week_click_sum_1_4","registration_delay_category",
    "id_assessment","id_site",
}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def build_pre(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(include=["object","category"]).columns.tolist()
    parts = []
    if num: parts.append(("n", SimpleImputer(strategy="median"), num))
    if cat: parts.append(("c", Pipeline([
        ("i", SimpleImputer(strategy="most_frequent")),
        ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat))
    return ColumnTransformer(parts, remainder="drop")


def make_lgb(cw=None, n=400):
    return LGBMClassifier(n_estimators=n, num_leaves=63, learning_rate=0.05,
                          feature_fraction=0.85, bagging_fraction=0.85,
                          bagging_freq=5, min_child_samples=15,
                          reg_alpha=0.1, reg_lambda=0.5,
                          class_weight=cw, random_state=SEED,
                          n_jobs=-1, verbosity=-1)

def make_xgb(n=400):
    return XGBClassifier(n_estimators=n, max_depth=6, learning_rate=0.05,
                         subsample=0.85, colsample_bytree=0.85,
                         use_label_encoder=False, eval_metric="logloss",
                         random_state=SEED, n_jobs=-1, verbosity=0,
                         tree_method="hist")

def make_cat(cw=None, n=400):
    kw = dict(iterations=n, depth=7, learning_rate=0.05,
              l2_leaf_reg=3, verbose=0, random_state=SEED)
    if cw == "balanced": kw["auto_class_weights"] = "Balanced"
    return CatBoostClassifier(**kw)

def best_clf(cw=None, n=400):
    if _LGB: return make_lgb(cw=cw, n=n)
    if _XGB: return make_xgb(n=n)
    if _CAT: return make_cat(cw=cw, n=n)
    return RandomForestClassifier(n_estimators=n, class_weight=cw,
                                   random_state=SEED, n_jobs=-1)


def fit_eval(X_tr, y_tr, X_te, y_te, clf, label: str,
             le: Optional[LabelEncoder] = None) -> Dict:
    """Fit clf on (X_tr,y_tr), eval on (X_te,y_te). Returns metrics dict."""
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    classes = le.classes_ if le else sorted(np.unique(y_te))

    r: Dict = {
        "label": label,
        "accuracy":    accuracy_score(y_te, preds),
        "f1_macro":    f1_score(y_te, preds, average="macro",    zero_division=0),
        "f1_weighted": f1_score(y_te, preds, average="weighted", zero_division=0),
        "balanced_acc":balanced_accuracy_score(y_te, preds),
        "kappa":       cohen_kappa_score(y_te, preds),
        "time_s":      round(time.time()-t0, 1),
    }
    # per-class recall
    for i, cls in enumerate(classes):
        mask = (y_te == i) if isinstance(preds[0], (int,np.integer)) else (y_te == cls)
        if mask.sum() > 0:
            r[f"recall_{cls}"] = recall_score(y_te, preds,
                                               labels=[i if le else cls],
                                               average="macro", zero_division=0)
    # ROC-AUC
    try:
        proba = clf.predict_proba(X_te)
        if len(classes) == 2:
            r["roc_auc"] = roc_auc_score(y_te, proba[:,1])
        else:
            yb = label_binarize(y_te, classes=range(len(classes)))
            r["roc_auc"] = roc_auc_score(yb, proba, average="macro",
                                          multi_class="ovr")
    except Exception:
        r["roc_auc"] = float("nan")
    return r


def cv_score(X, y, clf, folds=5) -> Tuple[float,float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    s  = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    return s.mean(), s.std()


def shap_top(clf, X_t: np.ndarray, feat_names: List[str], top=20) -> List[Tuple]:
    if not _SHAP: return []
    try:
        ex = _shap.TreeExplainer(clf)
        sv = ex.shap_values(X_t)
        if isinstance(sv, list):
            vals = np.mean([np.abs(v) for v in sv], axis=0).mean(axis=0)
        elif sv.ndim == 3:
            vals = np.abs(sv).mean(axis=(0,2))
        else:
            vals = np.abs(sv).mean(axis=0)
        n = min(len(feat_names), len(vals))
        order = np.argsort(vals[:n])[::-1][:top]
        return [(feat_names[i], float(vals[i])) for i in order]
    except Exception:
        return []


def print_row(r: Dict, minority_class: str = "") -> None:
    acc  = r.get("accuracy",0)
    f1m  = r.get("f1_macro",0)
    ba   = r.get("balanced_acc",0)
    kap  = r.get("kappa",0)
    roc  = r.get("roc_auc", float("nan"))
    rec  = r.get(f"recall_{minority_class}", float("nan"))
    flag = " ★" if acc >= 0.90 else (" ↑" if acc >= 0.87 else "")
    print(f"  {r['label']:<40} acc={acc:.4f}  f1={f1m:.4f}  "
          f"ba={ba:.4f}  rec_{minority_class}={rec:.4f}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Load V2 table & lock the split
# ─────────────────────────────────────────────────────────────────────────────

def phase1_load() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    p = ROOT / "oulad_ml_table_v2.csv"
    if not p.exists():
        raise FileNotFoundError("oulad_ml_table_v2.csv not found. Run oulad_pipeline_v2.py first.")
    df = pd.read_csv(p); df.columns = df.columns.str.strip()

    # fix silence_onset_week (was all-zeros due to bug — recompute from last_active_week)
    if "last_active_week" in df.columns:
        # went_silent = last_active_week < 30 (course typically 30-40 weeks)
        df["went_silent_flag"] = (df["last_active_week"] < 28).astype(int)
        df["early_dropout"]    = (df["last_active_week"] < 10).astype(int)
        df["mid_dropout"]      = ((df["last_active_week"] >= 10) & (df["last_active_week"] < 22)).astype(int)
        df["late_active"]      = (df["last_active_week"] >= 28).astype(int)

    y = df["final_result"]
    X = df.drop(columns=[c for c in DROP_ALWAYS | {"final_result"} if c in df.columns])
    X = X.dropna(axis=1, how="all")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"\n  Phase 1 complete.")
    print(f"  Train: {len(X_tr):,}  |  Test: {len(X_te):,}  |  Features: {X.shape[1]}")
    print(f"  Test class distribution: {y_te.value_counts().to_dict()}")
    return X_tr, X_te, y_tr, y_te


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Build branch datasets with branch-specific features
# ─────────────────────────────────────────────────────────────────────────────

# 2B: features that discriminate Pass vs Distinction (academic quality)
FEAT_2B = [
    "avg_score_v2","weighted_avg_score","max_score","min_score",
    "first_assessment_score","last_assessment_score",
    "score_trend","score_volatility","tma_count","cma_count",
    "first_tma_submitted","n_submissions","n_assessments_available",
    "submission_ratio","late_submission_count_v2","early_submission_count",
    "avg_submission_delay","max_submission_delay","assessment_submission_span",
    "studied_credits","num_of_prev_attempts",
    "gender","age_band","highest_education","imd_band","disability","region",
]

# 2A: features that discriminate Fail vs Withdrawn (withdrawal trajectory)
FEAT_2A = [
    "last_active_day","last_active_week","first_active_day","first_active_week",
    "active_days","active_weeks_v2","days_active_span","active_day_density",
    "went_silent_flag","early_dropout","mid_dropout","late_active",
    "consec_inactive_weeks","engagement_slope",
    "early_activity_ratio","mid_activity_ratio","late_activity_ratio",
    "peak_week","peak_week_clicks",
    "total_clicks_v2","precourse_clicks",
    "unique_resources_accessed","resource_diversity","unique_act_types",
    "forum_clicks","quiz_clicks","content_clicks","resource_clicks",
    "forum_ratio","quiz_ratio","content_ratio",
    "first_submission_day","last_submission_day",
    "n_submissions","submission_ratio",
    "avg_score_v2","score_trend","first_tma_submitted","tma_count",
    "late_submission_count_v2","avg_submission_delay",
    "studied_credits","num_of_prev_attempts",
    "gender","age_band","highest_education","imd_band","disability","region",
]

# Derived features to add during engineering
def engineer_2b(X: pd.DataFrame) -> pd.DataFrame:
    """Academic-quality features specifically for Pass vs Distinction."""
    X = X.copy()
    # score consistency
    if "score_volatility" in X.columns:
        X["score_consistency"] = 1 / (X["score_volatility"].replace(0, np.nan) + 1)
        X["score_consistency"] = X["score_consistency"].fillna(1)
    if "avg_score_v2" in X.columns:
        X["high_score_flag"]  = (X["avg_score_v2"] >= 70).astype(int)
        X["very_high_score"]  = (X["avg_score_v2"] >= 80).astype(int)
    if "max_score" in X.columns and "min_score" in X.columns:
        X["score_range"]      = X["max_score"] - X["min_score"]
    if "first_assessment_score" in X.columns and "last_assessment_score" in X.columns:
        X["score_improvement"]= X["last_assessment_score"] - X["first_assessment_score"]
        X["ended_strong"]     = (X["last_assessment_score"] >= 70).astype(int)
    if "submission_ratio" in X.columns:
        X["completed_all"]    = (X["submission_ratio"] >= 1.0).astype(int)
    if "late_submission_count_v2" in X.columns and "n_submissions" in X.columns:
        denom = X["n_submissions"].replace(0, np.nan)
        X["late_ratio"]       = (X["late_submission_count_v2"] / denom).fillna(0)
    if "avg_score_v2" in X.columns and "studied_credits" in X.columns:
        X["score_per_credit"] = (X["avg_score_v2"] /
                                  X["studied_credits"].replace(0, np.nan)).fillna(0)
    if "tma_count" in X.columns and "avg_score_v2" in X.columns:
        X["score_x_tmas"] = X["avg_score_v2"] * X["tma_count"]
    if "weighted_avg_score" in X.columns and "submission_ratio" in X.columns:
        X["weighted_x_completion"] = X["weighted_avg_score"] * X["submission_ratio"]
    return X


def engineer_2a(X: pd.DataFrame) -> pd.DataFrame:
    """Withdrawal-trajectory features specifically for Fail vs Withdrawn."""
    X = X.copy()
    if "last_active_week" in X.columns and "first_active_week" in X.columns:
        X["weeks_enrolled"] = X["last_active_week"] - X["first_active_week"]
        X["weeks_enrolled"] = X["weeks_enrolled"].clip(lower=0)
    if "last_active_day" in X.columns:
        X["days_since_last"] = 250 - X["last_active_day"].clip(upper=250)
    if "active_weeks_v2" in X.columns and "last_active_week" in X.columns:
        denom = (X["last_active_week"] - X["first_active_week"].fillna(0)).replace(0, np.nan)
        X["active_week_density"] = (X["active_weeks_v2"] / denom).fillna(0).clip(0, 1)
    if "total_clicks_v2" in X.columns and "last_active_week" in X.columns:
        denom = X["last_active_week"].replace(0, np.nan)
        X["clicks_per_active_week_v2"] = (X["total_clicks_v2"] / denom).fillna(0)
    if "n_submissions" in X.columns and "last_active_week" in X.columns:
        denom = X["last_active_week"].replace(0, np.nan)
        X["submissions_per_week"] = (X["n_submissions"] / denom).fillna(0)
    if "score_trend" in X.columns:
        X["score_declining"] = (X["score_trend"] < 0).astype(int)
    if "early_activity_ratio" in X.columns and "late_activity_ratio" in X.columns:
        X["activity_shift"] = X["late_activity_ratio"] - X["early_activity_ratio"]
    return X


def phase2_build_branches(X_tr, X_te, y_tr, y_te, out_dir: Path):
    """Create branch datasets and save them."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Branch 1: AtRisk vs Success (Model 1 — already good, just save)
    for split, X, y in [("train", X_tr, y_tr), ("test", X_te, y_te)]:
        df = X.copy(); df["final_result"] = y.values
        df["branch_target"] = y.map({"Pass":"Success","Distinction":"Success",
                                      "Fail":"AtRisk","Withdrawn":"AtRisk"}).values
        df.to_csv(out_dir / f"branch_1_atrisk_success_{split}.csv", index=False)

    # Branch 2A: Fail vs Withdrawn
    mask_tr_2a = y_tr.isin(["Fail","Withdrawn"])
    mask_te_2a = y_te.isin(["Fail","Withdrawn"])
    cols_2a = [c for c in FEAT_2A if c in X_tr.columns]
    X_tr_2a = engineer_2a(X_tr.loc[mask_tr_2a, cols_2a])
    X_te_2a = engineer_2a(X_te.loc[mask_te_2a, cols_2a])
    y_tr_2a = y_tr.loc[mask_tr_2a]; y_te_2a = y_te.loc[mask_te_2a]
    for split, X, y in [("train", X_tr_2a, y_tr_2a), ("test", X_te_2a, y_te_2a)]:
        df = X.copy(); df["final_result"] = y.values
        df.to_csv(out_dir / f"branch_2a_fail_withdrawn_{split}.csv", index=False)

    # Branch 2B: Pass vs Distinction
    mask_tr_2b = y_tr.isin(["Pass","Distinction"])
    mask_te_2b = y_te.isin(["Pass","Distinction"])
    cols_2b = [c for c in FEAT_2B if c in X_tr.columns]
    X_tr_2b = engineer_2b(X_tr.loc[mask_tr_2b, cols_2b])
    X_te_2b = engineer_2b(X_te.loc[mask_te_2b, cols_2b])
    y_tr_2b = y_tr.loc[mask_tr_2b]; y_te_2b = y_te.loc[mask_te_2b]
    for split, X, y in [("train", X_tr_2b, y_tr_2b), ("test", X_te_2b, y_te_2b)]:
        df = X.copy(); df["final_result"] = y.values
        df.to_csv(out_dir / f"branch_2b_pass_distinction_{split}.csv", index=False)

    print(f"\n  Branch datasets saved to {out_dir}")
    print(f"  2A train: {len(X_tr_2a):,} rows  {X_tr_2a.shape[1]} features")
    print(f"  2A class dist: {y_tr_2a.value_counts().to_dict()}")
    print(f"  2B train: {len(X_tr_2b):,} rows  {X_tr_2b.shape[1]} features")
    print(f"  2B class dist: {y_tr_2b.value_counts().to_dict()}")

    return (X_tr_2a, X_te_2a, y_tr_2a, y_te_2a,
            X_tr_2b, X_te_2b, y_tr_2b, y_te_2b)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Optimize Model 2B (Pass vs Distinction)
# ─────────────────────────────────────────────────────────────────────────────

def undersample_majority(X, y, ratio: float, seed=SEED):
    """Undersample majority class to given Pass:Distinction ratio."""
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    majority_cls = np.argmax(np.bincount(y_enc))
    minority_cls = 1 - majority_cls
    n_min  = (y_enc == minority_cls).sum()
    n_maj  = int(n_min * ratio)
    maj_idx = np.where(y_enc == majority_cls)[0]
    rng = np.random.RandomState(seed)
    keep = rng.choice(maj_idx, size=min(n_maj, len(maj_idx)), replace=False)
    min_idx = np.where(y_enc == minority_cls)[0]
    idx = np.concatenate([keep, min_idx])
    return X[idx], y_enc[idx], le


def phase3_optimize_2b(X_tr_raw: pd.DataFrame, X_te_raw: pd.DataFrame,
                        y_tr: pd.Series, y_te: pd.Series,
                        skip_ext: bool = False) -> Tuple[Dict, object]:
    print(f"\n{'='*65}")
    print("  PHASE 3 — Model 2B: Pass vs Distinction")
    print(f"{'='*65}")
    print(f"  Train: {len(X_tr_raw):,}  Test: {len(X_te_raw):,}")
    print(f"  Train dist: {y_tr.value_counts().to_dict()}")

    le = LabelEncoder(); le.fit(["Distinction","Pass"])
    y_tr_enc = le.transform(y_tr); y_te_enc = le.transform(y_te)

    # Preprocess once
    pre = build_pre(X_tr_raw)
    X_tr_t = pre.fit_transform(X_tr_raw).astype(np.float32)
    X_te_t = pre.transform(X_te_raw).astype(np.float32)

    # Feature names for SHAP
    try:
        feat_names = [n.replace("n__","").replace("c__","")
                      for n in pre.get_feature_names_out()]
    except Exception:
        feat_names = [f"f{i}" for i in range(X_tr_t.shape[1])]

    results_2b: List[Dict] = []

    def run(label, X_tr, y_tr, clf_fn, note=""):
        clf = clf_fn()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te_t)
        acc = accuracy_score(y_te_enc, preds)
        f1m = f1_score(y_te_enc, preds, average="macro", zero_division=0)
        ba  = balanced_accuracy_score(y_te_enc, preds)
        rec_dist = recall_score(y_te_enc, preds, pos_label=le.transform(["Distinction"])[0],
                                average="binary", zero_division=0)
        rec_pass = recall_score(y_te_enc, preds, pos_label=le.transform(["Pass"])[0],
                                average="binary", zero_division=0)
        try:
            proba = clf.predict_proba(X_te_t)
            roc = roc_auc_score(y_te_enc, proba[:,1])
        except Exception:
            roc = float("nan")
        r = dict(label=label, accuracy=acc, f1_macro=f1m, balanced_acc=ba,
                 recall_Distinction=rec_dist, recall_Pass=rec_pass, roc_auc=roc)
        flag = " ★" if acc >= 0.90 else (" ↑" if f1m >= 0.87 else "")
        print(f"  {label:<42} acc={acc:.4f}  f1={f1m:.4f}  "
              f"ba={ba:.4f}  rec_Dist={rec_dist:.4f}{flag}")
        return r, clf

    # ── 2B-A: Baseline (no balancing) ────────────────────────────────────────
    print("\n  --- Imbalance experiments ---")
    r, best_clf_2b = run("2B-A Baseline (no balance)",
                         X_tr_t, y_tr_enc, lambda: best_clf())
    results_2b.append(r); best_score_2b = r["f1_macro"]

    # ── 2B-B: Class weights ───────────────────────────────────────────────────
    r, clf_tmp = run("2B-B Class weights",
                     X_tr_t, y_tr_enc, lambda: best_clf(cw="balanced"))
    results_2b.append(r)
    if r["f1_macro"] > best_score_2b:
        best_score_2b = r["f1_macro"]; best_clf_2b = clf_tmp

    # ── 2B-C..F: Undersampling at different ratios ────────────────────────────
    for ratio, name in [(3.0,"3:1"), (2.0,"2:1"), (1.5,"1.5:1"), (1.0,"1:1")]:
        X_us, y_us, _ = undersample_majority(X_tr_t, y_tr_enc, ratio)
        r, clf_tmp = run(f"2B-{chr(67+results_2b.__len__()-1)} Undersample {name}",
                         X_us, y_us, lambda: best_clf())
        results_2b.append(r)
        if r["f1_macro"] > best_score_2b:
            best_score_2b = r["f1_macro"]; best_clf_2b = clf_tmp

    # ── 2B-SMOTE: SMOTE (6k Distinction) ─────────────────────────────────────
    if _SMOTE:
        n_pass = (y_tr_enc == le.transform(["Pass"])[0]).sum()
        n_tgt  = min(8000, n_pass)
        sm = SMOTE(sampling_strategy={le.transform(["Distinction"])[0]: n_tgt},
                   random_state=SEED, k_neighbors=5)
        X_sm, y_sm = sm.fit_resample(X_tr_t, y_tr_enc)
        r, clf_tmp = run("2B-SMOTE (moderate 4k→8k Distinction)",
                         X_sm, y_sm, lambda: best_clf())
        results_2b.append(r)
        if r["f1_macro"] > best_score_2b:
            best_score_2b = r["f1_macro"]; best_clf_2b = clf_tmp

        # SMOTE-Tomek
        smt = SMOTETomek(random_state=SEED)
        X_smt, y_smt = smt.fit_resample(X_tr_t, y_tr_enc)
        r, clf_tmp = run("2B-SMOTE-Tomek",
                         X_smt, y_smt, lambda: best_clf())
        results_2b.append(r)
        if r["f1_macro"] > best_score_2b:
            best_score_2b = r["f1_macro"]; best_clf_2b = clf_tmp

    # ── Model comparison (all three boosters with best imbalance strategy) ────
    print("\n  --- Model comparison (class weights) ---")
    best_ratio_X, best_ratio_y = X_tr_t, y_tr_enc
    for mname, clf_fn in [
        ("XGBoost+cw",  lambda: make_xgb() if _XGB else best_clf(cw="balanced")),
        ("LightGBM+cw", lambda: make_lgb(cw="balanced") if _LGB else best_clf(cw="balanced")),
        ("CatBoost+cw", lambda: make_cat(cw="balanced") if _CAT else best_clf(cw="balanced")),
    ]:
        r, clf_tmp = run(f"2B-{mname}", best_ratio_X, best_ratio_y, clf_fn)
        results_2b.append(r)
        if r["f1_macro"] > best_score_2b:
            best_score_2b = r["f1_macro"]; best_clf_2b = clf_tmp

    # ── Soft-voting ensemble ──────────────────────────────────────────────────
    print("\n  --- Soft-voting ensemble ---")
    estimators = []
    if _LGB: estimators.append(("lgb", make_lgb(cw="balanced")))
    if _XGB: estimators.append(("xgb", make_xgb()))
    if _CAT: estimators.append(("cat", make_cat(cw="balanced")))
    if len(estimators) >= 2:
        vote = VotingClassifier(estimators, voting="soft")
        r, clf_tmp = run("2B-SoftVote (LGB+XGB+CAT)",
                         X_tr_t, y_tr_enc, lambda: vote)
        results_2b.append(r)
        if r["f1_macro"] > best_score_2b:
            best_score_2b = r["f1_macro"]; best_clf_2b = clf_tmp

    # ── External academic data (Option 4) ─────────────────────────────────────
    if not skip_ext:
        ext = _load_external_2b(pre, le, X_tr_t, y_tr_enc, X_te_raw, y_te_enc)
        if ext:
            r, clf_tmp = ext
            results_2b.append(r)
            if r["f1_macro"] > best_score_2b:
                best_score_2b = r["f1_macro"]; best_clf_2b = clf_tmp

    # ── SHAP on best model ────────────────────────────────────────────────────
    print("\n  --- SHAP (best 2B model) ---")
    X_shap = X_te_t[:min(500, len(X_te_t))]
    shap_imp = shap_top(best_clf_2b, X_shap, feat_names)
    if shap_imp:
        print("  Top SHAP features (2B):")
        for feat, val in shap_imp[:12]:
            print(f"    {feat:<40} {val:.5f}")

    # best result
    best_r = max(results_2b, key=lambda x: x["f1_macro"])
    print(f"\n  ✓ Best 2B: {best_r['label']}")
    print(f"    acc={best_r['accuracy']:.4f}  f1_macro={best_r['f1_macro']:.4f}  "
          f"rec_Distinction={best_r['recall_Distinction']:.4f}")

    return best_r, best_clf_2b, pre, le, results_2b


def _load_external_2b(pre, le, X_tr_t, y_tr_enc, X_te_raw, y_te_enc):
    """Option 4: external academic data augmentation for Distinction class."""
    print("\n  --- External academic data (Option 4) ---")
    ext_rows = []

    # UCI Student Performance (mat + por)
    for fname in ["student+performance/student/student-mat.csv",
                  "student+performance/student/student-por.csv",
                  "UI_student+performance/student/student-mat.csv",
                  "UI_student+performance/student/student-por.csv"]:
        p = ROOT / fname
        if p.exists():
            try:
                raw = pd.read_csv(p, sep=";")
                raw.columns = [c.strip() for c in raw.columns]
                if "G3" not in raw.columns: continue
                raw["G3"] = pd.to_numeric(raw["G3"], errors="coerce").fillna(0)
                # Only high performers (G3 >= 15) → Distinction proxy
                high = raw[raw["G3"] >= 15].copy()
                if len(high) == 0: continue
                # Build common academic schema
                row = pd.DataFrame({
                    "avg_score_v2":          high["G3"] * 5,     # scale to ~0-100
                    "weighted_avg_score":     high["G3"] * 5,
                    "max_score":             high["G3"] * 5,
                    "min_score":             high[["G1","G2","G3"]].min(axis=1)*5 if "G1" in high.columns else high["G3"]*4,
                    "first_assessment_score": high["G1"]*5 if "G1" in high.columns else high["G3"]*5,
                    "last_assessment_score":  high["G3"]*5,
                    "score_trend":           (high["G3"]-high["G1"])*5 if "G1" in high.columns else 0,
                    "score_volatility":      3.0,
                    "submission_ratio":      1.0,
                    "n_submissions":         3.0,
                    "n_assessments_available":3.0,
                    "late_submission_count_v2":0.0,
                    "studied_credits":        60.0,
                    "num_of_prev_attempts":  high["failures"] if "failures" in high.columns else 0,
                })
                row = engineer_2b(row)
                ext_rows.append(row)
                print(f"  UCI ({fname.split('/')[-1]}): {len(high)} high-performers as Distinction proxy")
            except Exception as e:
                print(f"  Warning: {e}")

    if not ext_rows:
        print("  No compatible external data found.")
        return None

    ext_df = pd.concat(ext_rows, ignore_index=True)
    ext_y  = np.full(len(ext_df), le.transform(["Distinction"])[0])

    # transform using existing preprocessor
    try:
        ext_t = pre.transform(ext_df).astype(np.float32)
    except Exception:
        # column mismatch — align
        missing = set(pre.feature_names_in_) - set(ext_df.columns) \
                  if hasattr(pre, "feature_names_in_") else set()
        for c in missing:
            ext_df[c] = 0
        try:
            ext_t = pre.transform(ext_df).astype(np.float32)
        except Exception as e:
            print(f"  External transform failed: {e}")
            return None

    # domain similarity filter: cosine similarity to OULAD Distinction centroid
    dist_idx = np.where(y_tr_enc == le.transform(["Distinction"])[0])[0]
    centroid  = X_tr_t[dist_idx].mean(axis=0)
    norms_ext  = np.linalg.norm(ext_t, axis=1, keepdims=True) + 1e-9
    norms_cent = np.linalg.norm(centroid) + 1e-9
    sims = (ext_t @ centroid) / (norms_ext.squeeze() * norms_cent)
    keep = sims >= np.percentile(sims, 30)   # keep top-70% similar
    ext_t_filtered = ext_t[keep]
    ext_y_filtered = ext_y[keep]
    print(f"  After domain filter: {keep.sum()} / {len(ext_t)} external Distinction rows kept")

    X_aug = np.vstack([X_tr_t, ext_t_filtered])
    y_aug = np.concatenate([y_tr_enc, ext_y_filtered])
    print(f"  Augmented training set: {len(X_aug):,} rows  "
          f"Distinction={( y_aug==le.transform(['Distinction'])[0]).sum():,}")

    clf = best_clf(cw="balanced")
    clf.fit(X_aug, y_aug)
    preds = clf.predict(pre.transform(X_te_raw).astype(np.float32))
    acc = accuracy_score(y_te_enc, preds)
    f1m = f1_score(y_te_enc, preds, average="macro", zero_division=0)
    ba  = balanced_accuracy_score(y_te_enc, preds)
    rec_dist = recall_score(y_te_enc, preds,
                            pos_label=le.transform(["Distinction"])[0],
                            average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y_te_enc, clf.predict_proba(
            pre.transform(X_te_raw).astype(np.float32))[:,1])
    except Exception:
        roc = float("nan")
    r = dict(label="2B-G External academic aug", accuracy=acc,
             f1_macro=f1m, balanced_acc=ba,
             recall_Distinction=rec_dist, recall_Pass=0, roc_auc=roc)
    flag = " ★" if acc >= 0.90 else ""
    print(f"  2B-G External academic aug              acc={acc:.4f}  f1={f1m:.4f}  "
          f"ba={ba:.4f}  rec_Dist={rec_dist:.4f}{flag}")
    return r, clf

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Optimize Model 2A (Fail vs Withdrawn)
# ─────────────────────────────────────────────────────────────────────────────

def phase4_optimize_2a(X_tr_raw: pd.DataFrame, X_te_raw: pd.DataFrame,
                        y_tr: pd.Series, y_te: pd.Series) -> Tuple[Dict, object]:
    print(f"\n{'='*65}")
    print("  PHASE 4 — Model 2A: Fail vs Withdrawn")
    print(f"{'='*65}")
    print(f"  Train: {len(X_tr_raw):,}  Test: {len(X_te_raw):,}")
    print(f"  Train dist: {y_tr.value_counts().to_dict()}")

    le = LabelEncoder(); le.fit(["Fail","Withdrawn"])
    y_tr_enc = le.transform(y_tr); y_te_enc = le.transform(y_te)

    pre = build_pre(X_tr_raw)
    X_tr_t = pre.fit_transform(X_tr_raw).astype(np.float32)
    X_te_t = pre.transform(X_te_raw).astype(np.float32)

    try:
        feat_names = [n.replace("n__","").replace("c__","")
                      for n in pre.get_feature_names_out()]
    except Exception:
        feat_names = [f"f{i}" for i in range(X_tr_t.shape[1])]

    results_2a: List[Dict] = []

    def run(label, X_tr, y_tr_l, clf_fn):
        clf = clf_fn(); clf.fit(X_tr, y_tr_l)
        preds = clf.predict(X_te_t)
        acc = accuracy_score(y_te_enc, preds)
        f1m = f1_score(y_te_enc, preds, average="macro", zero_division=0)
        ba  = balanced_accuracy_score(y_te_enc, preds)
        wd_cls  = le.transform(["Withdrawn"])[0]
        fail_cls= le.transform(["Fail"])[0]
        rec_w = recall_score(y_te_enc, preds, pos_label=wd_cls,   average="binary", zero_division=0)
        rec_f = recall_score(y_te_enc, preds, pos_label=fail_cls, average="binary", zero_division=0)
        try:
            roc = roc_auc_score(y_te_enc, clf.predict_proba(X_te_t)[:,1])
        except Exception:
            roc = float("nan")
        r = dict(label=label, accuracy=acc, f1_macro=f1m, balanced_acc=ba,
                 recall_Withdrawn=rec_w, recall_Fail=rec_f, roc_auc=roc)
        flag = " ★" if f1m >= 0.82 else (" ↑" if f1m >= 0.79 else "")
        print(f"  {label:<42} acc={acc:.4f}  f1={f1m:.4f}  "
              f"ba={ba:.4f}  rec_W={rec_w:.4f}{flag}")
        return r, clf

    # Baseline
    print("\n  --- 2A baseline & model comparison ---")
    r, best_clf_2a = run("2A-1 Baseline LightGBM", X_tr_t, y_tr_enc,
                          lambda: best_clf())
    results_2a.append(r); best_score_2a = r["f1_macro"]

    # Class weights
    r, clf_tmp = run("2A-2 Class weights", X_tr_t, y_tr_enc,
                      lambda: best_clf(cw="balanced"))
    results_2a.append(r)
    if r["f1_macro"] > best_score_2a: best_score_2a = r["f1_macro"]; best_clf_2a = clf_tmp

    # Model comparison
    for mname, clf_fn in [
        ("XGBoost+cw",  lambda: make_xgb() if _XGB else best_clf(cw="balanced")),
        ("CatBoost+cw", lambda: make_cat(cw="balanced") if _CAT else best_clf(cw="balanced")),
    ]:
        r, clf_tmp = run(f"2A-{mname}", X_tr_t, y_tr_enc, clf_fn)
        results_2a.append(r)
        if r["f1_macro"] > best_score_2a: best_score_2a = r["f1_macro"]; best_clf_2a = clf_tmp

    # SMOTE (mild — 1:1.2 ratio)
    if _SMOTE:
        fail_cls  = le.transform(["Fail"])[0]
        n_withdrawn = (y_tr_enc != fail_cls).sum()
        n_target    = int(n_withdrawn * 1.2)
        sm = SMOTE(sampling_strategy={fail_cls: n_target},
                   random_state=SEED, k_neighbors=5)
        X_sm, y_sm = sm.fit_resample(X_tr_t, y_tr_enc)
        r, clf_tmp = run("2A-SMOTE (mild 1:1.2)", X_sm, y_sm,
                          lambda: best_clf())
        results_2a.append(r)
        if r["f1_macro"] > best_score_2a: best_score_2a = r["f1_macro"]; best_clf_2a = clf_tmp

    # Soft voting
    estimators = []
    if _LGB: estimators.append(("lgb", make_lgb(cw="balanced")))
    if _XGB: estimators.append(("xgb", make_xgb()))
    if _CAT: estimators.append(("cat", make_cat(cw="balanced")))
    if len(estimators) >= 2:
        vote = VotingClassifier(estimators, voting="soft")
        r, clf_tmp = run("2A-SoftVote", X_tr_t, y_tr_enc, lambda: vote)
        results_2a.append(r)
        if r["f1_macro"] > best_score_2a: best_score_2a = r["f1_macro"]; best_clf_2a = clf_tmp

    # SHAP
    print("\n  --- SHAP (best 2A model) ---")
    X_shap = X_te_t[:min(500, len(X_te_t))]
    shap_imp = shap_top(best_clf_2a, X_shap, feat_names)
    if shap_imp:
        print("  Top SHAP features (2A):")
        for feat, val in shap_imp[:12]:
            print(f"    {feat:<40} {val:.5f}")

    best_r = max(results_2a, key=lambda x: x["f1_macro"])
    print(f"\n  ✓ Best 2A: {best_r['label']}")
    print(f"    acc={best_r['accuracy']:.4f}  f1_macro={best_r['f1_macro']:.4f}  "
          f"rec_Withdrawn={best_r['recall_Withdrawn']:.4f}")

    return best_r, best_clf_2a, pre, le, results_2a

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Final end-to-end hierarchical evaluation
# ─────────────────────────────────────────────────────────────────────────────

FOUR_ORDER = ["Distinction","Fail","Pass","Withdrawn"]

def phase5_end_to_end(X_tr: pd.DataFrame, X_te: pd.DataFrame,
                       y_tr: pd.Series,  y_te: pd.Series,
                       best_2a_clf, pre_2a, le_2a,
                       best_2b_clf, pre_2b, le_2b,
                       cols_2a: List[str], cols_2b: List[str]) -> Dict:
    print(f"\n{'='*65}")
    print("  PHASE 5 — End-to-End Hierarchical Evaluation")
    print(f"{'='*65}")
    print("  Using UNTOUCHED original test set (all 4 classes)")
    print(f"  Test distribution: {y_te.value_counts().to_dict()}")

    # ── Model 1: train on full training set ───────────────────────────────────
    y_bin_tr = y_tr.map({"Pass":"Success","Distinction":"Success",
                          "Fail":"AtRisk","Withdrawn":"AtRisk"})
    le1 = LabelEncoder(); le1.fit(["AtRisk","Success"])
    y1_tr = le1.transform(y_bin_tr)

    pre1 = build_pre(X_tr)
    X1_tr = pre1.fit_transform(X_tr).astype(np.float32)
    X1_te = pre1.transform(X_te).astype(np.float32)

    m1 = best_clf(); m1.fit(X1_tr, y1_tr)
    p1 = m1.predict_proba(X1_te)
    p_atrisk  = p1[:, list(le1.classes_).index("AtRisk")]
    p_success = p1[:, list(le1.classes_).index("Success")]
    m1_acc = accuracy_score(y1_tr, m1.predict(X1_tr))

    # ── Model 2A: Fail vs Withdrawn probabilities on FULL test set ────────────
    avail_2a = [c for c in cols_2a if c in X_te.columns]
    X_te_2a  = engineer_2a(X_te[avail_2a])
    X2a_te   = pre_2a.transform(X_te_2a).astype(np.float32)
    p2a = best_2a_clf.predict_proba(X2a_te)
    idx_fail = list(le_2a.classes_).index("Fail")
    idx_with = list(le_2a.classes_).index("Withdrawn")
    p_fail_given_ar     = p2a[:, idx_fail]
    p_withdrawn_given_ar= p2a[:, idx_with]

    # ── Model 2B: Pass vs Distinction probabilities on FULL test set ──────────
    avail_2b = [c for c in cols_2b if c in X_te.columns]
    X_te_2b  = engineer_2b(X_te[avail_2b])
    X2b_te   = pre_2b.transform(X_te_2b).astype(np.float32)
    p2b = best_2b_clf.predict_proba(X2b_te)
    idx_pass = list(le_2b.classes_).index("Pass")
    idx_dist = list(le_2b.classes_).index("Distinction")
    p_pass_given_s  = p2b[:, idx_pass]
    p_dist_given_s  = p2b[:, idx_dist]

    # ── Probabilistic fusion ──────────────────────────────────────────────────
    # P(Fail)        = P(AtRisk) * P(Fail | AtRisk)
    # P(Withdrawn)   = P(AtRisk) * P(Withdrawn | AtRisk)
    # P(Pass)        = P(Success) * P(Pass | Success)
    # P(Distinction) = P(Success) * P(Distinction | Success)
    p_final = np.column_stack([
        p_success * p_dist_given_s,   # Distinction
        p_atrisk  * p_fail_given_ar,  # Fail
        p_success * p_pass_given_s,   # Pass
        p_atrisk  * p_withdrawn_given_ar  # Withdrawn
    ])
    # renormalise
    row_sums = p_final.sum(axis=1, keepdims=True)
    p_final  = p_final / np.where(row_sums > 0, row_sums, 1)

    pred_idx = np.argmax(p_final, axis=1)
    y_pred   = np.array(FOUR_ORDER)[pred_idx]
    y_true   = y_te.values

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro",    zero_division=0, labels=FOUR_ORDER)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=FOUR_ORDER)
    ba  = balanced_accuracy_score(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)
    try:
        y_bin  = label_binarize(y_true, classes=FOUR_ORDER)
        roc_   = roc_auc_score(y_bin, p_final, average="macro", multi_class="ovr")
    except Exception:
        roc_ = float("nan")

    per_class = {}
    for cls in FOUR_ORDER:
        mask = (y_true == cls)
        if mask.sum() > 0:
            per_class[f"recall_{cls}"]    = recall_score(y_true, y_pred,
                                                          labels=[cls], average="macro", zero_division=0)
            per_class[f"precision_{cls}"] = precision_score(y_true, y_pred,
                                                             labels=[cls], average="macro", zero_division=0)

    print(f"\n  M1 train-acc (AtRisk/Success) : {m1_acc:.4f}")
    print(f"\n  ─── FINAL 4-CLASS HIERARCHICAL RESULT ───")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Macro F1        : {f1m:.4f}")
    print(f"  Weighted F1     : {f1w:.4f}")
    print(f"  Balanced Acc    : {ba:.4f}")
    print(f"  Cohen Kappa     : {kap:.4f}")
    print(f"  ROC-AUC (macro) : {roc_:.4f}")
    print(f"\n  Per-class recall:")
    for cls in FOUR_ORDER:
        print(f"    {cls:<12} : {per_class.get(f'recall_{cls}', 0):.4f}")
    print(f"\n  Classification report:")
    print("  " + classification_report(y_true, y_pred,
                                        labels=FOUR_ORDER, zero_division=0
                                        ).replace("\n","\n  "))

    r = dict(label="Hierarchical V2 (best M1+2A+2B)", accuracy=acc,
             f1_macro=f1m, balanced_acc=ba, kappa=kap, roc_auc=roc_,
             **per_class)
    return r

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — Final comparison table
# ─────────────────────────────────────────────────────────────────────────────

def phase6_table(baseline_direct: Dict, best_2b_r: Dict,
                 best_2a_r: Dict, final_r: Dict,
                 all_2b: List[Dict], all_2a: List[Dict],
                 out_dir: Path):
    print(f"\n{'='*80}")
    print("  PHASE 6 — COMPLETE RESULTS TABLE")
    print(f"{'='*80}")

    # ── Direct 4-class baseline ───────────────────────────────────────────────
    print("\n  BASELINE (direct 4-class LightGBM on V2):")
    print(f"  acc={baseline_direct['accuracy']:.4f}  f1={baseline_direct['f1_macro']:.4f}  "
          f"ba={baseline_direct['balanced_acc']:.4f}  kappa={baseline_direct['kappa']:.4f}")

    # ── Branch 2B summary ─────────────────────────────────────────────────────
    print(f"\n  MODEL 2B (Pass vs Distinction) — all experiments:")
    print(f"  {'Experiment':<42} {'Acc':>6} {'F1-Mac':>7} {'BA':>6} {'Rec-Dist':>9}")
    print(f"  {'─'*70}")
    for r in all_2b:
        print(f"  {r['label']:<42} {r['accuracy']:>6.4f} {r['f1_macro']:>7.4f} "
              f"{r['balanced_acc']:>6.4f} {r.get('recall_Distinction',0):>9.4f}")

    # ── Branch 2A summary ─────────────────────────────────────────────────────
    print(f"\n  MODEL 2A (Fail vs Withdrawn) — all experiments:")
    print(f"  {'Experiment':<42} {'Acc':>6} {'F1-Mac':>7} {'BA':>6} {'Rec-With':>9}")
    print(f"  {'─'*70}")
    for r in all_2a:
        print(f"  {r['label']:<42} {r['accuracy']:>6.4f} {r['f1_macro']:>7.4f} "
              f"{r['balanced_acc']:>6.4f} {r.get('recall_Withdrawn',0):>9.4f}")

    # ── Final 4-class comparison ──────────────────────────────────────────────
    base_acc = baseline_direct["accuracy"]
    hier_acc = final_r["accuracy"]
    gain     = hier_acc - base_acc

    print(f"\n{'='*80}")
    print(f"  FINAL 4-CLASS COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Experiment':<45} {'Acc':>6} {'F1-Mac':>7} {'BA':>6} {'Kappa':>6}")
    print(f"  {'─'*70}")
    print(f"  {'Direct 4-class V2 (baseline)':<45} "
          f"{base_acc:>6.4f} {baseline_direct['f1_macro']:>7.4f} "
          f"{baseline_direct['balanced_acc']:>6.4f} {baseline_direct['kappa']:>6.4f}")
    print(f"  {'Hierarchical V2 (best M1+2A+2B)':<45} "
          f"{hier_acc:>6.4f} {final_r['f1_macro']:>7.4f} "
          f"{final_r['balanced_acc']:>6.4f} {final_r['kappa']:>6.4f}")
    sign = f"+{gain:.4f}" if gain >= 0 else f"{gain:.4f}"
    print(f"\n  Accuracy gain : {sign}  ({gain*100:.2f} percentage points)")
    print(f"  Model 2B best : {best_2b_r['label']:<35} f1={best_2b_r['f1_macro']:.4f}  "
          f"rec_Distinction={best_2b_r.get('recall_Distinction',0):.4f}")
    print(f"  Model 2A best : {best_2a_r['label']:<35} f1={best_2a_r['f1_macro']:.4f}  "
          f"rec_Withdrawn={best_2a_r.get('recall_Withdrawn',0):.4f}")

    # Save CSV
    rows = (
        [{"experiment":"Direct 4-class V2",    **baseline_direct}]
        + [{"experiment":r["label"], **r} for r in all_2b]
        + [{"experiment":r["label"], **r} for r in all_2a]
        + [{"experiment":"Final Hierarchical V2", **final_r}]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "branch_optimizer_results.csv", index=False)
    print(f"\n  Full results saved → {out_dir / 'branch_optimizer_results.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Branch Optimizer — 6-Phase Research Plan")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run only a specific phase (0 = all)")
    parser.add_argument("--skip-ext", action="store_true",
                        help="Skip external academic data experiment (Phase 3 step 3.3)")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)
    out_dir = ROOT / args.output_dir

    print("\n" + "="*65)
    print("  BRANCH OPTIMIZER — 6-Phase Research Plan")
    print("="*65)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print("\n--- Phase 1: Load V2 table & lock split ---")
    X_tr, X_te, y_tr, y_te = phase1_load()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if args.phase in (0, 2, 3, 4, 5):
        print("\n--- Phase 2: Build branch datasets ---")
        (X_tr_2a, X_te_2a, y_tr_2a, y_te_2a,
         X_tr_2b, X_te_2b, y_tr_2b, y_te_2b) = phase2_build_branches(
            X_tr, X_te, y_tr, y_te, out_dir / "branches")

    # ── Direct 4-class baseline (for comparison) ──────────────────────────────
    if args.phase in (0, 5):
        print("\n--- Baseline: Direct 4-class (V2) ---")
        le_base = LabelEncoder(); y_b_tr = le_base.fit_transform(y_tr)
        y_b_te  = le_base.transform(y_te)
        pre_b   = build_pre(X_tr)
        X_b_tr  = pre_b.fit_transform(X_tr).astype(np.float32)
        X_b_te  = pre_b.transform(X_te).astype(np.float32)
        clf_b   = best_clf(cw="balanced")
        clf_b.fit(X_b_tr, y_b_tr)
        preds_b = clf_b.predict(X_b_te)
        baseline_direct = {
            "accuracy":    accuracy_score(y_b_te, preds_b),
            "f1_macro":    f1_score(y_b_te, preds_b, average="macro", zero_division=0),
            "balanced_acc":balanced_accuracy_score(y_b_te, preds_b),
            "kappa":       cohen_kappa_score(y_b_te, preds_b),
        }
        print(f"  Baseline 4-class: acc={baseline_direct['accuracy']:.4f}  "
              f"f1={baseline_direct['f1_macro']:.4f}")

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    best_2b_r = best_2b_clf = pre_2b = le_2b = None
    all_2b    = []
    if args.phase in (0, 3):
        print("\n--- Phase 3: Optimize Model 2B ---")
        (best_2b_r, best_2b_clf, pre_2b, le_2b, all_2b) = phase3_optimize_2b(
            X_tr_2b, X_te_2b, y_tr_2b, y_te_2b,
            skip_ext=args.skip_ext)

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    best_2a_r = best_2a_clf = pre_2a = le_2a = None
    all_2a    = []
    if args.phase in (0, 4):
        print("\n--- Phase 4: Optimize Model 2A ---")
        (best_2a_r, best_2a_clf, pre_2a, le_2a, all_2a) = phase4_optimize_2a(
            X_tr_2a, X_te_2a, y_tr_2a, y_te_2a)

    # ── Phase 5 ───────────────────────────────────────────────────────────────
    final_r = None
    if args.phase in (0, 5) and best_2a_clf and best_2b_clf:
        print("\n--- Phase 5: End-to-end hierarchical evaluation ---")
        cols_2a = [c for c in FEAT_2A if c in X_tr.columns]
        cols_2b = [c for c in FEAT_2B if c in X_tr.columns]
        final_r = phase5_end_to_end(
            X_tr, X_te, y_tr, y_te,
            best_2a_clf, pre_2a, le_2a,
            best_2b_clf, pre_2b, le_2b,
            cols_2a, cols_2b)

    # ── Phase 6 ───────────────────────────────────────────────────────────────
    if args.phase in (0,) and final_r:
        print("\n--- Phase 6: Final comparison ---")
        phase6_table(baseline_direct, best_2b_r, best_2a_r,
                     final_r, all_2b, all_2a, out_dir)

    print("\n  Done.")


if __name__ == "__main__":
    main()
