#!/usr/bin/env python3
"""
Multi-Source Student Performance Pipeline
==========================================
Uses ALL available datasets to improve OULAD 4-class prediction.

Architecture — Option 3 (Auxiliary Score Transfer):
  Each external dataset trains an auxiliary model on its own features/target.
  The auxiliary model outputs a risk/performance SCORE that becomes a new
  feature fed into the main OULAD model.

Datasets used:
  OULAD V2        → main dataset (32,593 students)
  EdNet-KT2       → engagement depth model → ednet_engagement_score
  Dropout UCI     → academic risk model    → dropout_risk_score
  Mental Health   → wellbeing risk model   → mh_risk_score
  UCI Perf (mat)  → academic strength      → uci_academic_score
  UI Perf (por)   → academic strength      → uci_por_academic_score

Then:
  6 models trained on OULAD + auxiliary scores:
    LightGBM, XGBoost, CatBoost, Random Forest,
    Logistic Regression, Soft-Vote Ensemble

  Evaluated on:
    - Direct 4-class (V2 only, baseline)
    - Direct 4-class (V2 + auxiliary scores)
    - Hierarchical (best M1 + M2A + M2B), all with auxiliary features

Usage:
    python multi_source_pipeline.py
    python multi_source_pipeline.py --skip-ednet     # skip EdNet (slow)
    python multi_source_pipeline.py --ednet-users 5000
"""
from __future__ import annotations
import argparse, warnings, time, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

try:
    from lightgbm  import LGBMClassifier;     _LGB = True
except ImportError:
    LGBMClassifier = None; _LGB = False
try:
    from xgboost   import XGBClassifier;      _XGB = True
except ImportError:
    XGBClassifier = None; _XGB = False
try:
    from catboost  import CatBoostClassifier; _CAT = True
except ImportError:
    CatBoostClassifier = None; _CAT = False

from sklearn.model_selection   import (StratifiedKFold, cross_val_score,
                                        train_test_split)
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import (OneHotEncoder, LabelEncoder,
                                        StandardScaler, MinMaxScaler)
from sklearn.ensemble          import (RandomForestClassifier,
                                        GradientBoostingClassifier,
                                        VotingClassifier)
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import (accuracy_score, f1_score,
                                        balanced_accuracy_score,
                                        cohen_kappa_score, recall_score,
                                        classification_report, roc_auc_score)
from sklearn.preprocessing     import label_binarize

ROOT = Path(__file__).parent
SEED = 42
np.random.seed(SEED); random.seed(SEED)

FOUR_ORDER = ["Distinction", "Fail", "Pass", "Withdrawn"]
BINARY_MAP = {"Pass":"Success","Distinction":"Success",
              "Fail":"AtRisk","Withdrawn":"AtRisk"}
DROP_ALWAYS = {
    "final_result","id_student","code_module","code_presentation",
    "date_unregistration","date_unreg","date_unregistered","weighted_score",
    "first_ts","last_ts","active_weeks","clicks_per_active_week",
    "assessments_per_week","activity_count","days_active","avg_clicks_per_day",
    "week_click_sum_1_4","registration_delay_category",
    "id_assessment","id_site",
}

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def build_pre(X: pd.DataFrame, scale: bool = False) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(include=["object","category"]).columns.tolist()
    parts = []
    if num:
        steps = [("imp", SimpleImputer(strategy="median"))]
        if scale: steps.append(("sc", StandardScaler()))
        parts.append(("n", Pipeline(steps), num))
    if cat:
        parts.append(("c", Pipeline([
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
                         use_label_encoder=False, eval_metric="mlogloss",
                         random_state=SEED, n_jobs=-1, verbosity=0,
                         tree_method="hist")

def make_cat(cw=None, n=400):
    kw = dict(iterations=n, depth=7, learning_rate=0.05,
              l2_leaf_reg=3, verbose=0, random_state=SEED)
    if cw == "balanced": kw["auto_class_weights"] = "Balanced"
    return CatBoostClassifier(**kw)

def make_rf(cw=None):
    return RandomForestClassifier(n_estimators=400, class_weight=cw,
                                   random_state=SEED, n_jobs=-1)

def make_lr():
    return LogisticRegression(max_iter=2000, C=0.5, random_state=SEED,
                               class_weight="balanced", n_jobs=-1)

def all_models() -> Dict:
    models = {}
    if _LGB: models["LightGBM"]  = make_lgb(cw="balanced")
    if _XGB: models["XGBoost"]   = make_xgb()
    if _CAT: models["CatBoost"]  = make_cat(cw="balanced")
    models["RandomForest"]       = make_rf(cw="balanced")
    models["LogisticRegression"] = make_lr()
    # Soft-vote ensemble
    est = []
    if _LGB: est.append(("lgb", make_lgb(cw="balanced")))
    if _XGB: est.append(("xgb", make_xgb()))
    if _CAT: est.append(("cat", make_cat(cw="balanced")))
    if len(est) >= 2:
        models["SoftVoteEnsemble"] = VotingClassifier(est, voting="soft")
    return models


def evaluate(y_true, y_pred, y_proba, label: str) -> Dict:
    r = {
        "experiment":   label,
        "accuracy":     accuracy_score(y_true, y_pred),
        "f1_macro":     f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "kappa":        cohen_kappa_score(y_true, y_pred),
    }
    for cls in FOUR_ORDER:
        r[f"recall_{cls}"] = recall_score(y_true, y_pred, labels=[cls],
                                           average="macro", zero_division=0)
    try:
        yb  = label_binarize(y_true, classes=FOUR_ORDER)
        r["roc_auc"] = roc_auc_score(yb, y_proba, average="macro",
                                      multi_class="ovr")
    except Exception:
        r["roc_auc"] = float("nan")
    return r


def print_result(r: Dict):
    flag = " ★" if r["accuracy"] >= 0.80 else (" ↑" if r["accuracy"] >= 0.77 else "")
    print(f"  {r['experiment']:<45} "
          f"acc={r['accuracy']:.4f}  f1={r['f1_macro']:.4f}  "
          f"ba={r['balanced_acc']:.4f}  "
          f"Dist={r.get('recall_Distinction',0):.3f}  "
          f"Fail={r.get('recall_Fail',0):.3f}{flag}")

# ══════════════════════════════════════════════════════════════════════════════
# AUXILIARY MODEL 1 — EdNet-KT2: Engagement Depth Score
# ══════════════════════════════════════════════════════════════════════════════
# EdNet has no outcome labels compatible with OULAD.
# Strategy: aggregate per-user engagement features → train a self-supervised
# engagement-tier classifier (Low/Mid/High) → use predicted probabilities as
# a continuous engagement_depth_score for OULAD students (via a regressor
# trained on OULAD's own VLE features to predict the EdNet-calibrated score).

def build_ednet_engagement_model(kt2_dir: Path, n_users: int = 5000,
                                  seed: int = SEED) -> Optional[object]:
    """
    Sample n_users from EdNet-KT2, compute per-user features,
    define engagement tier (Low/Mid/High) by interaction count quantile,
    train a classifier, return (clf, feature_cols, scaler).
    """
    print(f"  Loading EdNet-KT2 ({n_users} users)...")
    files = list(kt2_dir.glob("*.csv"))
    if not files:
        print("  EdNet-KT2 files not found.")
        return None

    rng = random.Random(seed)
    sample_files = rng.sample(files, min(n_users, len(files)))

    rows = []
    for f in sample_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "action_type" not in df.columns: continue
            uid = f.stem

            responds = df[df["action_type"] == "respond"]
            enters   = df[df["action_type"] == "enter"]
            n_int    = len(df)
            n_resp   = len(responds)
            n_enter  = max(1, len(enters))
            completion_rate = n_resp / n_enter

            # source diversity
            n_sources = df["source"].nunique() if "source" in df.columns else 1
            # platform
            n_platforms = df["platform"].nunique() if "platform" in df.columns else 1
            # time span
            ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
            span_hrs = float((ts.max()-ts.min())/1e3/3600) if len(ts) > 1 else 0.0
            # session intensity (interactions per hour)
            intensity = n_int / max(span_hrs, 0.01)
            # unique items
            n_items = df["item_id"].nunique() if "item_id" in df.columns else 1
            # respond rate
            resp_rate = n_resp / max(n_int, 1)

            rows.append({
                "uid":              uid,
                "n_interactions":   n_int,
                "n_responds":       n_resp,
                "completion_rate":  min(completion_rate, 5.0),
                "n_sources":        n_sources,
                "n_platforms":      n_platforms,
                "span_hrs":         min(span_hrs, 10000.0),
                "intensity":        min(intensity, 500.0),
                "n_items":          n_items,
                "resp_rate":        resp_rate,
            })
        except Exception:
            continue

    if not rows:
        return None

    df_agg = pd.DataFrame(rows).dropna()
    print(f"  EdNet aggregated: {len(df_agg):,} users")

    # Define engagement tiers by n_interactions quantile
    q33 = df_agg["n_interactions"].quantile(0.33)
    q66 = df_agg["n_interactions"].quantile(0.66)
    df_agg["tier"] = 0  # Low
    df_agg.loc[df_agg["n_interactions"] >= q33, "tier"] = 1  # Mid
    df_agg.loc[df_agg["n_interactions"] >= q66, "tier"] = 2  # High

    feat_cols = ["n_interactions","n_responds","completion_rate","n_sources",
                 "n_platforms","span_hrs","intensity","n_items","resp_rate"]
    X_e = df_agg[feat_cols].values
    y_e = df_agg["tier"].values

    scaler = StandardScaler()
    X_e_s  = scaler.fit_transform(X_e)

    clf = make_lgb() if _LGB else make_rf()
    clf.fit(X_e_s, y_e)

    # CV score
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sc  = cross_val_score(clf, X_e_s, y_e, cv=cv, scoring="f1_macro", n_jobs=1)
    print(f"  EdNet engagement classifier CV f1_macro: {sc.mean():.4f} ± {sc.std():.4f}")

    return clf, feat_cols, scaler, q33, q66


def ednet_score_for_oulad(oulad_df: pd.DataFrame,
                           ednet_model) -> pd.Series:
    """
    Map OULAD VLE features to EdNet engagement tier probabilities.
    Returns a single score [0-2] representing engagement depth.
    """
    clf, feat_cols, scaler, q33, q66 = ednet_model

    # Map OULAD features → EdNet feature space
    # n_interactions ↔ total_clicks_v2
    # n_responds ↔ n_submissions
    # completion_rate ↔ submission_ratio
    # n_sources ↔ unique_resources_accessed
    # n_platforms → 1 (OULAD is single platform)
    # span_hrs ↔ days_active_span / 24
    # intensity ↔ total_clicks_v2 / (days_active_span/24 + 0.01)
    # n_items ↔ unique_resources_accessed
    # resp_rate ↔ submission_ratio

    tcv  = oulad_df.get("total_clicks_v2",    pd.Series(0, index=oulad_df.index)).fillna(0)
    nsub = oulad_df.get("n_submissions",       pd.Series(0, index=oulad_df.index)).fillna(0)
    srat = oulad_df.get("submission_ratio",    pd.Series(0, index=oulad_df.index)).fillna(0)
    ures = oulad_df.get("unique_resources_accessed", pd.Series(1, index=oulad_df.index)).fillna(1)
    span = oulad_df.get("days_active_span",    pd.Series(0, index=oulad_df.index)).fillna(0) / 24
    intens = tcv / (span.replace(0, 0.01))
    intens = intens.clip(upper=500)

    mapped = pd.DataFrame({
        "n_interactions":  tcv.clip(upper=9000),
        "n_responds":      nsub,
        "completion_rate": srat.clip(upper=5),
        "n_sources":       ures.clip(upper=6),
        "n_platforms":     1.0,
        "span_hrs":        span.clip(upper=10000),
        "intensity":       intens,
        "n_items":         ures.clip(upper=9000),
        "resp_rate":       srat.clip(upper=1),
    })

    X_m = scaler.transform(mapped[feat_cols].fillna(0).values)
    proba = clf.predict_proba(X_m)
    # weighted score: 0*P(Low) + 1*P(Mid) + 2*P(High)
    score = proba @ np.array([0.0, 1.0, 2.0])
    return pd.Series(score, index=oulad_df.index, name="ednet_engagement_score")

# ══════════════════════════════════════════════════════════════════════════════
# AUXILIARY MODEL 2 — Dropout UCI: Academic Risk Score
# ══════════════════════════════════════════════════════════════════════════════

def build_dropout_risk_model(path: Path) -> Optional[Tuple]:
    """
    Train a binary risk model on UCI Dropout dataset.
    Dropout → AtRisk=1, Graduate → AtRisk=0 (Enrolled excluded).
    Returns (clf, feature_cols, preprocessor).
    """
    if not path.exists():
        print("  Dropout dataset not found.")
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "Target" not in df.columns:
        return None

    # Keep only Dropout vs Graduate (cleaner signal)
    df = df[df["Target"].isin(["Dropout","Graduate"])].copy()
    df["risk_label"] = (df["Target"] == "Dropout").astype(int)

    print(f"  Dropout dataset: {len(df):,} rows  "
          f"(Dropout={( df['risk_label']==1).sum()}  Graduate={(df['risk_label']==0).sum()})")

    # Feature engineering
    for sem in ["1st","2nd"]:
        enr = f"Curricular units {sem} sem (enrolled)"
        apr = f"Curricular units {sem} sem (approved)"
        grd = f"Curricular units {sem} sem (grade)"
        if enr in df.columns and apr in df.columns:
            denom = pd.to_numeric(df[enr], errors="coerce").replace(0, np.nan)
            df[f"approval_rate_{sem}"] = (pd.to_numeric(df[apr], errors="coerce") / denom).fillna(0).clip(0,1)
        if grd in df.columns:
            df[f"log_grade_{sem}"] = np.log1p(pd.to_numeric(df[grd], errors="coerce").fillna(0))

    g1 = "Curricular units 1st sem (grade)"
    g2 = "Curricular units 2nd sem (grade)"
    if g1 in df.columns and g2 in df.columns:
        v1 = pd.to_numeric(df[g1], errors="coerce").fillna(0)
        v2 = pd.to_numeric(df[g2], errors="coerce").fillna(0)
        df["grade_improvement"] = v2 - v1
        df["avg_grade"]         = (v1 + v2) / 2

    if "Debtor" in df.columns and "Tuition fees up to date" in df.columns:
        df["financial_risk"] = (
            (pd.to_numeric(df["Debtor"], errors="coerce").fillna(0) == 1) |
            (pd.to_numeric(df["Tuition fees up to date"], errors="coerce").fillna(1) == 0)
        ).astype(int)

    drop_cols = {"Target","risk_label"}
    X = df.drop(columns=list(drop_cols), errors="ignore")
    y = df["risk_label"].values

    pre = build_pre(X)
    X_t = pre.fit_transform(X)

    clf = make_lgb() if _LGB else make_rf()
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sc  = cross_val_score(clf, X_t, y, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"  Dropout risk model CV ROC-AUC: {sc.mean():.4f} ± {sc.std():.4f}")
    clf.fit(X_t, y)

    return clf, X.columns.tolist(), pre


def dropout_risk_score(oulad_df: pd.DataFrame, model) -> pd.Series:
    """
    Map OULAD features → Dropout-space features → predict P(dropout/atrisk).
    Common schema: academic performance, prior attempts, financial proxy.
    """
    clf, feat_cols, pre = model
    # Build a mapping from OULAD → Dropout feature space
    rows = pd.DataFrame(index=oulad_df.index)

    # Approval rate proxy
    if "assessment_completion_ratio" in oulad_df.columns:
        rows["approval_rate_1st"] = oulad_df["assessment_completion_ratio"].fillna(0).clip(0,1)
        rows["approval_rate_2nd"] = oulad_df["assessment_completion_ratio"].fillna(0).clip(0,1)
    if "avg_score_v2" in oulad_df.columns:
        rows["avg_grade"]    = oulad_df["avg_score_v2"].fillna(0) / 20  # scale 0-5
        rows["log_grade_1st"]= np.log1p(oulad_df["avg_score_v2"].fillna(0))
        rows["log_grade_2nd"]= np.log1p(oulad_df["avg_score_v2"].fillna(0))
    if "score_trend" in oulad_df.columns:
        rows["grade_improvement"] = oulad_df["score_trend"].fillna(0)
    if "num_of_prev_attempts" in oulad_df.columns:
        rows["financial_risk"] = (oulad_df["num_of_prev_attempts"].fillna(0) > 0).astype(int)
    if "studied_credits" in oulad_df.columns:
        rows["studied_credits"] = oulad_df["studied_credits"].fillna(60)

    # fill any remaining dropout feature columns with 0
    for fc in feat_cols:
        if fc not in rows.columns:
            rows[fc] = 0

    try:
        X_t = pre.transform(rows[feat_cols])
    except Exception:
        X_t = pre.transform(rows.reindex(columns=feat_cols, fill_value=0))

    proba = clf.predict_proba(X_t)[:, 1]  # P(AtRisk)
    return pd.Series(proba, index=oulad_df.index, name="dropout_risk_score")


# ══════════════════════════════════════════════════════════════════════════════
# AUXILIARY MODEL 3 — Mental Health: Wellbeing Risk Score
# ══════════════════════════════════════════════════════════════════════════════

def build_mh_model(path: Path) -> Optional[Tuple]:
    """
    Train a mental health risk model.
    Target: any of Depression/Anxiety/Panic = 1 → at-risk.
    """
    if not path.exists():
        print("  Mental Health dataset not found.")
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Rename
    rename = {
        "Choose your gender": "gender",
        "What is your course?": "course",
        "Your current year of Study": "study_year",
        "What is your CGPA?": "cgpa",
        "Marital status": "marital",
        "Do you have Depression?": "depression",
        "Do you have Anxiety?": "anxiety",
        "Do you have Panic attack?": "panic",
        "Did you seek any specialist for a treatment?": "treatment",
    }
    df = df.rename(columns=rename).drop(columns=["Timestamp"], errors="ignore")

    # target: any mental health flag
    for col in ["depression","anxiety","panic"]:
        if col in df.columns:
            df[col] = (df[col].str.strip().str.lower() == "yes").astype(int)

    present = [c for c in ["depression","anxiety","panic"] if c in df.columns]
    if not present:
        return None
    df["mh_risk"] = (df[present].sum(axis=1) > 0).astype(int)

    # CGPA → numeric
    def cgpa_num(s):
        s = str(s)
        if "3.50" in s or "4.0" in s: return 3.75
        if "3.00" in s or "3.0" in s: return 3.25
        if "2.50" in s: return 2.75
        return 2.0
    df["cgpa_num"] = df["cgpa"].apply(cgpa_num)

    X = df[["gender","cgpa_num","study_year","marital"] +
            [c for c in ["treatment"] if c in df.columns]].copy()
    y = df["mh_risk"].values

    print(f"  Mental Health: {len(df)} rows  at-risk={(y==1).sum()}  ok={(y==0).sum()}")
    pre = build_pre(X)
    X_t = pre.fit_transform(X)
    clf = make_rf(cw="balanced")
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sc  = cross_val_score(clf, X_t, y, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"  MH risk model CV ROC-AUC: {sc.mean():.4f} ± {sc.std():.4f}")
    clf.fit(X_t, y)
    return clf, X.columns.tolist(), pre


def mh_risk_score(oulad_df: pd.DataFrame, mh_model) -> pd.Series:
    """Map OULAD demographics → MH risk score."""
    clf, feat_cols, pre = mh_model
    rows = pd.DataFrame(index=oulad_df.index)
    if "gender" in oulad_df.columns:
        rows["gender"] = oulad_df["gender"]
    if "age_band" in oulad_df.columns:
        rows["study_year"] = oulad_df["age_band"]
    if "highest_education" in oulad_df.columns:
        rows["marital"] = oulad_df["highest_education"]
    # CGPA proxy from avg_score
    if "avg_score_v2" in oulad_df.columns:
        rows["cgpa_num"] = oulad_df["avg_score_v2"].fillna(50) / 25  # scale
    else:
        rows["cgpa_num"] = 2.5
    rows["treatment"] = "No"

    for fc in feat_cols:
        if fc not in rows.columns:
            rows[fc] = "unknown" if rows.get(fc, pd.Series(dtype=str)).dtype == object else 0

    try:
        X_t = pre.transform(rows.reindex(columns=feat_cols, fill_value=0))
    except Exception:
        X_t = np.zeros((len(rows), len(feat_cols)))

    proba = clf.predict_proba(X_t)[:, 1]
    return pd.Series(proba, index=oulad_df.index, name="mh_risk_score")


# ══════════════════════════════════════════════════════════════════════════════
# AUXILIARY MODEL 4 — UCI Student Performance: Academic Strength Score
# ══════════════════════════════════════════════════════════════════════════════

def build_uci_academic_model(mat_path: Path, por_path: Path) -> Optional[Tuple]:
    """
    Train an academic strength model on UCI mat+por datasets.
    Target: G3 → Low/Mid/High performance tier.
    """
    dfs = []
    for p in [mat_path, por_path]:
        if p.exists():
            try:
                d = pd.read_csv(p, sep=";"); d.columns = d.columns.str.strip()
                if "G3" in d.columns: dfs.append(d)
            except Exception: pass

    if not dfs:
        print("  UCI Student Performance not found.")
        return None

    df = pd.concat(dfs, ignore_index=True)
    df["G3"] = pd.to_numeric(df["G3"], errors="coerce").fillna(0)
    df["acad_tier"] = pd.cut(df["G3"], bins=[-1, 9, 14, 20],
                              labels=[0, 1, 2]).astype(int)  # 0=Low 1=Mid 2=High

    print(f"  UCI Student Perf: {len(df):,} rows  "
          f"tier dist: {df['acad_tier'].value_counts().sort_index().to_dict()}")

    # Feature engineering
    if "Medu" in df.columns and "Fedu" in df.columns:
        df["parent_edu"] = pd.to_numeric(df["Medu"],errors="coerce").fillna(0) + \
                           pd.to_numeric(df["Fedu"],errors="coerce").fillna(0)
    if "Dalc" in df.columns and "Walc" in df.columns:
        df["total_alc"] = pd.to_numeric(df["Dalc"],errors="coerce").fillna(0) + \
                          pd.to_numeric(df["Walc"],errors="coerce").fillna(0)
    if "absences" in df.columns:
        df["log_absences"] = np.log1p(pd.to_numeric(df["absences"],errors="coerce").fillna(0))
    if "studytime" in df.columns and "failures" in df.columns:
        df["study_eff"] = pd.to_numeric(df["studytime"],errors="coerce").fillna(1) / \
                          (pd.to_numeric(df["failures"],errors="coerce").fillna(0)+1)

    drop_cols = {"G1","G2","G3","acad_tier"}
    X = df.drop(columns=list(drop_cols), errors="ignore")
    y = df["acad_tier"].values

    pre = build_pre(X)
    X_t = pre.fit_transform(X)
    clf = make_lgb() if _LGB else make_rf()
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sc  = cross_val_score(clf, X_t, y, cv=cv, scoring="f1_macro", n_jobs=1)
    print(f"  UCI academic model CV f1_macro: {sc.mean():.4f} ± {sc.std():.4f}")
    clf.fit(X_t, y)
    return clf, X.columns.tolist(), pre


def uci_academic_score(oulad_df: pd.DataFrame, uci_model) -> pd.Series:
    """Map OULAD features → UCI academic tier probabilities → score 0-2."""
    clf, feat_cols, pre = uci_model
    rows = pd.DataFrame(index=oulad_df.index)

    # Map OULAD → UCI feature space
    if "gender" in oulad_df.columns:
        rows["sex"] = oulad_df["gender"].map({"M":"M","F":"F"}).fillna("F")
    if "num_of_prev_attempts" in oulad_df.columns:
        rows["failures"] = oulad_df["num_of_prev_attempts"].fillna(0).astype(int)
    if "studied_credits" in oulad_df.columns:
        rows["studytime"] = (oulad_df["studied_credits"].fillna(60) / 30).clip(1,4).astype(int)
    if "avg_score_v2" in oulad_df.columns:
        s = oulad_df["avg_score_v2"].fillna(50)
        rows["G1"] = (s * 0.2).round().astype(int)   # scale 100→20
        rows["G2"] = (s * 0.2).round().astype(int)
        rows["absences"] = (100 - s.clip(0,100)).round().astype(int) // 5
    if "highest_education" in oulad_df.columns:
        rows["Medu"] = oulad_df["highest_education"].map({
            "No Formal quals":0,"Lower Than A Level":1,
            "A Level or Equivalent":2,"HE Qualification":3,
            "Post Graduate Qualification":4}).fillna(1).astype(int)
        rows["Fedu"] = rows["Medu"]
    if "imd_band" in oulad_df.columns:
        imd_map = {"0-10%":1,"10-20%":2,"20-30%":3,"30-40%":4,"40-50%":5,
                   "50-60%":6,"60-70%":7,"70-80%":8,"80-90%":9,"90-100%":10}
        imd_num = oulad_df["imd_band"].map(imd_map).fillna(5)
        rows["total_alc"]  = (10 - imd_num).clip(0,8).astype(int) // 2
        rows["famrel"]     = (imd_num / 2).clip(1,5).astype(int)
    rows["log_absences"] = np.log1p(rows.get("absences", pd.Series(5, index=rows.index)))
    rows["study_eff"]    = rows.get("studytime", pd.Series(2, index=rows.index)) / \
                           (rows.get("failures", pd.Series(0, index=rows.index)) + 1)

    for fc in feat_cols:
        if fc not in rows.columns:
            rows[fc] = 0

    try:
        X_t = pre.transform(rows.reindex(columns=feat_cols, fill_value=0))
    except Exception:
        X_t = np.zeros((len(rows), len(feat_cols)))

    proba = clf.predict_proba(X_t)
    score = proba @ np.arange(proba.shape[1], dtype=float)
    return pd.Series(score, index=oulad_df.index, name="uci_academic_score")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN OULAD FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_oulad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop = DROP_ALWAYS - {"final_result"}
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

    # flag corrections (silence_onset_week was all-zeros — use last_active_week)
    if "last_active_week" in df.columns:
        df["went_silent_flag"] = (df["last_active_week"] < 28).astype(int)
        df["early_dropout"]    = (df["last_active_week"] < 10).astype(int)
        df["mid_dropout"]      = ((df["last_active_week"] >= 10) &
                                   (df["last_active_week"] < 22)).astype(int)
        df["late_active"]      = (df["last_active_week"] >= 28).astype(int)
        df["days_since_last"]  = (250 - df["last_active_day"].fillna(0).clip(upper=250))

    zero_cols = ([f"week{w}_clicks" for w in range(1,13)] +
                 ["total_clicks_v2","avg_score_v2","score_trend","score_volatility",
                  "n_submissions","submission_ratio","assessment_completion_ratio",
                  "late_submission_count_v2","assessment_submission_span",
                  "tma_count","cma_count","first_tma_submitted"])
    for c in zero_cols:
        if c in df.columns: df[c] = df[c].fillna(0)

    if "total_clicks_v2" in df.columns:
        df["log_total_clicks"] = np.log1p(df["total_clicks_v2"])
        df["zero_clicks"]      = (df["total_clicks_v2"] == 0).astype(int)

    if "avg_score_v2" in df.columns:
        df["high_scorer"]      = (df["avg_score_v2"] >= 70).astype(int)
        df["very_high_scorer"] = (df["avg_score_v2"] >= 80).astype(int)
    if "avg_score_v2" in df.columns and "submission_ratio" in df.columns:
        df["score_x_completion"] = df["avg_score_v2"] * df["submission_ratio"]
    if "max_score" in df.columns and "min_score" in df.columns:
        df["score_range"] = df["max_score"] - df["min_score"]
    if "last_assessment_score" in df.columns and "first_assessment_score" in df.columns:
        df["score_improvement"] = df["last_assessment_score"] - df["first_assessment_score"]
    if "submission_ratio" in df.columns:
        df["late_ratio"] = (df["late_submission_count_v2"] /
                             df["n_submissions"].replace(0, np.nan)).fillna(0)
    if "active_weeks_v2" in df.columns and "last_active_week" in df.columns:
        denom = (df["last_active_week"] - df["first_active_week"].fillna(0)).replace(0, np.nan)
        df["active_week_density"] = (df["active_weeks_v2"] / denom).fillna(0).clip(0,1)
    if "imd_band" in df.columns:
        imd = {"0-10%":1,"10-20":2,"10-20%":2,"20-30%":3,"30-40%":4,"40-50%":5,
               "50-60%":6,"60-70%":7,"70-80%":8,"80-90%":9,"90-100%":10}
        df["imd_numeric"] = df["imd_band"].map(imd).fillna(5)
    if "num_of_prev_attempts" in df.columns:
        df["is_repeat"] = (df["num_of_prev_attempts"] > 0).astype(int)

    return df


def attach_auxiliary_scores(df: pd.DataFrame, aux_scores: Dict[str, pd.Series]) -> pd.DataFrame:
    """Add all auxiliary scores as new columns."""
    for name, scores in aux_scores.items():
        df[name] = scores.reindex(df.index).fillna(scores.median())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL 6 MODELS
# ══════════════════════════════════════════════════════════════════════════════

def run_6_models(X_tr: pd.DataFrame, y_tr: pd.Series,
                 X_te: pd.DataFrame, y_te: pd.Series,
                 experiment_label: str, le: LabelEncoder) -> List[Dict]:
    """Train and evaluate all 6 models. Returns list of result dicts."""
    pre = build_pre(X_tr)
    X_tr_t = pre.fit_transform(X_tr).astype(np.float32)
    X_te_t = pre.transform(X_te).astype(np.float32)

    y_tr_e = le.transform(y_tr)
    y_te_e = le.transform(y_te)

    results = []
    models  = all_models()

    for mname, clf in models.items():
        t0 = time.time()
        clf.fit(X_tr_t, y_tr_e)
        preds = clf.predict(X_te_t)
        # remap to string labels
        preds_str = le.inverse_transform(preds)
        y_te_str  = le.inverse_transform(y_te_e)

        try:
            proba = clf.predict_proba(X_te_t)
            # reorder columns to FOUR_ORDER
            cls_list = list(le.classes_)
            proba_ord = np.column_stack([
                proba[:, cls_list.index(c)] if c in cls_list
                else np.zeros(len(proba))
                for c in FOUR_ORDER
            ])
        except Exception:
            proba_ord = None

        r = evaluate(y_te_str, preds_str, proba_ord, f"{experiment_label} | {mname}")
        r["model"] = mname
        r["time_s"] = round(time.time()-t0, 1)
        results.append(r)
        print_result(r)

    return results

# ══════════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_master_table(all_results: List[Dict]):
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 115)
    print(f"  {'EXPERIMENT + MODEL':<55} {'ACC':>6} {'F1-M':>6} {'BA':>6} "
          f"{'KAPPA':>6} {'ROC':>6}  {'Dist':>5} {'Fail':>5} {'Pass':>5} {'With':>5}")
    print("=" * 115)

    baseline_acc = None
    for _, row in df.sort_values(["experiment","accuracy"],
                                  ascending=[True, False]).iterrows():
        exp = str(row.get("experiment",""))
        if baseline_acc is None:
            baseline_acc = row["accuracy"]

        delta = row["accuracy"] - baseline_acc
        flag  = " ★" if row["accuracy"] >= 0.80 else \
                (" ↑" if row["accuracy"] >= 0.77 else "")
        sign  = f"+{delta:.4f}" if delta > 0 else (
                f"({delta:.4f})" if delta < 0 else "")

        print(f"  {exp[:55]:<55} "
              f"{row['accuracy']:>6.4f} {row['f1_macro']:>6.4f} "
              f"{row['balanced_acc']:>6.4f} {row.get('kappa',0):>6.4f} "
              f"{row.get('roc_auc', float('nan')):>6.4f}  "
              f"{row.get('recall_Distinction',0):>5.3f} "
              f"{row.get('recall_Fail',0):>5.3f} "
              f"{row.get('recall_Pass',0):>5.3f} "
              f"{row.get('recall_Withdrawn',0):>5.3f}"
              f"  {sign}{flag}")
    print("=" * 115)

    # Best per experiment group
    print("\n  Best per experiment:")
    for grp in df["experiment"].unique():
        sub  = df[df["experiment"] == grp]
        best = sub.loc[sub["accuracy"].idxmax()]
        star = " ★" if best["accuracy"] >= 0.80 else ""
        print(f"    {grp:<35} → {best.get('model',''):18} "
              f"acc={best['accuracy']:.4f}  f1={best['f1_macro']:.4f}"
              f"  Dist_recall={best.get('recall_Distinction',0):.3f}{star}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Source Student Performance Pipeline — 6 Models")
    parser.add_argument("--skip-ednet",    action="store_true",
                        help="Skip EdNet-KT2 auxiliary model (fast mode)")
    parser.add_argument("--ednet-users",   type=int, default=5000,
                        help="Number of EdNet users to sample (default 5000)")
    parser.add_argument("--output-dir",    default="results")
    args = parser.parse_args()

    import logging; logging.basicConfig(level=logging.WARNING)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  MULTI-SOURCE PIPELINE — 6 Models + Auxiliary Scores")
    print("="*65)

    # ── Load OULAD V2 ─────────────────────────────────────────────────────────
    v2_path = ROOT / "oulad_ml_table_v2.csv"
    if not v2_path.exists():
        raise FileNotFoundError("oulad_ml_table_v2.csv not found. Run oulad_pipeline_v2.py first.")
    df_raw = pd.read_csv(v2_path); df_raw.columns = df_raw.columns.str.strip()

    df = engineer_oulad(df_raw)
    y  = df["final_result"].dropna()
    X  = df.drop(columns=["final_result"], errors="ignore").loc[y.index]
    X  = X.dropna(axis=1, how="all")

    print(f"\n  OULAD V2: {len(X):,} rows  {X.shape[1]} features")
    print(f"  Target: {y.value_counts().to_dict()}")

    # LOCKED train/test split — same seed always
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")
    print(f"  Test dist: {y_te.value_counts().to_dict()}")

    le = LabelEncoder(); le.fit(FOUR_ORDER)

    all_results: List[Dict] = []

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT A: Direct 4-class — OULAD V2 only (baseline)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  EXPERIMENT A — Direct 4-class OULAD V2 (baseline, 6 models)")
    print(f"{'─'*65}")
    res_a = run_6_models(X_tr, y_tr, X_te, y_te, "A: OULAD-V2 baseline", le)
    all_results.extend(res_a)

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD AUXILIARY MODELS
    # ══════════════════════════════════════════════════════════════════════════
    aux_scores_tr: Dict[str, pd.Series] = {}
    aux_scores_te: Dict[str, pd.Series] = {}

    # ── Auxiliary 1: EdNet ────────────────────────────────────────────────────
    if not args.skip_ednet:
        kt2_dir = ROOT / "EdNet-KT2" / "KT2"
        if kt2_dir.exists():
            print(f"\n{'─'*65}")
            print("  Building Auxiliary 1 — EdNet-KT2 Engagement Model")
            print(f"{'─'*65}")
            ednet_model = build_ednet_engagement_model(
                kt2_dir, n_users=args.ednet_users)
            if ednet_model:
                aux_scores_tr["ednet_engagement_score"] = ednet_score_for_oulad(X_tr, ednet_model)
                aux_scores_te["ednet_engagement_score"] = ednet_score_for_oulad(X_te, ednet_model)
                print(f"  EdNet score added — train mean: "
                      f"{aux_scores_tr['ednet_engagement_score'].mean():.3f}")

    # ── Auxiliary 2: Dropout risk ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Building Auxiliary 2 — Dropout UCI Risk Model")
    print(f"{'─'*65}")
    drop_model = build_dropout_risk_model(ROOT / "dropout" / "data.csv")
    if drop_model:
        aux_scores_tr["dropout_risk_score"] = dropout_risk_score(X_tr, drop_model)
        aux_scores_te["dropout_risk_score"] = dropout_risk_score(X_te, drop_model)
        print(f"  Dropout score added — train mean: "
              f"{aux_scores_tr['dropout_risk_score'].mean():.3f}")

    # ── Auxiliary 3: Mental Health risk ──────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Building Auxiliary 3 — Mental Health Risk Model")
    print(f"{'─'*65}")
    mh_model = build_mh_model(ROOT / "Student Mental health.csv")
    if mh_model:
        aux_scores_tr["mh_risk_score"] = mh_risk_score(X_tr, mh_model)
        aux_scores_te["mh_risk_score"] = mh_risk_score(X_te, mh_model)
        print(f"  MH risk score added — train mean: "
              f"{aux_scores_tr['mh_risk_score'].mean():.3f}")

    # ── Auxiliary 4: UCI Academic strength ───────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Building Auxiliary 4 — UCI Academic Strength Model")
    print(f"{'─'*65}")
    uci_model = build_uci_academic_model(
        ROOT / "student+performance/student/student-mat.csv",
        ROOT / "UI_student+performance/student/student-por.csv")
    if uci_model:
        aux_scores_tr["uci_academic_score"] = uci_academic_score(X_tr, uci_model)
        aux_scores_te["uci_academic_score"] = uci_academic_score(X_te, uci_model)
        print(f"  UCI academic score added — train mean: "
              f"{aux_scores_tr['uci_academic_score'].mean():.3f}")

    if not aux_scores_tr:
        print("\n  No auxiliary scores built — skipping Experiment B.")
    else:
        # ══════════════════════════════════════════════════════════════════════
        # EXPERIMENT B: OULAD V2 + all auxiliary scores (6 models)
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n{'─'*65}")
        print(f"  EXPERIMENT B — OULAD V2 + {len(aux_scores_tr)} auxiliary scores")
        print(f"  Scores: {list(aux_scores_tr.keys())}")
        print(f"{'─'*65}")

        X_tr_b = attach_auxiliary_scores(X_tr.copy(), aux_scores_tr)
        X_te_b = attach_auxiliary_scores(X_te.copy(), aux_scores_te)

        res_b = run_6_models(X_tr_b, y_tr, X_te_b, y_te,
                              "B: OULAD-V2 + AuxScores", le)
        all_results.extend(res_b)

        # ══════════════════════════════════════════════════════════════════════
        # EXPERIMENT C: Hierarchical with auxiliary scores — probabilistic fusion
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n{'─'*65}")
        print("  EXPERIMENT C — Hierarchical + Auxiliary Scores (best model per branch)")
        print(f"{'─'*65}")

        X_tr_c = X_tr_b.copy(); X_te_c = X_te_b.copy()

        # Model 1: binary
        y_bin_tr = y_tr.map(BINARY_MAP)
        y_bin_te = y_te.map(BINARY_MAP)
        le1 = LabelEncoder(); le1.fit(["AtRisk","Success"])
        y1_tr = le1.transform(y_bin_tr); y1_te = le1.transform(y_bin_te)
        pre1 = build_pre(X_tr_c)
        X1_tr = pre1.fit_transform(X_tr_c).astype(np.float32)
        X1_te = pre1.transform(X_te_c).astype(np.float32)
        m1 = make_lgb() if _LGB else make_rf(); m1.fit(X1_tr, y1_tr)
        p1    = m1.predict_proba(X1_te)
        p_ar  = p1[:, list(le1.classes_).index("AtRisk")]
        p_suc = p1[:, list(le1.classes_).index("Success")]
        m1_acc = accuracy_score(y1_te, m1.predict(X1_te))
        print(f"  M1 (AtRisk/Success): test acc={m1_acc:.4f}")

        # Model 2A: Fail vs Withdrawn
        mask_tr_2a = y_tr.isin(["Fail","Withdrawn"])
        mask_te_2a = y_te.isin(["Fail","Withdrawn"])
        le2a = LabelEncoder(); le2a.fit(["Fail","Withdrawn"])
        X2a_tr = X_tr_c.loc[mask_tr_2a]; y2a_tr = le2a.transform(y_tr.loc[mask_tr_2a])
        X2a_te = X_te_c                 ; y2a_te = le2a.transform(y_te.loc[mask_te_2a])
        pre2a = build_pre(X2a_tr)
        X2a_tr_t = pre2a.fit_transform(X2a_tr).astype(np.float32)
        X2a_te_t = pre2a.transform(X2a_te).astype(np.float32)
        m2a = make_lgb(cw="balanced") if _LGB else make_rf(cw="balanced")
        m2a.fit(X2a_tr_t, y2a_tr)
        p2a = m2a.predict_proba(X2a_te_t)
        p_fail = p2a[:, list(le2a.classes_).index("Fail")]
        p_with = p2a[:, list(le2a.classes_).index("Withdrawn")]
        m2a_acc = accuracy_score(y2a_te, m2a.predict(
            pre2a.transform(X_te_c.loc[mask_te_2a]).astype(np.float32)))
        print(f"  M2A (Fail/Withdrawn): test acc={m2a_acc:.4f}")

        # Model 2B: Pass vs Distinction
        mask_tr_2b = y_tr.isin(["Pass","Distinction"])
        mask_te_2b = y_te.isin(["Pass","Distinction"])
        le2b = LabelEncoder(); le2b.fit(["Distinction","Pass"])
        X2b_tr = X_tr_c.loc[mask_tr_2b]; y2b_tr = le2b.transform(y_tr.loc[mask_tr_2b])
        X2b_te = X_te_c
        pre2b = build_pre(X2b_tr)
        X2b_tr_t = pre2b.fit_transform(X2b_tr).astype(np.float32)
        X2b_te_t = pre2b.transform(X2b_te).astype(np.float32)
        m2b = make_cat(cw="balanced") if _CAT else make_lgb(cw="balanced")
        m2b.fit(X2b_tr_t, y2b_tr)
        p2b = m2b.predict_proba(X2b_te_t)
        p_dist = p2b[:, list(le2b.classes_).index("Distinction")]
        p_pass = p2b[:, list(le2b.classes_).index("Pass")]
        m2b_acc = accuracy_score(
            le2b.transform(y_te.loc[mask_te_2b]),
            m2b.predict(pre2b.transform(X_te_c.loc[mask_te_2b]).astype(np.float32)))
        print(f"  M2B (Pass/Distinction): test acc={m2b_acc:.4f}")

        # Probabilistic fusion
        p_final = np.column_stack([
            p_suc * p_dist,  # Distinction
            p_ar  * p_fail,  # Fail
            p_suc * p_pass,  # Pass
            p_ar  * p_with,  # Withdrawn
        ])
        p_final /= np.where(p_final.sum(axis=1, keepdims=True) > 0,
                             p_final.sum(axis=1, keepdims=True), 1)
        y_pred_c = np.array(FOUR_ORDER)[np.argmax(p_final, axis=1)]
        y_true_c = y_te.values

        r_c = evaluate(y_true_c, y_pred_c, p_final,
                       "C: Hierarchical + AuxScores")
        r_c["model"] = "HierarchicalFusion"
        all_results.append(r_c)
        print_result(r_c)
        print(f"\n  Classification report:")
        print("  " + classification_report(y_true_c, y_pred_c,
                                            labels=FOUR_ORDER, zero_division=0
                                            ).replace("\n","\n  "))

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print_master_table(all_results)

    # Save
    pd.DataFrame(all_results).to_csv(
        out_dir / "multi_source_results.csv", index=False)
    print(f"\n  Results → {out_dir / 'multi_source_results.csv'}")

    # Delta summary
    base_results = [r for r in all_results if r.get("experiment","").startswith("A:")]
    aug_results  = [r for r in all_results if r.get("experiment","").startswith("B:")]
    if base_results and aug_results:
        best_base = max(base_results, key=lambda x: x["accuracy"])
        best_aug  = max(aug_results,  key=lambda x: x["accuracy"])
        gain = best_aug["accuracy"] - best_base["accuracy"]
        print(f"\n  DELTA: Auxiliary scores improved accuracy by {gain:+.4f} "
              f"({gain*100:+.2f} pp)")
        print(f"  Baseline best : {best_base['model']:<20} acc={best_base['accuracy']:.4f}")
        print(f"  Augmented best: {best_aug['model']:<20} acc={best_aug['accuracy']:.4f}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
