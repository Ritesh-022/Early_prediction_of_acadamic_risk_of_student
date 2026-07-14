#!/usr/bin/env python3
"""
Branch diagnosis on V2 table — compare branch accuracy V1 vs V2.
Shows whether new temporal features improve the Fail/Withdrawn boundary.
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

SEED = 42
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

BINARY_MAP = {"Pass":"Success","Distinction":"Success","Fail":"AtRisk","Withdrawn":"AtRisk"}
DROP_ALWAYS = {
    "final_result","id_student","code_module","code_presentation",
    "date_unregistration","date_unreg","date_unregistered","weighted_score",
    "first_ts","last_ts","active_weeks","clicks_per_active_week",
    "assessments_per_week","activity_count","days_active","avg_clicks_per_day",
    "week_click_sum_1_4","registration_delay_category",
    "last_assessment_day","first_assessment_day","id_assessment","id_site",
}

def build_pre(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(include=["object","category"]).columns.tolist()
    parts = []
    if num: parts.append(("n", SimpleImputer(strategy="median"), num))
    if cat: parts.append(("c", Pipeline([
                ("i", SimpleImputer(strategy="most_frequent")),
                ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat))
    return ColumnTransformer(parts, remainder="drop")

def score_branches(path, label):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    fr = df["final_result"]
    X  = df.drop(columns=[c for c in DROP_ALWAYS | {"final_result"} if c in df.columns])
    X  = X.dropna(axis=1, how="all")

    results = {}
    # Branch 1 — binary
    y1  = LabelEncoder().fit_transform(fr.map(BINARY_MAP))
    pre = build_pre(X); Xt = pre.fit_transform(X)
    clf = LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05,
                         random_state=SEED, verbosity=-1, n_jobs=1)
    s = cross_val_score(clf, Xt, y1, cv=cv, scoring="accuracy", n_jobs=1)
    results["Model1 AtRisk/Success"] = (s.mean(), s.std())

    # Branch 2A — Fail vs Withdrawn
    mask = fr.isin(["Fail","Withdrawn"])
    Xar  = X.loc[mask]; yar = LabelEncoder().fit_transform(fr[mask])
    pre  = build_pre(Xar); Xt = pre.fit_transform(Xar)
    s = cross_val_score(clf, Xt, yar, cv=cv, scoring="accuracy", n_jobs=1)
    results["Model2A Fail/Withdrawn"] = (s.mean(), s.std())

    # Branch 2B — Pass vs Distinction
    mask = fr.isin(["Pass","Distinction"])
    Xs   = X.loc[mask]; ys = LabelEncoder().fit_transform(fr[mask])
    pre  = build_pre(Xs); Xt = pre.fit_transform(Xs)
    s = cross_val_score(clf, Xt, ys, cv=cv, scoring="accuracy", n_jobs=1)
    results["Model2B Pass/Distinction"] = (s.mean(), s.std())

    print(f"\n  === {label} ===")
    for name, (mean, std) in results.items():
        bar = "✓" if mean >= 0.82 else ("~" if mean >= 0.77 else "✗")
        print(f"  {bar} {name:<30}: {mean:.4f} ± {std:.4f}")

    # theoretical ceiling
    dist   = fr.value_counts(normalize=True)
    p_ar   = dist.get("Fail",0) + dist.get("Withdrawn",0)
    p_s    = dist.get("Pass",0) + dist.get("Distinction",0)
    m1     = results["Model1 AtRisk/Success"][0]
    m2a    = results["Model2A Fail/Withdrawn"][0]
    m2b    = results["Model2B Pass/Distinction"][0]
    ceil   = p_ar * m1 * m2a + p_s * m1 * m2b
    print(f"\n  Theoretical 4-class ceiling: {p_ar:.2f}×{m1:.4f}×{m2a:.4f} "
          f"+ {p_s:.2f}×{m1:.4f}×{m2b:.4f} = {ceil:.4f}")
    return results, ceil

print("="*60)
print("  BRANCH ACCURACY DIAGNOSIS — V1 vs V2")
print("="*60)

r1, c1 = score_branches("oulad_ml_table.csv",    "V1 (original ML table)")
r2, c2 = score_branches("oulad_ml_table_v2.csv", "V2 (rich temporal features)")

print("\n" + "="*60)
print("  DELTA (V2 - V1)")
print("="*60)
for name in r1:
    d = r2[name][0] - r1[name][0]
    sign = f"+{d:.4f}" if d >= 0 else f"{d:.4f}"
    print(f"  {name:<30}: {sign}")
print(f"\n  Theoretical ceiling V1 : {c1:.4f}")
print(f"  Theoretical ceiling V2 : {c2:.4f}")
print(f"  Ceiling gain           : {c2-c1:+.4f}")
print()
if r2["Model2A Fail/Withdrawn"][0] - r1["Model2A Fail/Withdrawn"][0] > 0.02:
    print("  ✓ Meaningful gain on Fail/Withdrawn — proceed with hierarchical V2")
else:
    print("  ! Small gain on Fail/Withdrawn — features help but boundary is tight")
    print("    Recommend: run full hierarchical V2 to confirm final accuracy")
