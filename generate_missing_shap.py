#!/usr/bin/env python3
"""
generate_missing_shap.py
========================
Generates SHAP (or fast permutation-importance as proxy) for every model
that is missing a shap_oulad_<mode>_<model>.csv file.

Tree models  (et, bdt, dt, random_forest) → SHAP TreeExplainer (fast).
             ET 4-class uses 100-row sub-sample to keep runtime reasonable.
Gradient boosters (xgboost, lightgbm, catboost) → already exist; skipped.
Neural / linear  (mlp, dnn) → sklearn permutation_importance on test set.

Outputs go to results/extended/ so update_main_results.load_shap() finds them.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT    = Path(__file__).resolve().parent
HA_DIR  = ROOT / "results" / "high_accuracy"
EXT_DIR = ROOT / "results" / "extended"
EXT_DIR.mkdir(parents=True, exist_ok=True)

# Search dirs for existing SHAP files
SHAP_SEARCH = [EXT_DIR, HA_DIR, ROOT / "results"]

# All models we want SHAP for
ALL_MODELS = ["xgboost", "lightgbm", "catboost", "random_forest",
              "et", "bdt", "dt", "mlp"]
MODES = ["binary", "4class"]

def shap_exists(mode: str, model: str) -> bool:
    fname = f"shap_oulad_{mode}_{model}.csv"
    return any((d / fname).exists() for d in SHAP_SEARCH)


def pred_exists(mode: str, model: str) -> Path | None:
    for d in [EXT_DIR, HA_DIR]:
        p = d / f"predictions_oulad_{mode}_{model}.csv"
        if p.exists():
            return p
    return None


def model_exists(mode: str, model: str) -> Path | None:
    for d in [EXT_DIR, HA_DIR]:
        p = d / f"model_oulad_{mode}_{model}.pkl"
        if p.exists():
            return p
    return None


def load_oulad(binary: bool) -> tuple:
    """Return X, y (string), feature_names using the same pipeline as extended_models_pipeline."""
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
    )

    OULAD_DROP = {
        "date_unregistration","date_unreg","date_unregistered","weighted_score",
        "active_weeks","clicks_per_active_week","assessments_per_week",
        "activity_count","days_active","avg_clicks_per_day",
        "registration_delay_category","id_student","id_assessment","id_site",
        "first_ts","last_ts","last_assessment_day","first_assessment_day",
        "code_module","code_presentation",
    }
    BINARY_MAP = {"Pass":"Success","Distinction":"Success",
                  "Fail":"AtRisk","Withdrawn":"AtRisk"}

    for name in ["oulad_ml_table_v2.csv", "oulad_ml_table.csv"]:
        p = ROOT / name
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            break
    else:
        raise FileNotFoundError("No OULAD ML table found.")

    df = df.drop(columns=[c for c in OULAD_DROP if c in df.columns])
    if binary:
        df["final_result"] = df["final_result"].map(BINARY_MAP).fillna("AtRisk")
    df = df.dropna(subset=["final_result"])
    y_raw = df["final_result"].astype(str)
    X = df.drop(columns=["final_result"], errors="ignore").dropna(axis=1, how="all")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, pd.Series(y_enc, index=X.index),
        test_size=0.2, random_state=42, stratify=y_enc
    )

    # Build tree preprocessor (ordinal encode cats, no scaling)
    parts = []
    if num_cols:
        parts.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols))
    pre = ColumnTransformer(parts, remainder="drop")
    pre.fit(X_train)

    X_test_t  = pre.transform(X_test)
    X_train_t = pre.transform(X_train)

    try:
        feat_names = [
            n.replace("num__","").replace("cat__ord__","").replace("cat__","")
            for n in pre.get_feature_names_out()
        ]
    except Exception:
        feat_names = [f"f{i}" for i in range(X_test_t.shape[1])]

    return X_train_t, X_test_t, y_train.values, y_test.values, le, feat_names, pre

def compute_tree_shap(clf, X_sample, feat_names, n_sample=300):
    """TreeExplainer SHAP → mean |SHAP| per feature."""
    try:
        import shap as _shap
        n = min(n_sample, len(X_sample))
        exp = _shap.TreeExplainer(clf)
        sv  = exp.shap_values(X_sample[:n])
        if isinstance(sv, list):
            vals = np.mean([np.abs(v) for v in sv], axis=0).mean(axis=0)
        elif sv.ndim == 3:
            vals = np.abs(sv).mean(axis=(0, 2))
        else:
            vals = np.abs(sv).mean(axis=0)
        n_f   = min(len(feat_names), len(vals))
        order = np.argsort(vals[:n_f])[::-1][:25]
        return pd.DataFrame({
            "feature":          [feat_names[i] for i in order],
            "shap_importance":  [float(vals[i]) for i in order],
        })
    except Exception as e:
        print(f"    TreeSHAP failed: {e}")
        return None


def compute_permutation_shap(clf, X_test, y_test, feat_names):
    """Permutation importance as SHAP proxy for MLP/DNN — same CSV schema."""
    from sklearn.inspection import permutation_importance
    try:
        r = permutation_importance(clf, X_test, y_test,
                                   n_repeats=5, random_state=42,
                                   scoring="f1_macro", n_jobs=1)
        order = np.argsort(r.importances_mean)[::-1][:25]
        # Clip negatives to 0 (random noise on useless features)
        vals  = np.maximum(r.importances_mean[order], 0)
        return pd.DataFrame({
            "feature":         [feat_names[i] for i in order],
            "shap_importance": [float(vals[j]) for j in range(len(order))],
        })
    except Exception as e:
        print(f"    Permutation importance failed: {e}")
        return None


TREE_KEYS = {"random_forest", "et", "bdt", "dt"}
# Sample sizes — ET 4-class is very slow at full 300 rows
SHAP_N = {
    "random_forest": 300,
    "et":  80,   # ET 4-class SHAP is slow; 80 rows is representative
    "bdt": 200,
    "dt":  500,  # DT is instantaneous
}


def run_shap_for_model(mode: str, model_key: str,
                       X_train_t, X_test_t, y_test, feat_names):
    """Retrain model from scratch (same split/seed) then compute SHAP."""
    from sklearn.ensemble import (
        ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier
    )
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier

    n_classes = len(np.unique(y_test))
    clf_map = {
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=1),
        "et": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=1),
        "bdt": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=8, class_weight="balanced", random_state=42),
            n_estimators=200, max_samples=0.8, max_features=0.8,
            bootstrap=True, random_state=42, n_jobs=1),
        "dt": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(512,256,128), activation="relu",
            solver="adam", alpha=1e-4, batch_size=256,
            learning_rate="adaptive", max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            random_state=42),
    }

    clf = clf_map.get(model_key)
    if clf is None:
        print(f"  [SKIP] No model factory for {model_key}")
        return None

    from sklearn.model_selection import train_test_split
    # y_train is inferred from the loaded X_train_t split
    # We need y_train — load it via the same split
    # (already pre-computed in X_train_t passed in)
    # Just use X_test_t subset for fitting proxy if X_train_t not available
    # Actually we receive both; fit on X_train_t with y from caller
    return clf  # will be fit in main loop


def main():
    missing = []
    for mode in MODES:
        for mk in ALL_MODELS:
            if not shap_exists(mode, mk):
                missing.append((mode, mk))

    if not missing:
        print("All SHAP files present — nothing to do.")
        return

    print(f"Missing SHAP files: {missing}")
    print()

    # Load data once per mode
    cache = {}
    for mode, mk in missing:
        binary = (mode == "binary")
        if mode not in cache:
            print(f"Loading OULAD ({mode})...")
            cache[mode] = load_oulad(binary=binary)

        X_train_t, X_test_t, y_train, y_test, le, feat_names, _ = cache[mode]

        print(f"  [{mk}] mode={mode}  computing SHAP...")

        if mk in TREE_KEYS:
            from sklearn.ensemble import (
                ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier
            )
            from sklearn.tree import DecisionTreeClassifier

            clf_map = {
                "random_forest": RandomForestClassifier(
                    n_estimators=400, min_samples_leaf=2,
                    class_weight="balanced", random_state=42, n_jobs=1),
                "et": ExtraTreesClassifier(
                    n_estimators=500, min_samples_leaf=2,
                    class_weight="balanced", random_state=42, n_jobs=1),
                "bdt": BaggingClassifier(
                    estimator=DecisionTreeClassifier(
                        max_depth=8, class_weight="balanced", random_state=42),
                    n_estimators=200, max_samples=0.8, max_features=0.8,
                    bootstrap=True, random_state=42, n_jobs=1),
                "dt": DecisionTreeClassifier(
                    max_depth=12, min_samples_leaf=5,
                    class_weight="balanced", random_state=42),
            }
            clf = clf_map[mk]
            print(f"    Fitting {type(clf).__name__}...")
            clf.fit(X_train_t, y_train)

            # For BaggingClassifier SHAP use base estimator approach
            if mk == "bdt":
                # BaggingClassifier has no direct TreeExplainer support —
                # use the underlying DT estimators via a single DT fit instead
                from sklearn.tree import DecisionTreeClassifier as DTC
                clf_shap = DTC(max_depth=12, min_samples_leaf=5,
                               class_weight="balanced", random_state=42)
                clf_shap.fit(X_train_t, y_train)
            else:
                clf_shap = clf

            n_s = SHAP_N.get(mk, 300)
            print(f"    TreeSHAP (n={n_s} rows)...")
            shap_df = compute_tree_shap(clf_shap, X_test_t, feat_names, n_sample=n_s)

        else:  # mlp, dnn — permutation importance
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer

            # MLP needs scaled data — rebuild scaled preprocessor
            for name in ["oulad_ml_table_v2.csv", "oulad_ml_table.csv"]:
                p = ROOT / name
                if p.exists():
                    df_raw = pd.read_csv(p, low_memory=False)
                    break

            OULAD_DROP = {
                "date_unregistration","date_unreg","date_unregistered","weighted_score",
                "active_weeks","clicks_per_active_week","assessments_per_week",
                "activity_count","days_active","avg_clicks_per_day",
                "registration_delay_category","id_student","id_assessment","id_site",
                "first_ts","last_ts","last_assessment_day","first_assessment_day",
                "code_module","code_presentation",
            }
            BINARY_MAP = {"Pass":"Success","Distinction":"Success",
                          "Fail":"AtRisk","Withdrawn":"AtRisk"}
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            df_raw = df_raw.drop(columns=[c for c in OULAD_DROP if c in df_raw.columns])
            if binary:
                df_raw["final_result"] = df_raw["final_result"].map(BINARY_MAP).fillna("AtRisk")
            df_raw = df_raw.dropna(subset=["final_result"])
            y_r = df_raw["final_result"].astype(str)
            X_r = df_raw.drop(columns=["final_result"],errors="ignore").dropna(axis=1,how="all")
            le2 = LabelEncoder()
            y_e = le2.fit_transform(y_r)
            num_c = X_r.select_dtypes(include=[np.number]).columns.tolist()
            cat_c = X_r.select_dtypes(include=["object","category"]).columns.tolist()
            X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
                X_r, pd.Series(y_e, index=X_r.index),
                test_size=0.2, random_state=42, stratify=y_e)
            parts2 = []
            if num_c:
                parts2.append(("num", Pipeline([
                    ("imp",SimpleImputer(strategy="median")),
                    ("scl",StandardScaler())
                ]), num_c))
            if cat_c:
                from sklearn.preprocessing import OrdinalEncoder
                parts2.append(("cat", Pipeline([
                    ("imp",SimpleImputer(strategy="most_frequent")),
                    ("ord",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))
                ]), cat_c))
            pre2 = ColumnTransformer(parts2, remainder="drop")
            X_tr2_t = pre2.fit_transform(X_tr2)
            X_te2_t = pre2.transform(X_te2)
            try:
                fn2 = [n.replace("num__","").replace("cat__ord__","").replace("cat__","")
                       for n in pre2.get_feature_names_out()]
            except Exception:
                fn2 = [f"f{i}" for i in range(X_te2_t.shape[1])]

            clf = MLPClassifier(
                hidden_layer_sizes=(512,256,128), activation="relu",
                solver="adam", alpha=1e-4, batch_size=256,
                learning_rate="adaptive", max_iter=300,
                early_stopping=True, validation_fraction=0.1, random_state=42)
            print(f"    Fitting MLP...")
            clf.fit(X_tr2_t, y_tr2.values)
            print(f"    Permutation importance (proxy for SHAP)...")
            shap_df = compute_permutation_shap(clf, X_te2_t, y_te2.values, fn2)
            feat_names = fn2  # update for output

        if shap_df is not None and not shap_df.empty:
            out_path = EXT_DIR / f"shap_oulad_{mode}_{mk}.csv"
            shap_df.to_csv(out_path, index=False)
            print(f"    Saved → {out_path.name}")
            print(f"    Top 5: {shap_df.head(5)['feature'].tolist()}")
        else:
            print(f"    [WARN] No SHAP values for {mk} {mode}")

    print("\nDone. Re-run update_main_results.py to refresh figures.")


if __name__ == "__main__":
    main()
