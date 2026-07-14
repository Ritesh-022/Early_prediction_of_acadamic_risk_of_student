#!/usr/bin/env python3
"""
extended_models_pipeline.py
============================
Adds five new model families to the OULAD binary and 4-class experiments:

  DNN   – Multi-Layer Perceptron (PyTorch, GPU-optional; falls back to sklearn MLP)
  MLP   – sklearn MLPClassifier  (always available, no GPU required)
  ET    – Extra Trees            (sklearn ExtraTreesClassifier)
  BDT   – Balanced Bagging over a Decision Tree (sklearn)
  DT    – Decision Tree          (sklearn DecisionTreeClassifier)

Outputs (written to results/extended/):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  predictions_oulad_binary_<model>.csv                               │
  │  predictions_oulad_4class_<model>.csv                               │
  │  shap_oulad_binary_<model>.csv   (where applicable)                 │
  │  shap_oulad_4class_<model>.csv   (where applicable)                 │
  │  extended_results.csv            (all rows, same schema as          │
  │                                   high_accuracy_results.csv)        │
  └─────────────────────────────────────────────────────────────────────┘

After this script finishes, run  update_main_results.py  to merge the new
rows into the comparison figures without touching the existing 35-figure set.

Usage
-----
    python extended_models_pipeline.py
    python extended_models_pipeline.py --models dnn,mlp,et,bdt,dt
    python extended_models_pipeline.py --mode binary
    python extended_models_pipeline.py --mode 4class
    python extended_models_pipeline.py --mode both    (default)
    python extended_models_pipeline.py --cv-folds 3  --no-shap
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

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, LabelEncoder,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef,
    classification_report, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import joblib

# ── optional: PyTorch DNN ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ── optional: SHAP ────────────────────────────────────────────────────────────
try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _shap = None
    _HAS_SHAP = False

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT    = Path(__file__).resolve().parent
SEED    = 42
logger  = logging.getLogger(__name__)

# ── reuse the same OULAD drop list and binary map as high_accuracy_pipeline ───
OULAD_DROP = {
    "date_unregistration", "date_unreg", "date_unregistered", "weighted_score",
    "active_weeks", "clicks_per_active_week", "assessments_per_week",
    "activity_count", "days_active", "avg_clicks_per_day",
    "registration_delay_category", "id_student", "id_assessment", "id_site",
    "first_ts", "last_ts", "last_assessment_day", "first_assessment_day",
    "code_module", "code_presentation",
}

BINARY_MAP = {
    "Pass": "Success", "Distinction": "Success",
    "Fail": "AtRisk",  "Withdrawn":   "AtRisk",
}

FOUR_CLASS_ORDER = ["Distinction", "Fail", "Pass", "Withdrawn"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_oulad(binary: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Load OULAD ML table, apply feature engineering, return X, y."""
    for name in ["oulad_ml_table_v2.csv", "oulad_ml_table.csv"]:
        p = ROOT / name
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            logger.info("  Loaded %s  shape=%s", name, df.shape)
            break
    else:
        raise FileNotFoundError("No OULAD ML table found (oulad_ml_table_v2.csv / oulad_ml_table.csv)")

    target_col = "final_result"
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found in OULAD table.")

    # Apply same drop list as high_accuracy_pipeline
    df = df.drop(columns=[c for c in OULAD_DROP if c in df.columns])

    # Binary target mapping
    if binary:
        df[target_col] = df[target_col].map(BINARY_MAP).fillna("AtRisk")

    df = df.dropna(subset=[target_col])
    y = df[target_col].astype(str)
    X = df.drop(columns=[target_col], errors="ignore")
    X = X.dropna(axis=1, how="all")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Standard scaler + OHE preprocessor. All models except trees use scaling."""
    parts = []
    if num_cols:
        parts.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False, max_categories=30)),
        ]), cat_cols))
    return ColumnTransformer(parts, remainder="drop")


def build_tree_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Tree models don't need scaling — just impute + ordinal-encode cats."""
    parts = []
    if num_cols:
        parts.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        from sklearn.preprocessing import OrdinalEncoder
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols))
    return ColumnTransformer(parts, remainder="drop")


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH DNN
# ══════════════════════════════════════════════════════════════════════════════

class _TorchDNN(nn.Module):
    """3-hidden-layer MLP with BatchNorm, Dropout, GELU activations."""

    def __init__(self, in_dim: int, n_classes: int,
                 hidden: Tuple[int, ...] = (512, 256, 128),
                 dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DNNClassifier:
    """
    sklearn-compatible wrapper around _TorchDNN.
    Falls back to sklearn MLP if PyTorch is not installed.
    """

    def __init__(self, n_classes: int = 2, epochs: int = 50,
                 batch_size: int = 512, lr: float = 1e-3,
                 hidden: Tuple = (512, 256, 128), dropout: float = 0.3,
                 seed: int = SEED):
        self.n_classes  = n_classes
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.hidden     = hidden
        self.dropout    = dropout
        self.seed       = seed
        self._model     = None
        self._device    = None
        self.classes_   = None

    def fit(self, X, y):
        if not _HAS_TORCH:
            # Fallback to sklearn MLP
            self._sklearn_fallback = MLPClassifier(
                hidden_layer_sizes=(512, 256, 128), activation="relu",
                max_iter=300, random_state=self.seed, early_stopping=True,
            )
            self._sklearn_fallback.fit(X, y)
            self.classes_ = self._sklearn_fallback.classes_
            return self

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        n_classes = len(np.unique(y))
        self.classes_ = np.unique(y)

        X_t = torch.tensor(X.astype(np.float32), device=self._device)
        y_t = torch.tensor(y.astype(np.int64),   device=self._device)

        ds     = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self._model = _TorchDNN(
            X_t.shape[1], n_classes, self.hidden, self.dropout
        ).to(self._device)

        # Class-weighted cross-entropy
        counts  = np.bincount(y)
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * len(counts)
        w_t     = torch.tensor(weights, dtype=torch.float32, device=self._device)

        criterion  = nn.CrossEntropyLoss(weight=w_t)
        optimizer  = torch.optim.AdamW(self._model.parameters(), lr=self.lr,
                                        weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs)

        self._model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        return self

    def predict_proba(self, X):
        if not _HAS_TORCH or hasattr(self, "_sklearn_fallback"):
            return self._sklearn_fallback.predict_proba(X)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(
                torch.tensor(X.astype(np.float32), device=self._device)
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def get_extended_models(n_classes: int, seed: int = SEED) -> Dict:
    """
    Returns dict of model_key -> (classifier, needs_scaling).
    needs_scaling=True  → use build_preprocessor  (StandardScaler)
    needs_scaling=False → use build_tree_preprocessor (no scaling)
    """
    return {
        "dnn": (
            DNNClassifier(n_classes=n_classes, epochs=60, seed=seed),
            True,   # DNN needs scaled inputs
        ),
        "mlp": (
            MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate="adaptive",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
            ),
            True,   # MLP needs scaling
        ),
        "et": (
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
            False,  # trees don't need scaling
        ),
        "bdt": (
            BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=8,
                    class_weight="balanced",
                    random_state=seed,
                ),
                n_estimators=200,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                random_state=seed,
                n_jobs=1,
            ),
            False,  # bagged DT — no scaling
        ),
        "dt": (
            DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=seed,
            ),
            False,  # DT — no scaling
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION  (same schema as high_accuracy_pipeline full_eval)
# ══════════════════════════════════════════════════════════════════════════════

def full_eval(y_true, y_pred, y_prob, le: LabelEncoder,
              model_name: str, ds_name: str) -> Dict:
    r: Dict = {
        "dataset":            ds_name,
        "model":              model_name,
        "n_samples":          len(y_true),
        "n_classes":          len(np.unique(y_true)),
        "accuracy":           accuracy_score(y_true, y_pred),
        "f1_macro":           f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy":  balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa":        cohen_kappa_score(y_true, y_pred),
        "mcc":                matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            classes = np.arange(len(le.classes_))
            if len(le.classes_) == 2:
                r["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                y_bin = label_binarize(y_true, classes=classes)
                r["roc_auc"] = roc_auc_score(
                    y_bin, y_prob, average="macro", multi_class="ovr"
                )
        except Exception:
            r["roc_auc"] = float("nan")
    else:
        r["roc_auc"] = float("nan")
    return r


# ══════════════════════════════════════════════════════════════════════════════
# SHAP (tree-based only via KernelExplainer fallback for others)
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap(clf, X_sample: np.ndarray,
                 feature_names: List[str], model_key: str) -> Optional[List]:
    """Returns list of (feature, mean_abs_shap) sorted descending, or None."""
    if not _HAS_SHAP:
        return None
    try:
        if model_key in ("et",):
            exp  = _shap.TreeExplainer(clf)
            sv   = exp.shap_values(X_sample)
        else:
            # KernelExplainer works for any model but is slow — use small sample
            n    = min(100, len(X_sample))
            bg   = _shap.sample(X_sample, min(50, n))
            exp  = _shap.KernelExplainer(
                lambda x: clf.predict_proba(x), bg
            )
            sv   = exp.shap_values(X_sample[:n], nsamples=50)

        if isinstance(sv, list):
            vals = np.mean([np.abs(v) for v in sv], axis=0).mean(axis=0)
        elif sv.ndim == 3:
            vals = np.abs(sv).mean(axis=(0, 2))
        else:
            vals = np.abs(sv).mean(axis=0)

        n = min(len(feature_names), len(vals))
        order = np.argsort(vals[:n])[::-1][:25]
        return [(feature_names[i], float(vals[i])) for i in order]
    except Exception as e:
        logger.debug("SHAP failed for %s: %s", model_key, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CORE EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_extended_experiment(
    binary: bool,
    model_keys: List[str],
    cv_folds: int = 5,
    do_shap: bool = True,
    output_dir: Optional[Path] = None,
    seed: int = SEED,
) -> List[Dict]:
    """
    Trains each model in model_keys on the OULAD table and returns a list
    of result dicts in the same schema as high_accuracy_results.csv.
    """
    mode  = "binary" if binary else "4class"
    label = f"oulad_{mode}"
    logger.info("\n%s\n=== %s ===\n%s", "=" * 65, label, "=" * 65)

    X, y_raw = load_oulad(binary=binary)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Single shared split — identical for all models (same as high_accuracy_pipeline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, pd.Series(y_enc, index=X.index),
        test_size=0.2, random_state=seed, stratify=y_enc,
    )
    logger.info("  Split: train=%d  test=%d  classes=%s",
                len(X_train), len(X_test), list(le.classes_))

    min_class = int(y_train.value_counts().min())
    folds = max(2, min(cv_folds, min_class))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    all_models = get_extended_models(n_classes=n_classes, seed=seed)
    selected   = {k: v for k, v in all_models.items() if k in model_keys}

    results: List[Dict] = []

    for mkey, (clf, needs_scaling) in selected.items():
        logger.info("  [%s] Training %s ...", mkey, type(clf).__name__)

        pre = (build_preprocessor(num_cols, cat_cols)
               if needs_scaling
               else build_tree_preprocessor(num_cols, cat_cols))

        X_train_t = pre.fit_transform(X_train)
        X_test_t  = pre.transform(X_test)

        # ── Cross-validation (training split only) ───────────────────────────
        try:
            cv_scores = cross_val_score(
                clf, X_train_t, y_train.values,
                cv=cv, scoring="f1_macro", n_jobs=1,
            )
            logger.info("  [%s] CV f1_macro: %.4f ± %.4f",
                        mkey, cv_scores.mean(), cv_scores.std())
        except Exception as exc:
            logger.warning("  [%s] CV failed: %s", mkey, exc)

        # ── Final fit on full training split ─────────────────────────────────
        clf.fit(X_train_t, y_train.values)

        y_pred = clf.predict(X_test_t)

        # Ensure y_pred are integer-encoded if classifier returns int-like
        try:
            y_pred_enc = y_pred.astype(int)
        except (ValueError, TypeError):
            # DNN wrapper returns original class ints already
            y_pred_enc = y_pred

        try:
            y_prob = clf.predict_proba(X_test_t)
        except Exception:
            y_prob = None

        # Decode back to string labels for saving CSVs
        y_test_str = le.inverse_transform(y_test.values.astype(int))
        y_pred_str = le.inverse_transform(y_pred_enc)

        r = full_eval(y_test.values, y_pred_enc, y_prob, le, mkey, label)
        results.append(r)

        logger.info(
            "  [%s] acc=%.4f  f1_mac=%.4f  bacc=%.4f  kappa=%.4f  roc=%.4f",
            mkey, r["accuracy"], r["f1_macro"], r["balanced_accuracy"],
            r["cohen_kappa"], r.get("roc_auc", float("nan")),
        )
        logger.info(
            "\n%s",
            classification_report(
                y_test.values, y_pred_enc,
                target_names=le.classes_, zero_division=0,
            ),
        )

        # ── Save prediction CSV ───────────────────────────────────────────────
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            pred_df = pd.DataFrame({
                "y_true": y_test_str,
                "y_pred": y_pred_str,
            })
            if y_prob is not None:
                for ci, cname in enumerate(le.classes_):
                    pred_df[f"prob_{cname}"] = y_prob[:, ci]
                pred_df["confidence"] = y_prob.max(axis=1)
            pred_path = output_dir / f"predictions_{label}_{mkey}.csv"
            pred_df.to_csv(pred_path, index=False)
            logger.info("  [%s] Predictions saved → %s", mkey, pred_path.name)

        # ── SHAP ─────────────────────────────────────────────────────────────
        if do_shap:
            try:
                feat_names = [
                    n.replace("num__", "").replace("cat__ohe__", "")
                     .replace("cat__ord__", "").replace("cat__", "")
                     .replace("remainder__", "")
                    for n in pre.get_feature_names_out()
                ]
            except Exception:
                feat_names = [f"f{i}" for i in range(X_test_t.shape[1])]

            # Use a smaller sample for SHAP to keep runtime reasonable
            n_shap   = min(300, len(X_test_t))
            X_s      = X_test_t[:n_shap]
            shap_imp = compute_shap(clf, X_s, feat_names, mkey)

            if shap_imp and output_dir is not None:
                shap_path = output_dir / f"shap_{label}_{mkey}.csv"
                pd.DataFrame(shap_imp,
                             columns=["feature", "shap_importance"]
                             ).to_csv(shap_path, index=False)
                logger.info("  [%s] SHAP saved → %s", mkey, shap_path.name)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: List[Dict]) -> None:
    if not results:
        return
    df = pd.DataFrame(results)
    print("\n" + "=" * 100)
    print(f"{'DATASET':<25} {'MODEL':<14} {'ACC':>7} {'F1-MAC':>7} "
          f"{'BACC':>7} {'KAPPA':>7} {'MCC':>7} {'ROC-AUC':>8}")
    print("=" * 100)
    for _, row in df.sort_values(
        ["dataset", "accuracy"], ascending=[True, False]
    ).iterrows():
        print(
            f"{row['dataset']:<25} {row['model']:<14} "
            f"{row['accuracy']:>7.4f} {row['f1_macro']:>7.4f} "
            f"{row['balanced_accuracy']:>7.4f} {row['cohen_kappa']:>7.4f} "
            f"{row['mcc']:>7.4f} {row.get('roc_auc', float('nan')):>8.4f}"
        )
    print("=" * 100)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

ALL_MODEL_KEYS = ["dnn", "mlp", "et", "bdt", "dt"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extended models (DNN / MLP / ET / BDT / DT) for OULAD"
    )
    parser.add_argument(
        "--models", default=",".join(ALL_MODEL_KEYS),
        help="Comma-separated list of models to run. "
             "Choices: dnn,mlp,et,bdt,dt  (default: all five)",
    )
    parser.add_argument(
        "--mode", default="both",
        choices=["binary", "4class", "both"],
        help="Which classification mode to run (default: both)",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--no-shap", dest="shap", action="store_false", default=True,
        help="Skip SHAP computation (faster)",
    )
    parser.add_argument(
        "--output-dir", default="results/extended",
        help="Output directory for CSVs (default: results/extended)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_keys = [m.strip().lower() for m in args.models.split(",")
                  if m.strip().lower() in ALL_MODEL_KEYS]
    if not model_keys:
        print(f"[ERROR] No valid model keys. Choose from: {ALL_MODEL_KEYS}")
        return

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  EXTENDED MODELS PIPELINE")
    print(f"  Models : {model_keys}")
    print(f"  Mode   : {args.mode}")
    print(f"  Output : {out_dir}")
    print("=" * 65)

    all_results: List[Dict] = []

    modes = []
    if args.mode in ("binary", "both"):
        modes.append(True)
    if args.mode in ("4class", "both"):
        modes.append(False)

    for binary in modes:
        mode_str = "binary" if binary else "4class"
        print(f"\n{'─'*65}")
        print(f"  Running {mode_str.upper()} classification ...")
        print(f"{'─'*65}")
        rows = run_extended_experiment(
            binary=binary,
            model_keys=model_keys,
            cv_folds=args.cv_folds,
            do_shap=args.shap,
            output_dir=out_dir,
            seed=args.seed,
        )
        all_results.extend(rows)

    # ── Print final table ─────────────────────────────────────────────────────
    print_summary(all_results)

    # ── Save combined results CSV — merge with existing rows ─────────────────
    out_csv = out_dir / "extended_results.csv"
    new_df = pd.DataFrame(all_results)
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        # Drop rows for dataset+model combos being re-run, then append fresh rows
        key_cols = ["dataset", "model"]
        if all(c in existing.columns for c in key_cols):
            mask = existing.set_index(key_cols).index.isin(
                new_df.set_index(key_cols).index)
            existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(out_csv, index=False)
    print(f"\n  Results saved → {out_csv}")
    print(f"  Run  python update_main_results.py  to merge into comparison figures.\n")


if __name__ == "__main__":
    main()
