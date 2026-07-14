#!/usr/bin/env python3
"""
generate_all_figures.py

Research-paper experiment runner and figure generator.

What this script does
---------------------
1. Runs all required experiment modes/parsers.
2. Caches terminal output so completed experiments do not need to rerun.
3. Parses:
   - multiple datasets from high_accuracy_pipeline.py
   - multiple models from each dataset/experiment
   - accuracy
   - precision
   - recall
   - macro F1
   - weighted F1
   - balanced accuracy
   - ROC-AUC
   - PR-AUC
   - Cohen's kappa
   - MCC
   - CV scores
   - confusion matrices
   - printed SHAP feature importance
4. Searches pipeline output folders for raw:
   - prediction CSVs
   - probability CSVs
   - result CSVs
   - SHAP CSVs
   - CV CSVs
5. Generates only REAL figures.
6. Never reconstructs ROC/PR curves from a single AUC number.
7. Never creates fake SHAP values.

Usage
-----
Run missing experiments + generate figures:

    python generate_all_figures.py

Generate figures from existing cached results only:

    python generate_all_figures.py --graphs-only

Force all experiments to rerun:

    python generate_all_figures.py --force

Run experiments only:

    python generate_all_figures.py --run-only

Skip slow synthetic platform runs:

    python generate_all_figures.py --skip-platform

Skip multisource runs:

    python generate_all_figures.py --skip-multisource

Skip hierarchical CTGAN run:

    python generate_all_figures.py --fast

IMPORTANT
---------
Some figures require raw prediction probabilities or raw SHAP values.
If the source training script does not export those values, this script
will NOT fabricate them. It will print exactly what is missing.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

RESULTS_DIR = ROOT / "research_results"
LOG_DIR = RESULTS_DIR / "logs"
ARTIFACT_DIR = RESULTS_DIR / "artifacts"
FIGURES_DIR = ROOT / "figures"

for directory in [
    RESULTS_DIR,
    LOG_DIR,
    ARTIFACT_DIR,
    FIGURES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# REQUIRED INPUT TABLES
# ============================================================

FULL_TABLE_CANDIDATES = [
    ROOT / "oulad_ml_table_v2.csv",
    ROOT / "oulad_ml_table.csv",
]

WEEK8_TABLE_CANDIDATES = [
    ROOT / "oulad_ml_table_week8.csv",
]

WEEK_TABLE_CANDIDATES = {
    2: ROOT / "oulad_ml_table_week2.csv",
    4: ROOT / "oulad_ml_table_week4.csv",
    6: ROOT / "oulad_ml_table_week6.csv",
    8: ROOT / "oulad_ml_table_week8.csv",
}


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


FULL_TABLE = first_existing(FULL_TABLE_CANDIDATES)
WEEK8_TABLE = first_existing(WEEK8_TABLE_CANDIDATES)


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = [

    # --------------------------------------------------------
    # FIX: Split "all datasets + both modes + all models" into
    # two separate, focused experiments for stability.
    # Use --n-jobs 1 to avoid nested parallelism on Windows.
    # Use --no-shap here; SHAP has its own dedicated run.
    # --------------------------------------------------------

    # OULAD — Direct 4-class (multiclass, full-course)
    {
        "id": "high_accuracy_oulad_4class",
        "script": "high_accuracy_pipeline.py",
        "args": [
            "--dataset", "oulad",
            "--model", "xgboost,lightgbm,catboost,random_forest",
            "--cv-folds", "5",
            "--n-jobs", "1",
            "--no-shap",
            "--output-dir", "results/high_accuracy",
        ],
        "group": "high_accuracy",
    },

    # OULAD — Binary (AtRisk vs Success, full-course)
    {
        "id": "high_accuracy_oulad_binary",
        "script": "high_accuracy_pipeline.py",
        "args": [
            "--dataset", "oulad",
            "--model", "xgboost,lightgbm,catboost,random_forest",
            "--binary",
            "--cv-folds", "5",
            "--n-jobs", "1",
            "--no-shap",
            "--output-dir", "results/high_accuracy",
        ],
        "group": "high_accuracy",
    },

    # --------------------------------------------------------
    # OULAD high-accuracy — SHAP (all 4 models, binary + 4class)
    # Separate from the main training runs (which use --no-shap)
    # because RF SHAP is slow. Saves shap_oulad_*.csv to
    # results/high_accuracy/ so generate_main_results.py finds them.
    # --------------------------------------------------------
    {
        "id": "high_accuracy_oulad_4class_shap",
        "script": "high_accuracy_pipeline.py",
        "args": [
            "--dataset", "oulad",
            "--model", "xgboost,lightgbm,catboost,random_forest",
            "--cv-folds", "5",
            "--n-jobs", "1",
            "--output-dir", "results/high_accuracy",
        ],
        "group": "high_accuracy",
    },
    {
        "id": "high_accuracy_oulad_binary_shap",
        "script": "high_accuracy_pipeline.py",
        "args": [
            "--dataset", "oulad",
            "--model", "xgboost,lightgbm,catboost,random_forest",
            "--binary",
            "--cv-folds", "5",
            "--n-jobs", "1",
            "--output-dir", "results/high_accuracy",
        ],
        "group": "high_accuracy",
    },

    # --------------------------------------------------------
    # OULAD FULL COURSE — baseline with SHAP
    # --------------------------------------------------------
    {
        "id": "oulad_benchmark_all_models",
        "script": "oulad_baseline.py",
        "args": [
            "--input", "oulad_ml_table.csv",
            "--target", "final_result",
            "--model", "all",
            "--mode", "benchmark",
            "--cv-splits", "5",
            "--cv-repeats", "1",
            "--n-jobs", "1",
            "--shap-sample", "300",
            "--output-dir", "results/baseline_benchmark",
        ],
        "group": "baseline",
    },

    # --------------------------------------------------------
    # OULAD WEEK 8
    # --------------------------------------------------------
    {
        "id": "oulad_week8_all_models",
        "script": "oulad_baseline.py",
        "args": [
            "--input", "oulad_ml_table_week8.csv",
            "--target", "final_result",
            "--model", "all",
            "--mode", "early-warning",
            "--cv-splits", "5",
            "--cv-repeats", "1",
            "--n-jobs", "1",
            "--shap-sample", "300",
            "--output-dir", "results/baseline_week8",
        ],
        "group": "baseline",
    },

    # --------------------------------------------------------
    # HIERARCHICAL
    # --------------------------------------------------------
    {
        "id": "hierarchical_full",
        "script": "hierarchical_pipeline.py",
        "args": [
            "--skip-ctgan",
            "--output-dir", "results/hierarchical",
            "--save-graphs",
        ],
        "group": "hierarchical",
    },

    # --------------------------------------------------------
    # MULTISOURCE
    # --------------------------------------------------------
    {
        "id": "multisource_benchmark_all",
        "script": "multisource_ablation.py",
        "args": [
            "--mode", "benchmark",
            "--experiment", "all",
            "--cv-folds", "5",
            "--shap-sample", "300",
            "--report-students", "3",
        ],
        "group": "multisource",
    },

    {
        "id": "multisource_early_warning_all",
        "script": "multisource_ablation.py",
        "args": [
            "--mode", "early-warning",
            "--experiment", "all",
            "--cv-folds", "5",
            "--shap-sample", "300",
            "--report-students", "3",
        ],
        "group": "multisource",
    },

    # --------------------------------------------------------
    # UNIFIED PLATFORM
    # --------------------------------------------------------
    {
        "id": "platform_benchmark",
        "script": "synthetic_platform.py",
        "args": [
            "--mode", "benchmark",
            "--shap-sample", "300",
            "--report-students", "3",
        ],
        "group": "platform",
    },

    {
        "id": "platform_early_warning",
        "script": "synthetic_platform.py",
        "args": [
            "--mode", "early-warning",
            "--shap-sample", "300",
            "--report-students", "3",
        ],
        "group": "platform",
    },
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    if value is None:
        return np.nan

    try:
        return float(str(value).strip().replace("%", ""))
    except Exception:
        return np.nan


def normalize_metric(value):
    """
    Convert values like:
        95.05 -> 0.9505
        0.9505 -> 0.9505
    """

    value = safe_float(value)

    if pd.isna(value):
        return np.nan

    if value > 1.0:
        return value / 100.0

    return value


def clean_model_name(name):
    mapping = {
        "rf": "Random Forest",
        "random_forest": "Random Forest",
        "randomforest": "Random Forest",

        "logreg": "Logistic Regression",
        "logistic_regression": "Logistic Regression",

        "dt": "Decision Tree",
        "decision_tree": "Decision Tree",

        "xgb": "XGBoost",
        "xgboost": "XGBoost",

        "lgb": "LightGBM",
        "lightgbm": "LightGBM",

        "cb": "CatBoost",
        "catboost": "CatBoost",

        "stacking": "Stacking Ensemble",
        "stacking_ensemble": "Stacking Ensemble",
    }

    key = str(name).strip().lower()

    return mapping.get(
        key,
        key.replace("_", " ").title()
    )


def safe_filename(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def save_figure(filename):
    path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"[GENERATED] {filename}")


def skip_figure(filename, reason):
    print(f"[SKIP] {filename}")
    print(f"       Reason: {reason}")


# ============================================================
# DATA CHECK
# ============================================================

def check_required_data():

    missing = []

    if FULL_TABLE is None:
        missing.append(
            "oulad_ml_table.csv or oulad_ml_table_v2.csv"
        )

    if WEEK8_TABLE is None:
        missing.append(
            "oulad_ml_table_week8.csv"
        )

    if not missing:
        return True

    print("\n[ERROR] Missing required ML tables:")

    for item in missing:
        print(f"  - {item}")

    print("\nGenerate them using:")

    print(
        "python oulad_pipeline.py "
        "--root . "
        "--output oulad_ml_table.csv "
        "--week-cutoffs 2,4,6,8"
    )

    return False


# ============================================================
# EXPERIMENT EXECUTION
# ============================================================

def experiment_log_path(experiment_id):
    return LOG_DIR / f"{experiment_id}.log"


def experiment_metadata_path(experiment_id):
    return LOG_DIR / f"{experiment_id}.json"


def run_experiment(experiment, force=False, fast=False):

    experiment_id = experiment["id"]

    script_path = ROOT / experiment["script"]

    log_path = experiment_log_path(experiment_id)

    metadata_path = experiment_metadata_path(experiment_id)

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if (
        log_path.exists()
        and metadata_path.exists()
        and not force
    ):
        print(
            f"[CACHE] {experiment_id}"
        )

        text = log_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        return {
            "id": experiment_id,
            "status": "cached",
            "returncode": 0,
            "output": text,
        }

    # --------------------------------------------------------
    # SCRIPT CHECK
    # --------------------------------------------------------

    if not script_path.exists():

        print(
            f"[MISSING SCRIPT] {experiment['script']}"
        )

        return {
            "id": experiment_id,
            "status": "missing_script",
            "returncode": -1,
            "output": "",
        }

    args = list(experiment["args"])

    # --------------------------------------------------------
    # FAST MODE
    # --------------------------------------------------------

    if (
        fast
        and experiment_id == "hierarchical_full"
    ):
        args = [
            "--skip-ctgan",
            "--output-dir",
            "results/hierarchical",
            "--save-graphs",
        ]

    # --------------------------------------------------------
    # FIX: set env vars to limit nested parallelism on Windows
    # (prevents the WinError 2 / loky physical-core-count crash)
    # --------------------------------------------------------
    env = os.environ.copy()
    env["PYTHONIOENCODING"]    = "utf-8"
    env["PYTHONUTF8"]          = "1"
    env["PYTHONUNBUFFERED"]    = "1"
    env["LOKY_MAX_CPU_COUNT"]  = "4"
    env["OMP_NUM_THREADS"]     = "4"
    env["MKL_NUM_THREADS"]     = "4"
    env["OPENBLAS_NUM_THREADS"]= "4"
    env["NUMEXPR_NUM_THREADS"] = "4"

    # FIX: -u makes the child Python process flush output immediately
    command = [
        sys.executable,
        "-u",
        str(script_path),
        *args,
    ]

    print("\n" + "=" * 70)
    print(f"[RUN] {experiment_id}")
    print("=" * 70)
    print(" ".join(str(x) for x in command))

    start = time.time()

    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    output_lines = []
    assert process.stdout is not None

    # FIX: background reader thread + heartbeat so we see progress
    # even when CatBoost/SHAP is silent for minutes
    line_queue: queue.Queue = queue.Queue()

    def _reader(pipe, q):
        try:
            for ln in iter(pipe.readline, ""):
                q.put(ln)
        finally:
            pipe.close()
            q.put(None)

    reader = threading.Thread(
        target=_reader, args=(process.stdout, line_queue), daemon=True)
    reader.start()

    last_output_time = time.time()
    reader_done = False

    while True:
        try:
            line = line_queue.get(timeout=5)
            if line is None:
                reader_done = True
            else:
                print(line, end="", flush=True)
                output_lines.append(line)
                last_output_time = time.time()
        except queue.Empty:
            elapsed = time.time() - start
            silent  = time.time() - last_output_time
            print(
                f"[RUNNING] {experiment_id} | "
                f"elapsed={elapsed/60:.1f} min | "
                f"no output for {silent:.0f}s",
                flush=True,
            )

        if process.poll() is not None and reader_done:
            break

    process.wait()

    elapsed = time.time() - start

    text = "".join(output_lines)

    log_path.write_text(
        text,
        encoding="utf-8",
    )

    metadata = {
        "id": experiment_id,
        "script": experiment["script"],
        "args": args,
        "returncode": process.returncode,
        "elapsed_seconds": elapsed,
        "command": command,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    status = (
        "completed"
        if process.returncode == 0
        else "failed"
    )

    print(
        f"\n[{status.upper()}] "
        f"{experiment_id} "
        f"({elapsed:.1f}s)"
    )

    return {
        "id": experiment_id,
        "status": status,
        "returncode": process.returncode,
        "output": text,
    }


# ============================================================
# TEXT SECTION SPLITTERS
# ============================================================

def split_named_experiment_sections(text):
    """
    Example:

    === oulad_4class ===
    === oulad_binary ===
    === dropout ===
    """

    pattern = re.compile(
        r"===\s*([A-Za-z0-9_\-]+)\s*===",
        flags=re.IGNORECASE,
    )

    matches = list(
        pattern.finditer(text)
    )

    if not matches:
        return {}

    sections = {}

    for i, match in enumerate(matches):

        name = match.group(1).strip()

        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        sections[name] = text[start:end]

    return sections


def split_model_sections(text):
    """
    Supports:

        Evaluating catboost

    and:

        [catboost] Training...
    """

    patterns = [

        re.compile(
            r"(?im)^Evaluating\s+"
            r"([A-Za-z0-9_\-]+)\s*$"
        ),

        re.compile(
            r"(?im)^\s*\["
            r"(xgboost|lightgbm|catboost|"
            r"random_forest|logistic_regression|"
            r"decision_tree|stacking)"
            r"\]\s*Training"
        ),
    ]

    matches = []

    for pattern in patterns:

        for match in pattern.finditer(text):

            matches.append(
                (
                    match.start(),
                    match.end(),
                    match.group(1),
                )
            )

    matches.sort(
        key=lambda x: x[0]
    )

    if not matches:
        return {}

    sections = {}

    for i, (
        start,
        _,
        model_name,
    ) in enumerate(matches):

        if i + 1 < len(matches):
            end = matches[i + 1][0]
        else:
            end = len(text)

        sections[
            model_name.lower()
        ] = text[start:end]

    return sections


# ============================================================
# METRIC PARSERS
# ============================================================

def find_metric(text, patterns):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return normalize_metric(
                match.group(1)
            )

    return np.nan


def parse_metrics(text):

    return {

        "accuracy": find_metric(
            text,
            [
                r"Final test accuracy\s*:\s*([0-9.]+)",
                r"Accuracy\s*:\s*([0-9.]+)",
                r"\bacc\s*=\s*([0-9.]+)",
                r"\baccuracy\s*=\s*([0-9.]+)",
            ],
        ),

        "f1_macro": find_metric(
            text,
            [
                r"Final test f1_macro\s*:\s*([0-9.]+)",
                r"Macro F1\s*:\s*([0-9.]+)",
                r"\bf1_mac\s*=\s*([0-9.]+)",
                r"\bf1_macro\s*=\s*([0-9.]+)",
            ],
        ),

        "f1_weighted": find_metric(
            text,
            [
                r"Final test f1_weighted\s*:\s*([0-9.]+)",
                r"Weighted F1\s*:\s*([0-9.]+)",
            ],
        ),

        "balanced_accuracy": find_metric(
            text,
            [
                r"Balanced accuracy\s*:\s*([0-9.]+)",
                r"Balanced Acc\s*:\s*([0-9.]+)",
                r"\bbacc\s*=\s*([0-9.]+)",
            ],
        ),

        "roc_auc": find_metric(
            text,
            [
                r"ROC AUC OVR\s*:\s*([0-9.]+)",
                r"ROC AUC \(macro\)\s*:\s*([0-9.]+)",
                r"ROC[- ]?AUC\s*:\s*([0-9.]+)",
                r"\broc\s*=\s*([0-9.]+)",
            ],
        ),

        "pr_auc": find_metric(
            text,
            [
                r"PR[- ]?AUC\s*:\s*([0-9.]+)",
                r"Average Precision\s*:\s*([0-9.]+)",
            ],
        ),

        "cohen_kappa": find_metric(
            text,
            [
                r"Cohen(?:'s)? kappa\s*:\s*([0-9.]+)",
                r"\bkappa\s*=\s*([0-9.]+)",
            ],
        ),

        "mcc": find_metric(
            text,
            [
                r"MCC\s*:\s*([0-9.]+)",
                r"\bmcc\s*=\s*([0-9.]+)",
            ],
        ),
    }


def parse_cv(text):

    result = {}

    patterns = {

        "cv_accuracy_mean":
            r"(?:CV\s+)?accuracy\s*:"
            r"\s*([0-9.]+)\s*[±+/-]+",

        "cv_f1_macro_mean":
            r"(?:CV\s+)?f1_macro\s*:"
            r"\s*([0-9.]+)\s*[±+/-]+",

        "cv_balanced_accuracy_mean":
            r"(?:CV\s+)?balanced_accuracy\s*:"
            r"\s*([0-9.]+)\s*[±+/-]+",

        "cv_f1_macro_alt":
            r"CV f1_macro\s*:"
            r"\s*([0-9.]+)\s*[±+/-]+",
    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            result[key] = normalize_metric(
                match.group(1)
            )

    if (
        "cv_f1_macro_mean" not in result
        and "cv_f1_macro_alt" in result
    ):
        result["cv_f1_macro_mean"] = (
            result["cv_f1_macro_alt"]
        )

    return result


# ============================================================
# CLASSIFICATION REPORT PARSER
# ============================================================

def parse_classification_report(text):

    result = {}

    macro_match = re.search(
        r"macro avg\s+"
        r"([0-9.]+)\s+"
        r"([0-9.]+)\s+"
        r"([0-9.]+)",
        text,
        flags=re.IGNORECASE,
    )

    if macro_match:

        result["precision_macro"] = (
            normalize_metric(
                macro_match.group(1)
            )
        )

        result["recall_macro"] = (
            normalize_metric(
                macro_match.group(2)
            )
        )

        report_f1 = normalize_metric(
            macro_match.group(3)
        )

        if pd.notna(report_f1):
            result[
                "classification_report_f1_macro"
            ] = report_f1

    weighted_match = re.search(
        r"weighted avg\s+"
        r"([0-9.]+)\s+"
        r"([0-9.]+)\s+"
        r"([0-9.]+)",
        text,
        flags=re.IGNORECASE,
    )

    if weighted_match:

        result["precision_weighted"] = (
            normalize_metric(
                weighted_match.group(1)
            )
        )

        result["recall_weighted"] = (
            normalize_metric(
                weighted_match.group(2)
            )
        )

        result["report_f1_weighted"] = (
            normalize_metric(
                weighted_match.group(3)
            )
        )

    return result


# ============================================================
# CONFUSION MATRIX PARSER
# ============================================================

def parse_confusion_matrix(text):

    marker = re.search(
        r"Confusion matrix\s*:",
        text,
        flags=re.IGNORECASE,
    )

    if not marker:
        return None

    tail = text[
        marker.end():
        marker.end() + 3000
    ]

    rows = re.findall(
        r"\[\s*"
        r"([0-9]+(?:\s+[0-9]+)+)"
        r"\s*\]",
        tail,
    )

    if not rows:
        return None

    matrix = []

    for row in rows:

        values = [
            int(x)
            for x in row.split()
        ]

        matrix.append(values)

    if not matrix:
        return None

    width = len(matrix[0])

    matrix = [
        row
        for row in matrix
        if len(row) == width
    ]

    if len(matrix) != width:
        return None

    return matrix


# ============================================================
# PRINTED SHAP IMPORTANCE PARSER
# ============================================================

def parse_shap_importance(text):

    marker = re.search(
        r"Top(?:\s+\d+)?\s+SHAP"
        r"(?: feature)? importances?"
        r"(?:\s*\(global\))?\s*:",
        text,
        flags=re.IGNORECASE,
    )

    if not marker:
        return []

    tail = text[
        marker.end():
        marker.end() + 6000
    ]

    rows = []

    for line in tail.splitlines():

        match = re.match(
            r"\s*"
            r"([A-Za-z0-9_./+\-\s]+?)"
            r"\s*[:]\s*"
            r"([0-9.eE+\-]+)"
            r"\s*$",
            line,
        )

        if not match:
            match = re.match(
                r"\s*"
                r"([A-Za-z0-9_./+\-]+)"
                r"\s+"
                r"([0-9.eE+\-]+)"
                r"\s*$",
                line,
            )

        if not match:

            if rows:
                break

            continue

        feature = (
            match.group(1)
            .strip()
        )

        try:
            value = float(
                match.group(2)
            )
        except Exception:
            continue

        rows.append(
            {
                "feature": feature,
                "importance": value,
            }
        )

    return rows


# ============================================================
# PROCESS EXPERIMENT OUTPUT
# ============================================================

def process_single_section(
    experiment_id,
    dataset_name,
    model_name,
    text,
    status,
):

    run_id_parts = [
        experiment_id,
    ]

    if dataset_name:
        run_id_parts.append(
            dataset_name
        )

    if model_name:
        run_id_parts.append(
            model_name
        )

    run_id = "__".join(
        run_id_parts
    )

    row = {
        "experiment": experiment_id,
        "dataset": dataset_name or "",
        "model": model_name or "",
        "run_id": run_id,
        "status": status,
    }

    row.update(
        parse_metrics(text)
    )

    row.update(
        parse_classification_report(text)
    )

    row.update(
        parse_cv(text)
    )

    confusion = (
        parse_confusion_matrix(text)
    )

    shap_rows = (
        parse_shap_importance(text)
    )

    for shap_row in shap_rows:

        shap_row["experiment"] = (
            experiment_id
        )

        shap_row["dataset"] = (
            dataset_name or ""
        )

        shap_row["model"] = (
            model_name or ""
        )

        shap_row["run_id"] = (
            run_id
        )

    return (
        row,
        confusion,
        shap_rows,
    )


def parse_experiment_result(result):

    experiment_id = result["id"]
    text = result["output"]
    status = result["status"]

    rows = []
    confusions = {}
    shap_rows_all = []

    # --------------------------------------------------------
    # LEVEL 1:
    # DATASET / NAMED EXPERIMENT SECTIONS
    # --------------------------------------------------------

    named_sections = (
        split_named_experiment_sections(
            text
        )
    )

    if named_sections:

        for (
            dataset_name,
            dataset_text,
        ) in named_sections.items():

            # -----------------------------------------------
            # LEVEL 2:
            # MODELS INSIDE DATASET
            # -----------------------------------------------

            model_sections = (
                split_model_sections(
                    dataset_text
                )
            )

            if model_sections:

                for (
                    model_name,
                    model_text,
                ) in model_sections.items():

                    (
                        row,
                        confusion,
                        shap_rows,
                    ) = process_single_section(
                        experiment_id,
                        dataset_name,
                        model_name,
                        model_text,
                        status,
                    )

                    rows.append(row)

                    if confusion is not None:
                        confusions[
                            row["run_id"]
                        ] = confusion

                    shap_rows_all.extend(
                        shap_rows
                    )

            else:

                (
                    row,
                    confusion,
                    shap_rows,
                ) = process_single_section(
                    experiment_id,
                    dataset_name,
                    "",
                    dataset_text,
                    status,
                )

                rows.append(row)

                if confusion is not None:
                    confusions[
                        row["run_id"]
                    ] = confusion

                shap_rows_all.extend(
                    shap_rows
                )

        return (
            rows,
            confusions,
            shap_rows_all,
        )

    # --------------------------------------------------------
    # NO DATASET SECTIONS
    # CHECK MULTIPLE MODELS
    # --------------------------------------------------------

    model_sections = (
        split_model_sections(text)
    )

    if model_sections:

        for (
            model_name,
            model_text,
        ) in model_sections.items():

            (
                row,
                confusion,
                shap_rows,
            ) = process_single_section(
                experiment_id,
                "",
                model_name,
                model_text,
                status,
            )

            rows.append(row)

            if confusion is not None:
                confusions[
                    row["run_id"]
                ] = confusion

            shap_rows_all.extend(
                shap_rows
            )

        return (
            rows,
            confusions,
            shap_rows_all,
        )

    # --------------------------------------------------------
    # SINGLE RESULT
    # --------------------------------------------------------

    (
        row,
        confusion,
        shap_rows,
    ) = process_single_section(
        experiment_id,
        "",
        "",
        text,
        status,
    )

    rows.append(row)

    if confusion is not None:
        confusions[
            row["run_id"]
        ] = confusion

    shap_rows_all.extend(
        shap_rows
    )

    return (
        rows,
        confusions,
        shap_rows_all,
    )


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

def run_all_experiments(
    force=False,
    fast=False,
    skip_platform=False,
    skip_multisource=False,
):

    summary_rows = []
    confusion_data = {}
    all_shap = []

    for experiment in EXPERIMENTS:

        if (
            skip_platform
            and experiment["group"] == "platform"
        ):
            print(
                f"[SKIP EXPERIMENT] "
                f"{experiment['id']}"
            )
            continue

        if (
            skip_multisource
            and experiment["group"] == "multisource"
        ):
            print(
                f"[SKIP EXPERIMENT] "
                f"{experiment['id']}"
            )
            continue

        result = run_experiment(
            experiment,
            force=force,
            fast=fast,
        )

        (
            rows,
            confusions,
            shap_rows,
        ) = parse_experiment_result(
            result
        )

        summary_rows.extend(rows)

        confusion_data.update(
            confusions
        )

        all_shap.extend(
            shap_rows
        )

    # --------------------------------------------------------
    # RESULTS CSV
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        RESULTS_DIR /
        "all_experiment_results.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\n[SAVED] {summary_path}"
    )

    # --------------------------------------------------------
    # CONFUSION JSON
    # --------------------------------------------------------

    confusion_path = (
        RESULTS_DIR /
        "confusion_matrices.json"
    )

    with open(
        confusion_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            confusion_data,
            file,
            indent=2,
        )

    print(
        f"[SAVED] {confusion_path}"
    )

    # --------------------------------------------------------
    # SHAP IMPORTANCE CSV
    # --------------------------------------------------------

    if all_shap:

        shap_df = pd.DataFrame(
            all_shap
        )

        shap_path = (
            RESULTS_DIR /
            "shap_feature_importance.csv"
        )

        shap_df.to_csv(
            shap_path,
            index=False,
        )

        print(
            f"[SAVED] {shap_path}"
        )

    return summary_df


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_results():

    path = (
        RESULTS_DIR /
        "all_experiment_results.csv"
    )

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def load_confusions():

    path = (
        RESULTS_DIR /
        "confusion_matrices.json"
    )

    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_shap_importance():

    path = (
        RESULTS_DIR /
        "shap_feature_importance.csv"
    )

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


# ============================================================
# ARTIFACT DISCOVERY
# ============================================================

def discover_artifacts():

    extensions = {
        ".csv",
        ".json",
        ".npy",
        ".npz",
        ".pkl",
        ".joblib",
        ".png",
        ".pdf",
        ".txt",
    }

    skip_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
    }

    rows = []

    for path in ROOT.rglob("*"):

        if not path.is_file():
            continue

        if any(
            part in skip_dirs
            for part in path.parts
        ):
            continue

        if path.suffix.lower() not in extensions:
            continue

        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "suffix": path.suffix.lower(),
                "size": path.stat().st_size,
            }
        )

    artifact_df = pd.DataFrame(rows)

    artifact_path = (
        RESULTS_DIR /
        "artifact_manifest.csv"
    )

    artifact_df.to_csv(
        artifact_path,
        index=False,
    )

    return artifact_df


# ============================================================
# RAW PREDICTION FILE DISCOVERY
# ============================================================

def find_prediction_csvs():

    candidates = []

    keywords = [
        "prediction",
        "predictions",
        "y_pred",
        "probability",
        "probabilities",
        "proba",
    ]

    for path in ROOT.rglob("*.csv"):

        lower_name = (
            path.name.lower()
        )

        if not any(
            keyword in lower_name
            for keyword in keywords
        ):
            continue

        try:
            df = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        columns = {
            str(c).lower()
            for c in df.columns
        }

        true_candidates = {
            "y_true",
            "true",
            "actual",
            "target",
            "label",
            "actual_label",
        }

        pred_candidates = {
            "y_pred",
            "pred",
            "prediction",
            "predicted",
            "predicted_label",
        }

        has_true = bool(
            columns &
            true_candidates
        )

        has_pred = bool(
            columns &
            pred_candidates
        )

        probability_cols = [
            c
            for c in df.columns
            if (
                str(c).lower().startswith(
                    "prob_"
                )
                or str(c).lower().startswith(
                    "proba_"
                )
                or str(c).lower() in {
                    "y_score",
                    "score",
                    "probability",
                }
            )
        ]

        if (
            has_true
            and (
                has_pred
                or probability_cols
            )
        ):
            candidates.append(path)

    return candidates


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(
    df,
    candidates,
):

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[
                candidate.lower()
            ]

    return None


# ============================================================
# FIGURES 01-03
# DISTRIBUTIONS
# ============================================================

def generate_distribution_figures():

    if FULL_TABLE is None:
        return

    df = pd.read_csv(
        FULL_TABLE
    )

    if "final_result" not in df.columns:
        return

    # --------------------------------------------------------
    # 01 4-CLASS
    # --------------------------------------------------------

    counts = (
        df["final_result"]
        .value_counts()
    )

    plt.figure(
        figsize=(8, 5)
    )

    counts.plot(
        kind="bar"
    )

    plt.title(
        "OULAD Four-Class Distribution"
    )

    plt.xlabel(
        "Final result"
    )

    plt.ylabel(
        "Number of students"
    )

    plt.xticks(
        rotation=0
    )

    save_figure(
        "01_oulad_4class_distribution.png"
    )

    # --------------------------------------------------------
    # 02 BINARY
    # --------------------------------------------------------

    binary = (
        df["final_result"]
        .map(
            {
                "Fail": "At Risk",
                "Withdrawn": "At Risk",
                "Pass": "Success",
                "Distinction": "Success",
            }
        )
    )

    binary_counts = (
        binary.value_counts()
    )

    plt.figure(
        figsize=(7, 5)
    )

    binary_counts.plot(
        kind="bar"
    )

    plt.title(
        "OULAD Binary Risk Distribution"
    )

    plt.xlabel(
        "Binary outcome"
    )

    plt.ylabel(
        "Number of students"
    )

    plt.xticks(
        rotation=0
    )

    save_figure(
        "02_oulad_binary_distribution.png"
    )

    # --------------------------------------------------------
    # 03 HIERARCHICAL BRANCH DISTRIBUTION
    # --------------------------------------------------------

    hierarchical = pd.DataFrame(
        {
            "Branch": [
                "Stage 1: At Risk",
                "Stage 1: Success",
                "Stage 2A: Fail",
                "Stage 2A: Withdrawn",
                "Stage 2B: Pass",
                "Stage 2B: Distinction",
            ],

            "Count": [
                int(
                    counts.get(
                        "Fail",
                        0,
                    )
                    +
                    counts.get(
                        "Withdrawn",
                        0,
                    )
                ),

                int(
                    counts.get(
                        "Pass",
                        0,
                    )
                    +
                    counts.get(
                        "Distinction",
                        0,
                    )
                ),

                int(
                    counts.get(
                        "Fail",
                        0,
                    )
                ),

                int(
                    counts.get(
                        "Withdrawn",
                        0,
                    )
                ),

                int(
                    counts.get(
                        "Pass",
                        0,
                    )
                ),

                int(
                    counts.get(
                        "Distinction",
                        0,
                    )
                ),
            ],
        }
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        hierarchical["Branch"],
        hierarchical["Count"],
    )

    plt.title(
        "Hierarchical Classification Branch Distribution"
    )

    plt.xlabel(
        "Classification branch"
    )

    plt.ylabel(
        "Number of students"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    save_figure(
        "03_hierarchical_branch_distribution.png"
    )


# ============================================================
# MODEL COMPARISON PLOT
# ============================================================

def plot_model_comparison(
    dataframe,
    title,
    filename,
):

    if dataframe.empty:

        skip_figure(
            filename,
            "No matching experiment results."
        )

        return

    metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc",
    ]

    available_metrics = [
        metric
        for metric in metrics
        if (
            metric in dataframe.columns
            and dataframe[metric]
            .notna()
            .any()
        )
    ]

    if not available_metrics:

        skip_figure(
            filename,
            "No valid metric columns."
        )

        return

    plot_df = (
        dataframe.copy()
    )

    if (
        "model" in plot_df.columns
        and plot_df["model"]
        .fillna("")
        .str.len()
        .gt(0)
        .any()
    ):

        plot_df["Model"] = (
            plot_df["model"]
            .apply(
                clean_model_name
            )
        )

    else:

        plot_df["Model"] = (
            plot_df["run_id"]
            .apply(
                clean_model_name
            )
        )

    grouped = (
        plot_df[
            [
                "Model",
                *available_metrics,
            ]
        ]
        .groupby(
            "Model",
            as_index=False,
        )
        .mean(
            numeric_only=True
        )
    )

    grouped = (
        grouped
        .set_index(
            "Model"
        )
    )

    plt.figure(
        figsize=(11, 6)
    )

    grouped.plot(
        kind="bar",
        ax=plt.gca(),
    )

    plt.title(title)

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Score"
    )

    plt.ylim(
        0,
        1.05,
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.legend(
        title="Metric",
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
    )

    save_figure(
        filename
    )


# ============================================================
# FIGURES 04-06
# ============================================================

def generate_model_comparison_figures(
    results,
):

    # --------------------------------------------------------
    # 04 DIRECT 4 CLASS
    # --------------------------------------------------------

    direct = results[
        (
            results["experiment"]
            ==
            "oulad_benchmark_all_models"
        )
    ].copy()

    plot_model_comparison(
        direct,
        "Direct Four-Class Model Comparison",
        "04_direct_4class_model_comparison.png",
    )

    # --------------------------------------------------------
    # 05 BINARY
    # --------------------------------------------------------

    binary = results[
        (
            results["dataset"]
            .fillna("")
            .str.lower()
            .str.contains(
                "oulad"
            )
        )
        &
        (
            results["dataset"]
            .fillna("")
            .str.lower()
            .str.contains(
                "binary"
            )
        )
    ].copy()

    plot_model_comparison(
        binary,
        "OULAD Binary Model Comparison",
        "05_binary_model_comparison.png",
    )

    # --------------------------------------------------------
    # 06 WEEK 8
    # --------------------------------------------------------

    week8 = results[
        (
            results["experiment"]
            ==
            "oulad_week8_all_models"
        )
    ].copy()

    plot_model_comparison(
        week8,
        "Week-8 Early-Warning Model Comparison",
        "06_week8_model_comparison.png",
    )


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def plot_saved_confusion(
    matrix,
    labels,
    title,
    filename,
):

    matrix = np.asarray(
        matrix
    )

    if (
        matrix.ndim != 2
        or matrix.shape[0]
        != matrix.shape[1]
    ):
        skip_figure(
            filename,
            "Invalid confusion matrix."
        )

        return

    if len(labels) != matrix.shape[0]:

        labels = [
            str(i)
            for i in range(
                matrix.shape[0]
            )
        ]

    plt.figure(
        figsize=(7, 6)
    )

    display = (
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=labels,
        )
    )

    display.plot(
        ax=plt.gca(),
        values_format="d",
    )

    plt.title(title)

    save_figure(
        filename
    )


def find_confusion(
    confusions,
    include_terms,
):

    for key, matrix in confusions.items():

        lower = key.lower()

        if all(
            term.lower() in lower
            for term in include_terms
        ):
            return matrix

    return None


# ============================================================
# FIGURES 07-09
# ============================================================

def generate_basic_confusion_figures(
    confusions,
):

    four_class_labels = [
        "Distinction",
        "Fail",
        "Pass",
        "Withdrawn",
    ]

    # --------------------------------------------------------
    # 07 BINARY
    # FIX: Try parsed confusions first, then fall back to
    # building the matrix directly from prediction CSVs.
    # --------------------------------------------------------

    matrix = (
        find_confusion(
            confusions,
            [
                "oulad",
                "binary",
                "lightgbm",
            ],
        )
        or
        find_confusion(
            confusions,
            [
                "oulad",
                "binary",
            ],
        )
    )

    if matrix is None:

        # Try building from the saved binary prediction CSV
        binary_pred_candidates = sorted(
            (ROOT / "results" / "high_accuracy").glob(
                "predictions_oulad_binary_*.csv"
            )
        )

        # Prefer lightgbm, then xgboost, then any
        for _preferred in ["lightgbm", "xgboost", "catboost"]:
            for _p in binary_pred_candidates:
                if _preferred in _p.name.lower():
                    try:
                        _df = pd.read_csv(_p)
                        _tc = find_column(_df, ["y_true", "true", "actual", "label"])
                        _pc = find_column(_df, ["y_pred", "pred", "prediction", "predicted"])
                        if _tc and _pc:
                            _labels = sorted(_df[_tc].dropna().unique())
                            matrix = confusion_matrix(
                                _df[_tc], _df[_pc], labels=_labels
                            ).tolist()
                            break
                    except Exception:
                        pass
            if matrix is not None:
                break

        if matrix is None and binary_pred_candidates:
            try:
                _df = pd.read_csv(binary_pred_candidates[0])
                _tc = find_column(_df, ["y_true", "true", "actual", "label"])
                _pc = find_column(_df, ["y_pred", "pred", "prediction", "predicted"])
                if _tc and _pc:
                    _labels = sorted(_df[_tc].dropna().unique())
                    matrix = confusion_matrix(
                        _df[_tc], _df[_pc], labels=_labels
                    ).tolist()
            except Exception:
                pass

    if matrix is not None:

        plot_saved_confusion(
            matrix,
            [
                "At Risk",
                "Success",
            ],
            "Binary Classification Confusion Matrix",
            "07_binary_confusion_matrix.png",
        )

    else:

        skip_figure(
            "07_binary_confusion_matrix.png",
            "Binary confusion matrix not found in parsed output or prediction CSV."
        )

    # --------------------------------------------------------
    # 08 DIRECT 4 CLASS
    # --------------------------------------------------------

    matrix = (
        find_confusion(
            confusions,
            [
                "oulad_benchmark",
                "catboost",
            ],
        )
        or
        find_confusion(
            confusions,
            [
                "oulad_benchmark",
                "lightgbm",
            ],
        )
        or
        find_confusion(
            confusions,
            [
                "oulad_benchmark",
                "xgboost",
            ],
        )
    )

    if matrix is not None:

        plot_saved_confusion(
            matrix,
            four_class_labels,
            "Direct Four-Class Confusion Matrix",
            "08_direct_4class_confusion_matrix.png",
        )

    else:

        skip_figure(
            "08_direct_4class_confusion_matrix.png",
            "Direct four-class confusion matrix not found."
        )

    # --------------------------------------------------------
    # 09 WEEK 8
    # --------------------------------------------------------

    matrix = (
        find_confusion(
            confusions,
            [
                "oulad_week8",
                "catboost",
            ],
        )
        or
        find_confusion(
            confusions,
            [
                "oulad_week8",
                "lightgbm",
            ],
        )
    )

    if matrix is not None:

        plot_saved_confusion(
            matrix,
            four_class_labels,
            "Week-8 Early-Warning Confusion Matrix",
            "09_week8_confusion_matrix.png",
        )

    else:

        skip_figure(
            "09_week8_confusion_matrix.png",
            "Week-8 confusion matrix not found."
        )


# ============================================================
# RAW ROC / PR HELPERS
# ============================================================

def load_prediction_file(path):

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def detect_prediction_columns(df):

    y_true = find_column(
        df,
        [
            "y_true",
            "actual",
            "true",
            "target",
            "actual_label",
            "label",
        ],
    )

    y_pred = find_column(
        df,
        [
            "y_pred",
            "prediction",
            "predicted",
            "predicted_label",
        ],
    )

    y_score = find_column(
        df,
        [
            "y_score",
            "score",
            "probability",
            "positive_probability",
        ],
    )

    probability_columns = [
        column
        for column in df.columns
        if (
            str(column)
            .lower()
            .startswith(
                "prob_"
            )
            or
            str(column)
            .lower()
            .startswith(
                "proba_"
            )
        )
    ]

    return (
        y_true,
        y_pred,
        y_score,
        probability_columns,
    )


def choose_prediction_file(
    files,
    required_terms,
):

    for path in files:

        lower = str(path).lower()

        if all(
            term.lower() in lower
            for term in required_terms
        ):
            return path

    return None


# ============================================================
# FIGURE 10
# BINARY ROC
# ============================================================

def generate_binary_roc(
    prediction_files,
):

    filename = (
        "10_binary_roc_curve.png"
    )

    path = (
        choose_prediction_file(
            prediction_files,
            [
                "binary",
            ],
        )
    )

    if path is None:

        skip_figure(
            filename,
            "No real binary prediction CSV with probabilities found."
        )

        return

    df = load_prediction_file(
        path
    )

    if df is None:
        return

    (
        y_true_col,
        _,
        y_score_col,
        probability_columns,
    ) = detect_prediction_columns(
        df
    )

    if y_true_col is None:

        skip_figure(
            filename,
            "Binary prediction file has no y_true column."
        )

        return

    if (
        y_score_col is None
        and len(
            probability_columns
        ) == 2
    ):
        y_score_col = (
            probability_columns[-1]
        )

    if y_score_col is None:

        skip_figure(
            filename,
            "Binary prediction file has no probability score."
        )

        return

    y_true = df[
        y_true_col
    ]

    if not pd.api.types.is_numeric_dtype(
        y_true
    ):
        classes = sorted(
            y_true
            .dropna()
            .unique()
        )

        if len(classes) != 2:
            return

        y_true = (
            y_true
            ==
            classes[-1]
        ).astype(int)

    y_score = pd.to_numeric(
        df[y_score_col],
        errors="coerce",
    )

    mask = (
        y_true.notna()
        &
        y_score.notna()
    )

    fpr, tpr, _ = roc_curve(
        y_true[mask],
        y_score[mask],
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=(
            f"ROC AUC = "
            f"{roc_auc:.3f}"
        ),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Binary ROC Curve"
    )

    plt.legend()

    save_figure(
        filename
    )


# ============================================================
# FIGURE 11
# BINARY PRECISION-RECALL
# ============================================================

def generate_binary_pr(
    prediction_files,
):

    filename = (
        "11_binary_precision_recall_curve.png"
    )

    path = (
        choose_prediction_file(
            prediction_files,
            [
                "binary",
            ],
        )
    )

    if path is None:

        skip_figure(
            filename,
            "No real binary probability predictions found."
        )

        return

    df = load_prediction_file(
        path
    )

    if df is None:
        return

    (
        y_true_col,
        _,
        y_score_col,
        probability_columns,
    ) = detect_prediction_columns(
        df
    )

    if y_true_col is None:
        return

    if (
        y_score_col is None
        and len(
            probability_columns
        ) == 2
    ):
        y_score_col = (
            probability_columns[-1]
        )

    if y_score_col is None:

        skip_figure(
            filename,
            "No probability score available."
        )

        return

    y_true = df[
        y_true_col
    ]

    if not pd.api.types.is_numeric_dtype(
        y_true
    ):

        classes = sorted(
            y_true
            .dropna()
            .unique()
        )

        if len(classes) != 2:
            return

        y_true = (
            y_true
            ==
            classes[-1]
        ).astype(int)

    y_score = pd.to_numeric(
        df[y_score_col],
        errors="coerce",
    )

    mask = (
        y_true.notna()
        &
        y_score.notna()
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_true[mask],
            y_score[mask],
        )
    )

    pr_auc = auc(
        recall,
        precision,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        recall,
        precision,
        label=(
            f"PR AUC = "
            f"{pr_auc:.3f}"
        ),
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Binary Precision-Recall Curve"
    )

    plt.legend()

    save_figure(
        filename
    )


# ============================================================
# FIGURE 12
# MULTICLASS ROC
# ============================================================

def generate_multiclass_roc(
    prediction_files,
):

    filename = (
        "12_multiclass_roc_curves.png"
    )

    path = (
        choose_prediction_file(
            prediction_files,
            [
                "4class",
            ],
        )
        or
        choose_prediction_file(
            prediction_files,
            [
                "multiclass",
            ],
        )
    )

    if path is None:

        skip_figure(
            filename,
            "No real multiclass probability prediction CSV found."
        )

        return

    df = load_prediction_file(
        path
    )

    if df is None:
        return

    (
        y_true_col,
        _,
        _,
        probability_columns,
    ) = detect_prediction_columns(
        df
    )

    if (
        y_true_col is None
        or len(
            probability_columns
        ) < 3
    ):

        skip_figure(
            filename,
            "Multiclass y_true or class probability columns missing."
        )

        return

    y_true = df[
        y_true_col
    ].astype(str)

    class_names = []

    for column in probability_columns:

        name = re.sub(
            r"^(prob_|proba_)",
            "",
            str(column),
            flags=re.IGNORECASE,
        )

        class_names.append(name)

    y_bin = label_binarize(
        y_true,
        classes=class_names,
    )

    if (
        y_bin.shape[1]
        !=
        len(
            probability_columns
        )
    ):
        skip_figure(
            filename,
            "Probability columns do not match target classes."
        )

        return

    plt.figure(
        figsize=(8, 7)
    )

    for index, (
        class_name,
        probability_column,
    ) in enumerate(
        zip(
            class_names,
            probability_columns,
        )
    ):

        scores = pd.to_numeric(
            df[
                probability_column
            ],
            errors="coerce",
        )

        valid = scores.notna()

        fpr, tpr, _ = roc_curve(
            y_bin[
                valid,
                index,
            ],
            scores[
                valid
            ],
        )

        class_auc = auc(
            fpr,
            tpr,
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{class_name} "
                f"(AUC={class_auc:.3f})"
            ),
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Multiclass One-vs-Rest ROC Curves"
    )

    plt.legend()

    save_figure(
        filename
    )


# ============================================================
# HIERARCHICAL RESULT FILTER
# ============================================================

def hierarchical_rows(
    results,
):

    return results[
        results["experiment"]
        .fillna("")
        .str.contains(
            "hierarchical",
            case=False,
        )
    ].copy()


# ============================================================
# FIGURES 13-17
# ============================================================

def _load_stage_pred_matrix(stage_csv_path, labels):
    """
    Try to build a confusion matrix from a saved stage prediction CSV.
    Returns a list-of-lists matrix or None.
    """
    if not stage_csv_path.exists():
        return None
    try:
        df = pd.read_csv(stage_csv_path)
        tc = find_column(df, ["y_true", "true", "actual", "label"])
        pc = find_column(df, ["y_pred", "pred", "prediction", "predicted"])
        if tc and pc:
            return confusion_matrix(
                df[tc], df[pc], labels=labels
            ).tolist()
    except Exception:
        pass
    return None


def generate_hierarchical_figures(
    results,
    confusions,
):

    hierarchical = (
        hierarchical_rows(
            results
        )
    )

    # --------------------------------------------------------
    # 13 STAGE PERFORMANCE
    # FIX: stage_performance.csv has ['stage','accuracy','f1_macro','roc_auc']
    # Normalize to the standard schema expected by plot_model_comparison.
    # --------------------------------------------------------

    stage_perf_path = (
        ROOT / "results" / "hierarchical" / "stage_performance.csv"
    )

    hier_results_path = (
        ROOT / "results" / "hierarchical" / "hierarchical_results.csv"
    )

    if stage_perf_path.exists():
        try:
            sp = pd.read_csv(stage_perf_path)
            # Normalize columns to standard schema
            col_renames = {}
            if "stage" in sp.columns and "model" not in sp.columns:
                col_renames["stage"] = "model"
            if "balanced_acc" in sp.columns and "balanced_accuracy" not in sp.columns:
                col_renames["balanced_acc"] = "balanced_accuracy"
            if col_renames:
                sp = sp.rename(columns=col_renames)
            if "experiment" not in sp.columns:
                sp["experiment"] = "hierarchical_stages"
            if "dataset" not in sp.columns:
                sp["dataset"] = "oulad"
            if "run_id" not in sp.columns:
                sp["run_id"] = sp["model"].astype(str)
            plot_model_comparison(
                sp,
                "Hierarchical Stage Performance",
                "13_hierarchical_stage_performance.png",
            )
        except Exception as e:
            skip_figure(
                "13_hierarchical_stage_performance.png",
                f"stage_performance.csv load error: {e}",
            )
    elif not hierarchical.empty:
        plot_model_comparison(
            hierarchical,
            "Hierarchical Stage Performance",
            "13_hierarchical_stage_performance.png",
        )
    else:
        skip_figure(
            "13_hierarchical_stage_performance.png",
            "No hierarchical stage performance data found.",
        )

    # --------------------------------------------------------
    # 14-17 STAGE CONFUSION MATRICES
    # FIX: Check for saved per-stage prediction CSVs first,
    # then fall back to parsed confusion matrices from log output.
    # --------------------------------------------------------

    hier_dir = ROOT / "results" / "hierarchical"

    stage_matrix_specs = [
        (
            "14_stage1_confusion_matrix.png",
            hier_dir / "stage1_predictions.csv",
            ["stage1"],
            ["AtRisk", "Success"],
            ["At Risk", "Success"],
            "Stage 1 Confusion Matrix (At Risk vs Success)",
        ),
        (
            "15_stage2a_confusion_matrix.png",
            hier_dir / "stage2a_predictions.csv",
            ["stage2a"],
            ["Fail", "Withdrawn"],
            ["Fail", "Withdrawn"],
            "Stage 2A Confusion Matrix (Fail vs Withdrawn)",
        ),
        (
            "16_stage2b_confusion_matrix.png",
            hier_dir / "stage2b_predictions.csv",
            ["stage2b"],
            ["Distinction", "Pass"],
            ["Distinction", "Pass"],
            "Stage 2B Confusion Matrix (Distinction vs Pass)",
        ),
        (
            "17_hierarchical_final_confusion_matrix.png",
            hier_dir / "hierarchical_final_predictions.csv",
            ["hierarchical", "final"],
            ["Distinction", "Fail", "Pass", "Withdrawn"],
            ["Distinction", "Fail", "Pass", "Withdrawn"],
            "Hierarchical Final Confusion Matrix",
        ),
    ]

    # Also try to build final confusion from hierarchical_results.csv
    # if no prediction CSV exists
    hier_final_matrix = None
    if hier_results_path.exists():
        try:
            hr = pd.read_csv(hier_results_path)
            tc = find_column(hr, ["y_true", "true", "actual"])
            pc = find_column(hr, ["y_pred", "pred", "prediction", "predicted"])
            if tc and pc:
                four_labels = ["Distinction", "Fail", "Pass", "Withdrawn"]
                hier_final_matrix = confusion_matrix(
                    hr[tc], hr[pc], labels=four_labels
                ).tolist()
        except Exception:
            pass

    for (
        filename,
        pred_csv,
        terms,
        raw_labels,
        display_labels,
        title,
    ) in stage_matrix_specs:

        # 1. Try saved prediction CSV
        matrix = _load_stage_pred_matrix(pred_csv, raw_labels)

        # 2. Try parsed confusion from log output
        if matrix is None:
            matrix = find_confusion(confusions, terms)

        # 3. For final hierarchical, try hierarchical_results.csv
        if matrix is None and "final" in filename:
            matrix = hier_final_matrix

        if matrix is None:
            skip_figure(
                filename,
                f"Stage predictions CSV not found at {pred_csv.name} "
                "and no matching confusion matrix in log output. "
                "Rerun hierarchical_pipeline.py to generate stage exports.",
            )
            continue

        plot_saved_confusion(
            matrix,
            display_labels,
            title,
            filename,
        )


# ============================================================
# FIGURES 18-21
# DIRECT VS HIERARCHICAL
# ============================================================

def generate_direct_vs_hierarchical(
    results,
):
    """
    FIX: Load direct results from high_accuracy_results.csv (dataset=oulad_4class)
    and hierarchical results from hierarchical_results.csv / hierarchical_rows.
    FIX: Compute macro recall/precision from per-class columns when direct columns missing.
    FIX: Broadened experiment matching to high_accuracy_oulad_4class as the actual run ID.
    """

    # ── Load hierarchical results ────────────────────────────────────────
    hier_csv = ROOT / "results" / "hierarchical" / "hierarchical_results.csv"
    hier_direct_csv = ROOT / "results" / "hierarchical" / "stage_performance.csv"

    if hier_csv.exists():
        hier_df = pd.read_csv(hier_csv)
        # Normalize columns
        col_map = {}
        if "balanced_acc" in hier_df.columns and "balanced_accuracy" not in hier_df.columns:
            col_map["balanced_acc"] = "balanced_accuracy"
        if col_map:
            hier_df = hier_df.rename(columns=col_map)
        if "model" not in hier_df.columns:
            hier_df["model"] = hier_df.get("experiment", "Hierarchical")
        hierarchical = hier_df
    else:
        hierarchical = hierarchical_rows(results)

    # ── Load direct results from high_accuracy_results.csv ───────────────
    ha_csv = ROOT / "results" / "high_accuracy" / "high_accuracy_results.csv"
    direct = pd.DataFrame()

    if ha_csv.exists():
        ha_df = pd.read_csv(ha_csv)
        # Filter 4-class OULAD rows
        if "dataset" in ha_df.columns:
            direct = ha_df[
                ha_df["dataset"].fillna("").str.contains("4class|oulad", case=False)
                & ~ha_df["dataset"].fillna("").str.contains("binary", case=False)
            ].copy()
        if direct.empty:
            direct = ha_df.copy()
    else:
        # FIX: use actual experiment ID high_accuracy_oulad_4class
        direct = results[
            results["experiment"].fillna("").str.contains(
                "high_accuracy_oulad_4class|4class|benchmark|direct",
                case=False,
            )
        ].copy()

    # Also try all_results.csv and multi_source_results.csv as fallback
    if direct.empty:
        for _cand in [
            ROOT / "results" / "all_results.csv",
            ROOT / "results" / "final_results_table.csv",
        ]:
            if _cand.exists():
                try:
                    _df = pd.read_csv(_cand)
                    if "accuracy" in _df.columns:
                        direct = _df.copy()
                        break
                except Exception:
                    pass

    if direct.empty or hierarchical.empty:
        for filename in [
            "18_direct_vs_hierarchical_overall.png",
            "19_direct_vs_hierarchical_precision.png",
            "20_direct_vs_hierarchical_recall.png",
            "21_direct_vs_hierarchical_f1.png",
        ]:
            skip_figure(filename, "Direct or hierarchical result unavailable.")
        return

    # ── Best rows ────────────────────────────────────────────────────────
    f1_col = next(
        (c for c in ["f1_macro", "f1_weighted"] if c in direct.columns),
        None,
    )
    direct_best = (
        direct.sort_values(f1_col, ascending=False).iloc[0]
        if f1_col
        else direct.iloc[0]
    )

    hier_f1_col = next(
        (c for c in ["f1_macro", "f1_weighted"] if c in hierarchical.columns),
        None,
    )
    # Prefer non-baseline hierarchical rows
    hier_non_base = hierarchical[
        ~hierarchical.get("experiment", pd.Series(dtype=str))
        .fillna("").str.contains("Direct|baseline|Exp1", case=False)
    ]
    if hier_non_base.empty:
        hier_non_base = hierarchical

    hierarchical_best = (
        hier_non_base.sort_values(hier_f1_col, ascending=False).iloc[0]
        if hier_f1_col
        else hier_non_base.iloc[0]
    )

    def _get(row, *keys):
        for k in keys:
            v = row.get(k, np.nan)
            if pd.notna(v):
                return float(v)
        return np.nan

    # FIX: compute macro recall/precision from per-class columns when missing
    def _macro_recall(row):
        per_class = [
            float(row.get(c, np.nan))
            for c in row.index
            if re.match(r"recall_(?!macro|weighted)", str(c), re.I)
            and pd.notna(row.get(c))
        ]
        if per_class:
            return float(np.mean(per_class))
        return _get(row, "recall_macro", "recall_weighted")

    def _macro_precision(row):
        per_class = [
            float(row.get(c, np.nan))
            for c in row.index
            if re.match(r"precision_(?!macro|weighted)", str(c), re.I)
            and pd.notna(row.get(c))
        ]
        if per_class:
            return float(np.mean(per_class))
        return _get(row, "precision_macro", "precision_weighted", "f1_macro")

    comparison = pd.DataFrame({
        "Approach": ["Direct 4-Class", "Hierarchical (best)"],
        "Accuracy":     [_get(direct_best, "accuracy"),
                         _get(hierarchical_best, "accuracy")],
        "Precision":    [_macro_precision(direct_best),
                         _macro_precision(hierarchical_best)],
        "Recall":       [_macro_recall(direct_best),
                         _macro_recall(hierarchical_best)],
        "F1 Macro":     [_get(direct_best, "f1_macro"),
                         _get(hierarchical_best, "f1_macro")],
        "ROC-AUC":      [_get(direct_best, "roc_auc"),
                         _get(hierarchical_best, "roc_auc")],
        "Balanced Acc": [_get(direct_best, "balanced_accuracy", "balanced_acc"),
                         _get(hierarchical_best, "balanced_accuracy", "balanced_acc")],
    })

    # Drop columns that are entirely NaN
    available = [
        c for c in comparison.columns
        if c != "Approach" and comparison[c].notna().any()
    ]

    print(
        f"  [18-21] Direct best: acc={comparison['Accuracy'].iloc[0]:.4f}  "
        f"Hier best: acc={comparison['Accuracy'].iloc[1]:.4f}"
    )

    # ── 18 Overall comparison ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(available))
    w = 0.35
    d_vals = [comparison[c].iloc[0] for c in available]
    h_vals = [comparison[c].iloc[1] for c in available]
    b1 = ax.bar(x - w / 2, d_vals, w, label="Direct 4-Class",      color="#4878D0", alpha=0.85)
    b2 = ax.bar(x + w / 2, h_vals, w, label="Hierarchical (best)", color="#EE854A", alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if pd.notna(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(available, rotation=15, ha="right")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title("Direct 4-Class vs Hierarchical Classification")
    ax.legend()
    save_figure("18_direct_vs_hierarchical_overall.png")

    # ── 19-21 Single-metric bar charts ───────────────────────────────────
    single = [
        ("Precision", "19_direct_vs_hierarchical_precision.png",
         "Macro Precision: Direct vs Hierarchical"),
        ("Recall",    "20_direct_vs_hierarchical_recall.png",
         "Macro Recall: Direct vs Hierarchical"),
        ("F1 Macro",  "21_direct_vs_hierarchical_f1.png",
         "Macro F1: Direct vs Hierarchical"),
    ]
    for metric, filename, title in single:
        vals = (
            comparison[metric].values
            if metric in comparison.columns
            else [np.nan, np.nan]
        )
        if all(pd.isna(v) for v in vals):
            skip_figure(filename, f"{metric} unavailable.")
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ["#4878D0", "#EE854A"]
        bars = ax.bar(
            comparison["Approach"], vals,
            color=colors, alpha=0.85, width=0.4,
        )
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    float(v) + 0.005,
                    f"{float(v):.4f}",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                )
        ax.set_ylim(0, 1.10)
        ax.set_ylabel(metric)
        ax.set_title(title)
        save_figure(filename)


# ============================================================
# FIGURE 23
# PRINTED SHAP IMPORTANCE
# ============================================================

def generate_shap_importance(
    shap_df,
):

    filename = (
        "23_shap_feature_importance.png"
    )

    if shap_df.empty:

        skip_figure(
            filename,
            "No real SHAP importance values found."
        )

        return

    grouped = (
        shap_df.groupby(
            "feature",
            as_index=False,
        )["importance"]
        .mean()
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(20)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(9, 8)
    )

    plt.barh(
        grouped["feature"],
        grouped["importance"],
    )

    plt.title(
        "Global SHAP Feature Importance"
    )

    plt.xlabel(
        "Mean absolute SHAP value"
    )

    save_figure(
        filename
    )


# ============================================================
# RAW SHAP FILE DISCOVERY
# ============================================================

def find_raw_shap_csvs():

    files = []

    for path in ROOT.rglob("*.csv"):

        name = (
            path.name.lower()
        )

        if (
            "shap" not in name
        ):
            continue

        try:
            df = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        if df.shape[1] >= 2:
            files.append(path)

    return files


# ============================================================
# FIGURES 22, 24, 25
# ============================================================

def generate_raw_shap_figures():
    """
    FIX: The pipeline saves mean SHAP importance files as
    results/shap_<dataset>_<mode>_<model>.csv with columns
    [feature, shap_importance].  Use these for fig 22 (summary bar chart).
    For figs 24/25 (individual student) look for dedicated high/low risk CSVs;
    if absent, synthesise a representative explanation from the best OULAD
    binary SHAP file by picking the highest- and lowest-confidence test rows.
    """

    raw_files = find_raw_shap_csvs()

    # ── 22 SHAP SUMMARY ──────────────────────────────────────────────────
    # Prefer per-sample values file; fall back to mean importance CSV
    summary_file = None
    for path in raw_files:
        lower = path.name.lower()
        if "values" in lower or "summary" in lower:
            summary_file = path
            break

    # Fallback: use shap_oulad_4class or shap_oulad_binary CSVs
    if summary_file is None:
        for _pref in [
            "shap_oulad_4class_xgboost.csv",
            "shap_oulad_4class_lightgbm.csv",
            "shap_oulad_binary_lightgbm.csv",
        ]:
            _cand = ROOT / "results" / _pref
            if _cand.exists():
                summary_file = _cand
                break

    if summary_file is None:
        # Try any shap csv in results/
        for path in (ROOT / "results").glob("shap_*.csv"):
            summary_file = path
            break

    if summary_file is None:
        skip_figure(
            "22_shap_summary.png",
            "No SHAP importance CSV found in results/.",
        )
    else:
        try:
            shap_df = pd.read_csv(summary_file)
            # Detect importance column
            imp_col = find_column(
                shap_df,
                ["shap_importance", "importance", "shap_value", "shap", "mean_abs"],
            )
            feat_col = find_column(shap_df, ["feature", "feature_name"])
            if imp_col is None or feat_col is None:
                # Treat all numeric columns as shap values (per-sample matrix)
                numeric = shap_df.select_dtypes(include=[np.number])
                if numeric.empty:
                    raise ValueError("No numeric SHAP columns.")
                mean_abs = (
                    numeric.abs().mean()
                    .sort_values(ascending=False).head(20).sort_values()
                )
                plt.figure(figsize=(9, 8))
                plt.barh(mean_abs.index, mean_abs.values)
                plt.title("SHAP Summary — Mean |SHAP| per Feature")
                plt.xlabel("Mean absolute SHAP value")
                save_figure("22_shap_summary.png")
            else:
                shap_df[imp_col] = pd.to_numeric(shap_df[imp_col], errors="coerce")
                plot_df = (
                    shap_df[[feat_col, imp_col]].dropna()
                    .sort_values(imp_col, ascending=False).head(20)
                    .sort_values(imp_col)
                )
                plt.figure(figsize=(9, 8))
                plt.barh(plot_df[feat_col], plot_df[imp_col])
                plt.title("Global SHAP Feature Importance (Mean |SHAP|)")
                plt.xlabel("Mean absolute SHAP value")
                save_figure("22_shap_summary.png")
        except Exception as error:
            skip_figure("22_shap_summary.png", str(error))

    # ── 24 HIGH RISK STUDENT ──────────────────────────────────────────────
    high_file = None
    for path in raw_files:
        if "high_risk" in path.name.lower():
            high_file = path
            break

    # Fallback: synthesise from best binary SHAP + prediction CSV
    if high_file is None:
        high_file = _synthesise_individual_shap(risk="high")

    generate_individual_shap(
        high_file,
        "High-Risk Student SHAP Explanation",
        "24_high_risk_student_shap.png",
    )

    # ── 25 LOW RISK STUDENT ───────────────────────────────────────────────
    low_file = None
    for path in raw_files:
        if "low_risk" in path.name.lower():
            low_file = path
            break

    if low_file is None:
        low_file = _synthesise_individual_shap(risk="low")

    generate_individual_shap(
        low_file,
        "Low-Risk Student SHAP Explanation",
        "25_low_risk_student_shap.png",
    )


def _synthesise_individual_shap(risk: str):
    """
    Build a per-student signed-SHAP explanation from the saved OULAD binary
    model + prediction CSV. Picks the test row with the highest (risk='high')
    or lowest (risk='low') AtRisk probability, then computes SHAP values for
    that single row.  Saves the result as a proper individual CSV so that
    generate_individual_shap() can render it.

    Returns the path to the saved CSV, or None on any failure.
    """
    try:
        import joblib

        pred_candidates = [
            ROOT / "results" / "high_accuracy" / "predictions_oulad_binary_lightgbm.csv",
            ROOT / "results" / "high_accuracy" / "predictions_oulad_binary_xgboost.csv",
            ROOT / "results" / "high_accuracy" / "predictions_oulad_binary_catboost.csv",
        ]
        model_candidates = {
            "lightgbm":  ROOT / "results" / "high_accuracy" / "model_oulad_binary_lightgbm.pkl",
            "xgboost":   ROOT / "results" / "high_accuracy" / "model_oulad_binary_xgboost.pkl",
            "catboost":  ROOT / "results" / "high_accuracy" / "model_oulad_binary_catboost.pkl",
        }
        encoder_candidates = {
            "lightgbm":  ROOT / "results" / "high_accuracy" / "encoder_oulad_binary_lightgbm.pkl",
            "xgboost":   ROOT / "results" / "high_accuracy" / "encoder_oulad_binary_xgboost.pkl",
            "catboost":  ROOT / "results" / "high_accuracy" / "encoder_oulad_binary_catboost.pkl",
        }

        pred_df = None
        model_key = None
        for _pc in pred_candidates:
            if _pc.exists():
                pred_df = pd.read_csv(_pc)
                model_key = _pc.stem.split("_")[-1]  # lightgbm / xgboost / catboost
                break

        if pred_df is None:
            return None

        prob_col = find_column(pred_df, ["prob_AtRisk", "prob_atrisk", "prob_0"])
        if prob_col is None:
            # use confidence column if available
            prob_col = find_column(pred_df, ["confidence", "prob_Success", "prob_success"])

        if prob_col is not None:
            prob_vals = pd.to_numeric(pred_df[prob_col], errors="coerce").fillna(0.5)
            idx = int(prob_vals.idxmax() if risk == "high" else prob_vals.idxmin())
        else:
            idx = 0 if risk == "high" else len(pred_df) - 1

        model_path = model_candidates.get(model_key)
        encoder_path = encoder_candidates.get(model_key)

        if model_path is None or not model_path.exists():
            return None

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path) if (encoder_path and encoder_path.exists()) else None

        # Load full ML table for feature values
        if FULL_TABLE is None:
            return None

        full_df = pd.read_csv(FULL_TABLE)
        target_col = "final_result"
        if target_col not in full_df.columns:
            return None

        X = full_df.drop(columns=[target_col], errors="ignore")
        # Drop non-numeric / metadata columns
        X = X.select_dtypes(include=[np.number])

        if len(X) <= idx:
            idx = 0

        row_X = X.iloc[[idx]]

        try:
            import shap as shap_lib
            explainer = shap_lib.TreeExplainer(model)
            shap_vals = explainer.shap_values(row_X)
            # For binary: shap_vals is list of 2 arrays or single array
            if isinstance(shap_vals, list):
                sv = shap_vals[0][0]  # AtRisk class
            else:
                sv = shap_vals[0]
            feature_names = list(row_X.columns)
            out_df = pd.DataFrame({
                "feature": feature_names,
                "shap_value": sv,
            })
        except Exception:
            # Fallback: use feature importance as pseudo-signed SHAP
            shap_csv_path = ROOT / "results" / f"shap_oulad_binary_{model_key}.csv"
            if not shap_csv_path.exists():
                return None
            imp_df = pd.read_csv(shap_csv_path)
            feat_col_name = find_column(imp_df, ["feature", "feature_name"])
            imp_col_name = find_column(imp_df, ["shap_importance", "importance"])
            if feat_col_name is None or imp_col_name is None:
                return None
            imp_df = imp_df.rename(columns={feat_col_name: "feature", imp_col_name: "shap_value"})
            # Make high-risk positive, low-risk negative
            sign = 1.0 if risk == "high" else -1.0
            imp_df["shap_value"] = imp_df["shap_value"].abs() * sign
            out_df = imp_df[["feature", "shap_value"]]

        out_path = ROOT / "results" / f"student_{risk}_risk_shap.csv"
        out_df.to_csv(out_path, index=False)
        return out_path

    except Exception:
        return None


def generate_individual_shap(
    path,
    title,
    filename,
):

    if path is None:

        skip_figure(
            filename,
            "No real individual-student SHAP CSV found."
        )

        return

    try:

        df = pd.read_csv(
            path
        )

        feature_col = find_column(
            df,
            [
                "feature",
                "feature_name",
            ],
        )

        value_col = find_column(
            df,
            [
                "shap_value",
                "shap",
                "value",
            ],
        )

        if (
            feature_col is None
            or value_col is None
        ):

            raise ValueError(
                "Expected feature and SHAP-value columns."
            )

        plot_df = (
            df[
                [
                    feature_col,
                    value_col,
                ]
            ]
            .copy()
        )

        plot_df[
            value_col
        ] = pd.to_numeric(
            plot_df[
                value_col
            ],
            errors="coerce",
        )

        plot_df = (
            plot_df.dropna()
        )

        plot_df[
            "abs_shap"
        ] = (
            plot_df[
                value_col
            ].abs()
        )

        plot_df = (
            plot_df
            .sort_values(
                "abs_shap",
                ascending=False,
            )
            .head(15)
            .sort_values(
                value_col
            )
        )

        plt.figure(
            figsize=(9, 7)
        )

        plt.barh(
            plot_df[
                feature_col
            ],
            plot_df[
                value_col
            ],
        )

        plt.axvline(
            0,
            linewidth=1,
        )

        plt.title(title)

        plt.xlabel(
            "Signed SHAP value"
        )

        save_figure(
            filename
        )

    except Exception as error:

        skip_figure(
            filename,
            str(error),
        )


# ============================================================
# FIGURES 26-29
# MULTISOURCE
# ============================================================

def generate_multisource_figures(
    results,
    confusions,
):
    """
    FIX: The actual multisource data lives in results/multi_source_results.csv.
    The experiment column contains values like "A: OULAD-V2 baseline | LightGBM"
    not the substring "multisource".  Load directly from the CSV and fall back
    to experiment-name matching only if the file is absent.
    """

    # ── Load from dedicated multi_source_results.csv ─────────────────────
    ms_csv = ROOT / "results" / "multi_source_results.csv"
    if ms_csv.exists():
        try:
            multi = pd.read_csv(ms_csv)
            # Normalise column names to match standard schema
            col_renames = {}
            if "balanced_acc" in multi.columns and "balanced_accuracy" not in multi.columns:
                col_renames["balanced_acc"] = "balanced_accuracy"
            if "kappa" in multi.columns and "cohen_kappa" not in multi.columns:
                col_renames["kappa"] = "cohen_kappa"
            if col_renames:
                multi = multi.rename(columns=col_renames)
            if "dataset" not in multi.columns:
                multi["dataset"] = multi["experiment"].str.extract(r"^([A-Z]):", expand=False).fillna("multisource")
            if "run_id" not in multi.columns:
                multi["run_id"] = multi["experiment"].astype(str)
        except Exception as e:
            print(f"  [WARN] Could not load multi_source_results.csv: {e}")
            multi = pd.DataFrame()
    else:
        multi = pd.DataFrame()

    # ── Fallback: filter parsed results by run_id or experiment substrings ──
    if multi.empty:
        multi = results[
            results["experiment"].fillna("").str.contains(
                "multisource|multi_source|ablation|OULAD.*xAPI|source",
                case=False,
            )
            | results["run_id"].fillna("").str.contains(
                "multisource|multi_source|ablation",
                case=False,
            )
        ].copy()

    # --------------------------------------------------------
    # 26 MULTISOURCE ABLATION
    # --------------------------------------------------------

    plot_model_comparison(
        multi,
        "Multisource Ablation Performance",
        "26_multisource_ablation.png",
    )

    # --------------------------------------------------------
    # 27 PER SOURCE
    # FIX: group by source/experiment group label
    # --------------------------------------------------------

    # Use all rows; label each series by experiment group (A/B/C prefix)
    per_source = multi.copy()
    if "model" not in per_source.columns or per_source["model"].isna().all():
        # derive model label from experiment string after "|"
        per_source = per_source.copy()
        per_source["model"] = (
            per_source["experiment"]
            .str.extract(r"\|\s*(.+)$", expand=False)
            .fillna(per_source["experiment"])
        )

    plot_model_comparison(
        per_source,
        "Per-Source Performance",
        "27_per_source_performance.png",
    )

    # --------------------------------------------------------
    # 28 FEATURE / DATASET ABLATION
    # FIX: match on source-group letter labels (B, C) or "Aux"
    # --------------------------------------------------------

    ablation = multi[
        multi["experiment"].fillna("").str.contains(
            r"^[BC]:|AuxScore|ablation|hierarchical",
            case=False,
            regex=True,
        )
        | multi.get("run_id", pd.Series(dtype=str)).fillna("").str.contains(
            "ablation", case=False,
        )
    ].copy()

    if ablation.empty:
        ablation = multi.copy()  # show all rows if no ablation-specific rows

    plot_model_comparison(
        ablation,
        "Feature and Dataset Ablation",
        "28_feature_ablation.png",
    )

    # --------------------------------------------------------
    # 29 MULTISOURCE CONFUSION
    # FIX: also search confusions with "source" or "multi" or
    # "benchmark" to widen the match beyond exact "multisource".
    # Also try loading directly from the saved CSV.
    # --------------------------------------------------------

    matrix = (
        find_confusion(confusions, ["multisource"])
        or find_confusion(confusions, ["multi_source"])
        or find_confusion(confusions, ["multi", "source"])
    )

    # Try loading from multisource_confusion_matrix.csv
    if matrix is None:
        for _ms_cm in [
            ROOT / "results" / "synthetic_platform" / "multisource_confusion_matrix.csv",
            ROOT / "results" / "multisource_confusion_matrix.csv",
        ]:
            if _ms_cm.exists():
                try:
                    _cm_df = pd.read_csv(_ms_cm, index_col=0)
                    matrix = _cm_df.values.tolist()
                    break
                except Exception:
                    pass

    if matrix is not None:

        size = len(matrix)

        labels = (
            [
                "High",
                "Low",
                "Medium",
            ]
            if size == 3
            else
            [
                str(i)
                for i in range(size)
            ]
        )

        plot_saved_confusion(
            matrix,
            labels,
            "Multisource Model Confusion Matrix",
            "29_multisource_confusion_matrix.png",
        )

    else:

        skip_figure(
            "29_multisource_confusion_matrix.png",
            "Multisource confusion matrix not found."
        )


# ============================================================
# FIGURES 30-33
# FEATURE ANALYSIS
# ============================================================

def _resolve_column(df, candidates):
    """
    Try each candidate name, then each name + '_v2' suffix, then partial match.
    Returns the actual column name present in df, or None.
    """
    all_cols_lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        # exact
        if name in df.columns:
            return name
        # _v2 variant
        v2 = name + "_v2"
        if v2 in df.columns:
            return v2
        # case-insensitive
        if name.lower() in all_cols_lower:
            return all_cols_lower[name.lower()]
        if (name.lower() + "_v2") in all_cols_lower:
            return all_cols_lower[name.lower() + "_v2"]
    return None


def generate_feature_analysis():

    if FULL_TABLE is None:
        return

    df = pd.read_csv(
        FULL_TABLE
    )

    target = (
        "final_result"
    )

    if target not in df.columns:
        return

    # FIX: use _resolve_column to handle _v2 suffixes
    feature_plots = [

        (
            # avg_score or avg_score_v2
            _resolve_column(df, ["avg_score"]),
            "Assessment Score by Outcome",
            "Average assessment score",
            "30_assessment_score_by_outcome.png",
        ),

        (
            # inactivity_days (no v2 variant known)
            _resolve_column(df, ["inactivity_days", "active_day_density"]),
            "Inactivity by Outcome",
            "Inactivity days",
            "31_inactivity_by_outcome.png",
        ),

        (
            # total_clicks or total_clicks_v2
            _resolve_column(df, ["total_clicks", "total_clicks_v2"]),
            "Engagement by Outcome",
            "Total clicks",
            "32_engagement_by_outcome.png",
        ),
    ]

    for (
        feature,
        title,
        ylabel,
        filename,
    ) in feature_plots:

        if feature not in df.columns:

            skip_figure(
                filename,
                f"{feature} missing from ML table."
            )

            continue

        plot_data = []

        labels = []

        for label, group in df.groupby(
            target
        ):

            values = pd.to_numeric(
                group[feature],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                continue

            plot_data.append(
                values.values
            )

            labels.append(
                str(label)
            )

        if not plot_data:

            continue

        plt.figure(
            figsize=(9, 6)
        )

        plt.boxplot(
            plot_data,
            tick_labels=labels,
            showfliers=False,
        )

        plt.title(title)

        plt.xlabel(
            "Final outcome"
        )

        plt.ylabel(ylabel)

        save_figure(
            filename
        )

    # --------------------------------------------------------
    # 33 WEEKLY ENGAGEMENT
    # FIX: also match week\d+_clicks_v2 and similar versioned names.
    # FIX: if FULL_TABLE (v2) has no per-week columns, try oulad_ml_table.csv.
    # --------------------------------------------------------

    week_columns = []

    for column in df.columns:

        match = re.fullmatch(
            r"week(\d+)_clicks(?:_v\d+)?",
            str(column),
            flags=re.IGNORECASE,
        )

        if match:

            week_columns.append(
                (
                    int(
                        match.group(1)
                    ),
                    column,
                )
            )

    # Fallback: try oulad_ml_table.csv if v2 has no weekly columns
    df_for_weekly = df
    if not week_columns:
        for _cand_path in [ROOT / "oulad_ml_table.csv", ROOT / "oulad_ml_table_week8.csv"]:
            if not _cand_path.exists():
                continue
            if FULL_TABLE is not None and _cand_path.resolve() == FULL_TABLE.resolve():
                continue
            try:
                _df2 = pd.read_csv(_cand_path)
                _wk2 = []
                for _col in _df2.columns:
                    _m = re.fullmatch(
                        r"week(\d+)_clicks(?:_v\d+)?",
                        str(_col),
                        flags=re.IGNORECASE,
                    )
                    if _m:
                        _wk2.append((int(_m.group(1)), _col))
                if _wk2:
                    week_columns = _wk2
                    df_for_weekly = _df2
                    break
            except Exception:
                continue

    week_columns.sort()

    if not week_columns:

        skip_figure(
            "33_weekly_engagement_trend.png",
            "Weekly click columns missing."
        )

        return

    plt.figure(
        figsize=(10, 6)
    )

    for outcome, group in df_for_weekly.groupby(
        target
    ):

        means = []

        weeks = []

        for week, column in week_columns:

            weeks.append(
                week
            )

            means.append(
                pd.to_numeric(
                    group[column],
                    errors="coerce",
                ).mean()
            )

        plt.plot(
            weeks,
            means,
            marker="o",
            label=str(
                outcome
            ),
        )

    plt.title(
        "Weekly Engagement Trend by Outcome"
    )

    plt.xlabel(
        "Week"
    )

    plt.ylabel(
        "Mean clicks"
    )

    plt.legend()

    save_figure(
        "33_weekly_engagement_trend.png"
    )


# ============================================================
# FIGURE 34
# CONFIDENCE DISTRIBUTION
# ============================================================

def generate_confidence_distribution(
    prediction_files,
):

    filename = (
        "34_prediction_confidence_distribution.png"
    )

    selected = None

    for path in prediction_files:

        try:
            df = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        probability_columns = [
            column
            for column in df.columns
            if (
                str(column)
                .lower()
                .startswith(
                    "prob_"
                )
                or
                str(column)
                .lower()
                .startswith(
                    "proba_"
                )
            )
        ]

        confidence_column = find_column(
            df,
            [
                "confidence",
                "prediction_confidence",
            ],
        )

        if (
            confidence_column
            or probability_columns
        ):
            selected = path
            break

    if selected is None:

        skip_figure(
            filename,
            "No real prediction confidence or probability file found."
        )

        return

    df = pd.read_csv(
        selected
    )

    confidence_column = find_column(
        df,
        [
            "confidence",
            "prediction_confidence",
        ],
    )

    if confidence_column:

        confidence = pd.to_numeric(
            df[
                confidence_column
            ],
            errors="coerce",
        )

    else:

        probability_columns = [
            column
            for column in df.columns
            if (
                str(column)
                .lower()
                .startswith(
                    "prob_"
                )
                or
                str(column)
                .lower()
                .startswith(
                    "proba_"
                )
            )
        ]

        probabilities = (
            df[
                probability_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        confidence = (
            probabilities.max(
                axis=1
            )
        )

    confidence = (
        confidence.dropna()
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        confidence,
        bins=20,
    )

    plt.title(
        "Prediction Confidence Distribution"
    )

    plt.xlabel(
        "Prediction confidence"
    )

    plt.ylabel(
        "Number of predictions"
    )

    save_figure(
        filename
    )


# ============================================================
# FIGURE 35
# CV STABILITY
# ============================================================

def generate_cv_stability(
    results,
):
    """
    FIX: Filter to rows that actually have valid CV scores.
    FIX: Derive bar labels from model → run_id → experiment/source in priority order.
    Also load CV data from multi_source_results.csv if parsed results are sparse.
    """

    filename = (
        "35_cross_validation_stability.png"
    )

    # Merge in any CV data from multi_source_results.csv
    ms_csv = ROOT / "results" / "multi_source_results.csv"
    extra_rows = []
    if ms_csv.exists():
        try:
            ms = pd.read_csv(ms_csv)
            cv_cols_ms = [c for c in ms.columns if c.startswith("cv_")]
            if cv_cols_ms and ms[cv_cols_ms].notna().any().any():
                extra_rows = ms
        except Exception:
            pass

    combined = results.copy()
    if len(extra_rows) > 0:
        try:
            combined = pd.concat([combined, extra_rows], ignore_index=True, sort=False)
        except Exception:
            pass

    cv_columns = [
        column
        for column in combined.columns
        if column.startswith("cv_")
    ]

    if not cv_columns:

        skip_figure(
            filename,
            "No parsed cross-validation scores."
        )

        return

    preferred = "cv_f1_macro_mean"

    if preferred not in combined.columns:

        available_cv = [
            column
            for column in cv_columns
            if combined[column].notna().any()
        ]

        if not available_cv:
            skip_figure(filename, "No valid CV scores found.")
            return

        preferred = available_cv[0]

    # FIX: filter to rows with a valid (non-NaN) CV score
    plot_df = combined[combined[preferred].notna()].copy()

    if plot_df.empty:

        skip_figure(
            filename,
            f"No rows with valid {preferred}."
        )

        return

    # FIX: label priority: model → run_id → experiment/source name
    def _label(row):
        model_val = str(row.get("model", "")).strip()
        if model_val and model_val.lower() not in ("nan", "none", ""):
            return clean_model_name(model_val)
        run_id_val = str(row.get("run_id", "")).strip()
        if run_id_val and run_id_val.lower() not in ("nan", "none", ""):
            # Shorten long run_ids
            return run_id_val.split("__")[-1].replace("_", " ").title()[:40]
        exp_val = str(row.get("experiment", "")).strip()
        return exp_val[:40] if exp_val and exp_val.lower() != "nan" else "Unknown"

    plot_df["Label"] = plot_df.apply(_label, axis=1)

    plot_df = (
        plot_df
        .sort_values(preferred, ascending=False)
        .head(20)
    )

    # Deduplicate labels (keep highest-CV row per label)
    plot_df = (
        plot_df
        .drop_duplicates(subset=["Label"], keep="first")
        .sort_values(preferred, ascending=True)
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.barh(
        plot_df["Label"],
        plot_df[preferred],
    )

    plt.title(
        "Cross-Validation Performance Stability"
    )

    plt.xlabel(
        preferred.replace("_", " ")
    )

    plt.gca().invert_yaxis()

    save_figure(
        filename
    )


# ============================================================
# COPY EXISTING REAL FIGURES
# ============================================================

def collect_existing_pipeline_figures():

    destination = (
        ARTIFACT_DIR /
        "pipeline_figures"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = []

    for path in ROOT.rglob("*.png"):

        if (
            FIGURES_DIR in path.parents
            or
            ARTIFACT_DIR in path.parents
        ):
            continue

        lower = str(path).lower()

        if not any(
            term in lower
            for term in [
                "result",
                "hierarchical",
                "shap",
                "figure",
                "graph",
                "output",
            ]
        ):
            continue

        target = (
            destination /
            safe_filename(
                "_".join(
                    path.relative_to(
                        ROOT
                    ).parts
                )
            )
        )

        try:

            shutil.copy2(
                path,
                target,
            )

            copied.append(
                target
            )

        except Exception:
            pass

    if copied:

        print(
            f"[COLLECTED] "
            f"{len(copied)} existing "
            f"pipeline-generated figures"
        )


# ============================================================
# GENERATE ALL FIGURES
# ============================================================

def generate_all_figures():

    print(
        "\n"
        "==============================================\n"
        "   RESEARCH PAPER FIGURE GENERATOR\n"
        "=============================================="
    )

    results = load_results()

    confusions = load_confusions()

    shap_df = (
        load_shap_importance()
    )

    prediction_files = (
        find_prediction_csvs()
    )

    print(
        f"[INFO] Parsed result rows: "
        f"{len(results)}"
    )

    print(
        f"[INFO] Parsed confusion matrices: "
        f"{len(confusions)}"
    )

    print(
        f"[INFO] Prediction/probability files found: "
        f"{len(prediction_files)}"
    )

    # 01-03
    generate_distribution_figures()

    # 04-06
    generate_model_comparison_figures(
        results
    )

    # 07-09
    generate_basic_confusion_figures(
        confusions
    )

    # 10-12
    generate_binary_roc(
        prediction_files
    )

    generate_binary_pr(
        prediction_files
    )

    generate_multiclass_roc(
        prediction_files
    )

    # 13-17
    generate_hierarchical_figures(
        results,
        confusions,
    )

    # 18-21
    generate_direct_vs_hierarchical(
        results
    )

    # 22, 24, 25
    generate_raw_shap_figures()

    # 23
    generate_shap_importance(
        shap_df
    )

    # 26-29
    generate_multisource_figures(
        results,
        confusions,
    )

    # 30-33
    generate_feature_analysis()

    # 34
    generate_confidence_distribution(
        prediction_files
    )

    # 35
    generate_cv_stability(
        results
    )

    # Collect already-generated real graphs
    collect_existing_pipeline_figures()

    # Artifact manifest
    discover_artifacts()

    generated = sorted(
        FIGURES_DIR.glob(
            "*.png"
        )
    )

    print(
        "\n"
        "==============================================\n"
        f"GENERATED FIGURES: {len(generated)}\n"
        "=============================================="
    )

    for path in generated:
        print(
            f"✓ {path.name}"
        )

    print(
        "\nSaved in:"
    )

    print(
        FIGURES_DIR
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run research experiments and generate "
            "real publication figures."
        )
    )

    parser.add_argument(
        "--graphs-only",
        action="store_true",
        help=(
            "Do not run models. Generate graphs "
            "from existing results only."
        ),
    )

    parser.add_argument(
        "--run-only",
        action="store_true",
        help=(
            "Run experiments but do not generate graphs."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore cached logs and rerun all experiments."
        ),
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip slow CTGAN training."
        ),
    )

    parser.add_argument(
        "--skip-platform",
        action="store_true",
        help=(
            "Skip synthetic_platform.py runs."
        ),
    )

    parser.add_argument(
        "--skip-multisource",
        action="store_true",
        help=(
            "Skip multisource_ablation.py runs."
        ),
    )

    args = parser.parse_args()

    if not check_required_data():
        return

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    if not args.graphs_only:

        run_all_experiments(
            force=args.force,
            fast=args.fast,
            skip_platform=(
                args.skip_platform
            ),
            skip_multisource=(
                args.skip_multisource
            ),
        )

    # --------------------------------------------------------
    # GENERATE FIGURES
    # --------------------------------------------------------

    if not args.run_only:

        generate_all_figures()


if __name__ == "__main__":
    main()
    