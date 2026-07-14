#!/usr/bin/env python3
"""Dataset Analysis - v1.0

Produces a terminal audit for CSV datasets found under a root folder.

Run:
    python dataset_analysis.py [root_folder]

This first version prints a readable audit (no files written).
"""

from pathlib import Path
import argparse
import sys
import math
import csv
from typing import List, Dict, Tuple, Optional

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("Missing dependencies. Please install requirements.txt and retry.")
    print(e)
    sys.exit(1)


BANNER = "=" * 60
SUB = "-" * 60

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('cp'):
    CHECK = '[OK]'
    WARN  = '[!]'
    CROSS = '[X]'
else:
    CHECK = '\u2713'
    WARN  = '\u26A0'
    CROSS = '\u274C'


def find_csvs(root: Path) -> List[Path]:
    csvs = []
    for p in root.rglob("*.csv"):
        if any(part.startswith('.') for part in p.parts):
            continue
        csvs.append(p)
    return sorted(csvs)


TARGET_COLUMNS = {
    'studentInfo.csv': 'final_result',
    'xAPI-Edu-Data.csv': 'Class',
    'student-mat.csv': 'G3',
    'student-por.csv': 'G3'
}

# Known ID columns per dataset (helps skip IDs for numeric checks)
KNOWN_ID_SUFFIX = ['id', 'id_student', 'id_assessment', 'id_site', 'stageid', 'gradeid', 'sectionid']

# Explicit leakage columns per dataset
LEAKAGE_COLUMNS = {
    'OULAD': ['date_unregistration', 'date_unreg', 'date_unregistered'],
    'UI_student+performance': [],
    'xAPI': []
}

# OULAD table relations for join checks
OULAD_RELATIONS = {
    'studentInfo.csv': {'pk': 'id_student'},
    'studentAssessment.csv': {'fk': 'id_student', 'pk': 'id_assessment'},
    'studentRegistration.csv': {'fk': 'id_student'},
    'studentVle.csv': {'fk': 'id_student'},
    'vle.csv': {'pk': 'id_site'},
    'assessments.csv': {'pk': 'id_assessment'},
    'courses.csv': {'pk': 'code_module'}
}


def sniff_delimiter(path: Path) -> str:
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            sample = ''.join([next(f) for _ in range(10)])
    except Exception:
        try:
            with path.open('r', encoding='latin1', errors='ignore') as f:
                sample = ''.join([next(f) for _ in range(10)])
        except Exception:
            return ','
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[',', ';', '\t', '|'])
        return dialect.delimiter
    except Exception:
        # fallback: semicolon common for UCI
        return ';' if ';' in sample and sample.count(';') > sample.count(',') else ','


def read_csv_auto(path: Path) -> pd.DataFrame:
    # First try pandas automatic separator detection
    try:
        return pd.read_csv(path, sep=None, engine='python')
    except Exception:
        pass
    # Sniff delimiter and try that
    delim = sniff_delimiter(path)
    try:
        return pd.read_csv(path, sep=delim, engine='python')
    except Exception:
        for sep in [';', ',', '\t', '|']:
            try:
                return pd.read_csv(path, sep=sep, engine='python')
            except Exception:
                continue
    # final fallback: let pandas try defaults
    return pd.read_csv(path)


def group_by_dataset(root: Path, paths: List[Path]) -> Dict[str, List[Path]]:
    groups = {}
    for p in paths:
        try:
            rel = p.relative_to(root)
        except Exception:
            rel = p
        parts = rel.parts
        if len(parts) >= 2:
            key = parts[0]
        else:
            key = str(rel.parent) if str(rel.parent) != '.' else 'root'
        groups.setdefault(key, []).append(p)
    return groups


def human_bytes(num_bytes: int) -> str:
    mb = num_bytes / (1024 * 1024)
    return f"{mb:.2f} MB"


def detect_target_column(df: pd.DataFrame, path: Path) -> Tuple[Optional[str], Optional[pd.Series]]:
    # Heuristics for target columns
    candidates = [
        'final_result', 'finalgrade', 'final_grade', 'grade', 'G3', 'GPA', 'passed',
        'result', 'pass', 'outcome'
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    # First check explicit mapping
    fname = path.name
    if fname in TARGET_COLUMNS:
        colname = TARGET_COLUMNS[fname]
        if colname in df.columns:
            return colname, df[colname]
    for cand in candidates:
        if cand in lower_cols:
            col = lower_cols[cand]
            return col, df[col]

    # Otherwise, choose a low-cardinality non-id column with <=20 unique values
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 20 and df[c].dtype == object:
            return c, df[c]

    return None, None


def is_id_column(col: str) -> bool:
    low = col.lower()
    exact_ids = {'id', 'id_student', 'id_assessment', 'id_site', 'stageid', 'gradeid', 'sectionid'}
    if low in exact_ids:
        return True
    # common pattern: ends with _id or starts with id_
    if low.endswith('_id') or low.startswith('id_') or '_id_' in low:
        return True
    return False


def summarize_dataframe(path: Path) -> Dict:
    info = {'path': path}
    try:
        df = read_csv_auto(path)
    except Exception as e:
        info['error'] = f"Failed to read CSV: {e}"
        return info

    info['rows'], info['cols'] = df.shape
    info['memory'] = df.memory_usage(deep=True).sum()
    dtypes = df.dtypes.apply(lambda x: x.name).value_counts().to_dict()
    info['dtypes'] = dtypes

    # Target detection
    target_col, target_series = detect_target_column(df, path)
    info['target'] = target_col
    if target_col is not None:
        info['class_distribution'] = target_series.value_counts(dropna=False).to_dict()
    else:
        info['class_distribution'] = None

    # Missing values
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100
    miss = pd.concat([missing_counts, missing_percent], axis=1)
    miss.columns = ['count', 'percent']
    miss = miss.sort_values('count', ascending=False)
    info['missing'] = miss

    # Duplicates
    info['duplicate_rows'] = int(df.duplicated().sum())

    # Candidate ID columns
    id_candidates = [c for c in df.columns if is_id_column(c) or c.lower().endswith('code')]
    dup_ids = {}
    for c in id_candidates:
        nun = df[c].nunique(dropna=True)
        dup = len(df) - nun
        unique_ratio = nun / max(1, len(df))
        expected_unique = unique_ratio >= 0.99
        if expected_unique and dup > 0:
            dup_ids[c] = {'duplicates': int(dup), 'expected_unique': True}
        else:
            # provide info but don't flag as issue
            if dup > 0:
                dup_ids[c] = {'duplicates': int(dup), 'expected_unique': False}
    info['duplicate_ids'] = dup_ids

    # Constant cols
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    info['constant_columns'] = const_cols

    # High cardinality (only for categorical columns)
    high_card = []
    for c in df.select_dtypes(include=['object', 'category']).columns:
        nun = df[c].nunique(dropna=True)
        if nun > 50 or nun / max(1, len(df)) > 0.5:
            high_card.append(c)
    info['high_cardinality'] = high_card

    # Potential leakage: column names that look like dates or that contain 'unregister' or 'final' but are not target
    leakage = []
    fname = path.name
    grp = path.parts[0] if len(path.parts) > 1 else None
    explicit_leak = []
    if grp and grp in LEAKAGE_COLUMNS:
        explicit_leak = LEAKAGE_COLUMNS.get(grp, [])
    for c in df.columns:
        low = c.lower()
        if low == (target_col or '').lower():
            continue
        # explicit dataset-level leakage
        if c in explicit_leak:
            leakage.append(c)
            continue
        # mark only strong leakage candidates
        if any(k in low for k in ['unreg', 'unregistration', 'withdraw', 'withdrawal']):
            leakage.append(c)
            continue
        # final_result, final grade, or result-like columns are potential leakage
        if low in ['final_result', 'finalgrade', 'final_grade', 'result']:
            leakage.append(c)
            continue
    info['leakage'] = leakage

    # Categorical vs numerical
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    info['categorical_columns'] = cat_cols
    info['numerical_columns'] = num_cols

    # Outliers (IQR method)
    # Outliers (IQR) - skip id columns
    outliers = {}
    for c in num_cols:
        if is_id_column(c):
            continue
        series = df[c].dropna()
        if len(series) < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        if count > 0:
            outliers[c] = count
    info['outliers'] = outliers

    # Correlations
    corr_info = None
    # exclude id-like numeric columns from correlation
    numeric_for_corr = [c for c in num_cols if not is_id_column(c)]
    if len(numeric_for_corr) >= 2:
        corr = df[numeric_for_corr].corr().abs()
        corr_vals = corr.unstack()
        corr_vals = corr_vals[corr_vals.index.get_level_values(0) != corr_vals.index.get_level_values(1)]
        if not corr_vals.empty:
            top_idx = corr_vals.sort_values(ascending=False).dropna().index
            if len(top_idx) > 0:
                top_val = float(corr_vals.loc[top_idx[0]])
                corr_info = (top_idx[0][0], top_idx[0][1], top_val)
    info['top_correlation'] = corr_info

    # Class imbalance heuristic
    imbalance = None
    if info['class_distribution']:
        total = sum(info['class_distribution'].values())
        top = max(info['class_distribution'].values())
        prop = top / max(1, total)
        if prop > 0.6:
            imbalance = 'Yes'
        elif prop > 0.45:
            imbalance = 'Moderate'
        else:
            imbalance = 'No'
    info['imbalance'] = imbalance

    # Additional checks: data type suggestions, category cleaning, impossible values, skewness, near-zero variance
    suggestions = []
    # Date parse suggestions
    date_suggestions = []
    for c in df.columns:
        if 'date' in c.lower() and df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c], errors='coerce')
                nonnull = parsed.notnull().sum()
                if nonnull / max(1, len(df)) > 0.5:
                    date_suggestions.append(c)
            except Exception:
                pass
    if date_suggestions:
        suggestions.append({'dates_parse': date_suggestions})

    # Category cleaning (case variants)
    case_fix = {}
    for c in cat_cols:
        vals = df[c].dropna().astype(str)
        uniq = vals.nunique()
        uniq_lower = vals.str.lower().nunique()
        if uniq_lower < uniq:
            case_fix[c] = {'orig_unique': int(uniq), 'lower_unique': int(uniq_lower)}
    if case_fix:
        suggestions.append({'case_normalization': case_fix})

    # Impossible values heuristic
    impossible = {}
    for c in num_cols:
        low = c.lower()
        s = df[c].dropna()
        if s.empty:
            continue
        minv = s.min()
        maxv = s.max()
        issues = []
        if 'age' in low and (minv < 0 or maxv > 120):
            issues.append((minv, maxv))
        if any(k in low for k in ['percent', 'attendance']) and maxv > 100:
            issues.append((minv, maxv))
        if 'score' in low and maxv > 100:
            issues.append((minv, maxv))
        if issues:
            impossible[c] = issues
    if impossible:
        suggestions.append({'impossible_values': impossible})

    # Skewness
    skewed = {}
    for c in num_cols:
        if is_id_column(c):
            continue
        s = df[c].dropna()
        if len(s) < 10:
            continue
        sk = float(s.skew())
        if abs(sk) > 1:
            skewed[c] = sk
    if skewed:
        suggestions.append({'skewed': skewed})

    # Near-zero variance
    nzv = []
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if nun == 1:
            nzv.append(c)
        else:
            vc = df[c].value_counts(normalize=True, dropna=True)
            if vc.empty:
                continue
            top_freq = vc.iloc[0]
            if top_freq > 0.98:
                nzv.append(c)
    if nzv:
        suggestions.append({'near_zero_variance': nzv})

    # Memory downcast suggestions for numeric columns
    downcast = {}
    for c in num_cols:
        if is_id_column(c):
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        minv = s.min()
        maxv = s.max()
        if pd.api.types.is_integer_dtype(s):
            if minv >= 0 and maxv <= 255:
                downcast[c] = 'uint8'
            elif minv >= -32768 and maxv <= 32767:
                downcast[c] = 'int16'
            elif minv >= -2147483648 and maxv <= 2147483647:
                downcast[c] = 'int32'
        elif pd.api.types.is_float_dtype(s):
            downcast[c] = 'float32'
    if downcast:
        suggestions.append({'downcast_suggestions': downcast})

    info['suggestions'] = suggestions

    # Training readiness score (simple heuristic)
    score = 100
    total_missing = int(info['missing']['count'].sum())
    pct_missing = (total_missing / max(1, len(df) * len(df.columns))) * 100
    score -= min(50, pct_missing * 0.5)
    score -= min(30, len(info['leakage']) * 10)
    score -= 5 if info['duplicate_rows'] > 0 else 0
    score -= 5 if info['constant_columns'] else 0
    score = max(0, int(score))
    info['readiness'] = score

    info['num_rows'] = len(df)

    return info


def print_dataset_report(name: str, files: List[Path], infos: List[Dict]):
    print(BANNER)
    print(f"DATASET : {name}")
    print(BANNER)
    print()
    for info in infos:
        path = info.get('path')
        print(SUB)
        print(path.name)
        print(SUB)
        if 'error' in info:
            print(f"Error reading file: {info['error']}")
            continue

        print('\nShape')
        print('------')
        print(f"Rows : {info['rows']}")
        print(f"Columns : {info['cols']}")
        print()

        print('Memory Usage')
        print('-------------')
        print(human_bytes(info['memory']))
        print()

        print('Data Types')
        print('----------')
        for k, v in info['dtypes'].items():
            print(f"{k.ljust(8)}: {v}")
        print()

        print('Target Column')
        print('-------------')
        if info['target']:
            print(info['target'])
        else:
            print('None detected')
        print()

        if info['class_distribution']:
            print('Class Distribution')
            print('------------------')
            for k, v in info['class_distribution'].items():
                print(f"{str(k).ljust(12)}: {v}")
            print()

        print('Missing Values')
        print('--------------')
        miss = info['missing']
        for c, row in miss.head(10).iterrows():
            print(f"{c.ljust(18)} : {int(row['count'])} ({row['percent']:.1f}%)")
        if miss['count'].sum() == 0:
            print('None')
        print()

        print('Duplicate Rows')
        print('--------------')
        print(info['duplicate_rows'])
        if info['duplicate_rows'] > 0:
            print(f'\n{WARN} Recommendation: Remove duplicate rows because they can bias the model.')
        print()

        print('Duplicate IDs')
        print('-------------')
        if info['duplicate_ids']:
            for k, v in info['duplicate_ids'].items():
                if isinstance(v, dict):
                    mark = ' (EXPECTED UNIQUE)' if v.get('expected_unique') else ''
                    print(f"{k} : {v.get('duplicates')}{mark}")
                else:
                    print(f"{k} : {v}")
            print(f'\n{WARN} Recommendation: Investigate identifier collisions when the ID is expected to be unique.')
        else:
            print('0')
        print()

        print('Constant Columns')
        print('----------------')
        if info['constant_columns']:
            for c in info['constant_columns']:
                print(c)
        else:
            print('None')
        print()

        print('High Cardinality Columns')
        print('------------------------')
        if info['high_cardinality']:
            for c in info['high_cardinality']:
                print(c)
            print(f'\n{WARN} Recommendation: Consider hashing or target-encoding high-cardinality features.')
        else:
            print('None')
        print()

        print('Potential Leakage Columns')
        print('-------------------------')
        if info['leakage']:
            for c in info['leakage']:
                print(c)
            print('\nReason: Strong indicators (unregistration/withdrawal/final results) that may reveal labels.')
            print(f'\n{CROSS} Critical Issue: Remove these columns before training or ensure temporal cutoffs.')
        else:
            print('None (generic date columns treated conservatively)')
        print()

        print('Categorical Columns')
        print('-------------------')
        for c in info['categorical_columns'][:10]:
            print(c)
        if len(info['categorical_columns']) == 0:
            print('None')
        print()

        print('Numerical Columns')
        print('-----------------')
        for c in info['numerical_columns'][:10]:
            print(c)
        if len(info['numerical_columns']) == 0:
            print('None')
        print()

        print('Outliers')
        print('---------')
        if info['outliers']:
            for c, v in info['outliers'].items():
                print(f"{c} : {v}")
        else:
            print('None')
        print()

        print('Feature Correlation')
        print('-------------------')
        if info['top_correlation']:
            a, b, val = info['top_correlation']
            print(f"{a} 	26 {b} = {val:.2f}")
        else:
            print('Insufficient numeric features')
        print()

        print('Recommendation')
        print('--------------')
        recs = []
        if info['leakage']:
            recs.append(f'{CHECK} Remove leakage columns')
        if info['missing']['count'].sum() > 0:
            recs.append(f'{CHECK} Impute or remove missing values')
        if info['categorical_columns']:
            recs.append(f'{CHECK} Encode categorical features (One-Hot / Target encoding)')
        if info['numerical_columns']:
            recs.append(f'{CHECK} Scaling may not be needed for tree models')
        if not recs:
            print(f'{CHECK} No immediate actions required')
        else:
            for r in recs:
                print(r)
        # Suggestions
        if info.get('suggestions'):
            print('\nAdditional Checks')
            for s in info['suggestions']:
                for k, v in s.items():
                    print(f"- {k}: {v}")
        print('\n')


def overall_summary(group_infos: Dict[str, List[Dict]]):
    print(BANNER)
    print('OVERALL PROJECT SUMMARY')
    print(BANNER)
    print()
    datasets = len(group_infos)
    csv_files = sum(len(v) for v in group_infos.values())
    total_records = sum(info.get('num_rows', 0) for infos in group_infos.values() for info in infos)
    print(f"Datasets          : {datasets}")
    print(f"CSV Files         : {csv_files}")
    print(f"Total Records     : {total_records}")
    print()

    print('Data Leakage')
    print('------------')
    for name, infos in group_infos.items():
        leak = any(info.get('leakage') for info in infos)
        print(f"{name.ljust(6)} : {'YES' if leak else 'NO'}")
    print()

    print('Missing Values')
    print('--------------')
    for name, infos in group_infos.items():
        total_missing = sum(int(info.get('missing')['count'].sum()) for info in infos if 'missing' in info)
        total_cells = sum(info.get('num_rows',0) * info.get('cols',0) for info in infos)
        pct = (total_missing / max(1, total_cells)) * 100
        severity = 'None' if pct < 1 else ('Low' if pct < 5 else ('Moderate' if pct < 20 else 'High'))
        print(f"{name.ljust(6)} : {severity}")
    print()

    print('Class Imbalance')
    print('---------------')
    for name, infos in group_infos.items():
        imbs = [info.get('imbalance') for info in infos if info.get('imbalance') is not None]
        if any(i == 'Yes' for i in imbs):
            s = 'Yes'
        elif any(i == 'Moderate' for i in imbs):
            s = 'Moderate'
        elif any(i == 'No' for i in imbs):
            s = 'No'
        else:
            s = 'Unknown'
        print(f"{name.ljust(6)} : {s}")
    print()

    print('Recommended Models')
    print('------------------')
    print('1. CatBoost')
    print('2. XGBoost')
    print('3. LightGBM')
    print()

    print('Training Readiness')
    print('------------------')
    for name, infos in group_infos.items():
        scores = [info.get('readiness', 0) for info in infos]
        score = int(sum(scores) / max(1, len(scores)))
        print(f"{name.ljust(6)} : {score}%")
    print()

    print('Next Steps')
    print('----------')
    print('1. Clean missing values')
    print('2. Remove leakage columns')
    print('3. Encode categorical features')
    print('4. Handle imbalance')
    print('5. Train baseline models')
    print()


def perform_oulad_join_checks(root: Path, paths: List[Path]) -> Dict[str, Dict]:
    # For OULAD, check id_student presence across tables
    checks = {}
    student_info = None
    # find the studentInfo file
    for p in paths:
        if p.name == 'studentInfo.csv':
            student_info = p
            break
    if not student_info:
        return checks

    try:
        info_ids = pd.read_csv(student_info, usecols=['id_student'])['id_student'].dropna().unique()
        info_set = set(info_ids)
    except Exception:
        return checks

    for p in paths:
        if p.name == 'studentInfo.csv':
            continue
        try:
            if 'id_student' in pd.read_csv(p, nrows=0).columns:
                ids = pd.read_csv(p, usecols=['id_student'])['id_student'].dropna().unique()
                ids_set = set(ids)
                missing_in_info = ids_set - info_set
                checks[p.name] = {
                    'unique_ids': int(len(ids_set)),
                    'missing_in_studentInfo': int(len(missing_in_info)),
                    'pct_missing': float(len(missing_in_info) / max(1, len(ids_set)))
                }
        except Exception:
            continue
    return checks


def main():
    parser = argparse.ArgumentParser(
        description='Scans a project folder for CSV datasets and prints a full audit report.')
    parser.add_argument('root', nargs='?', default='.', help='Root folder to scan')
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root path does not exist: {root}")
        return

    print(BANNER)
    print('        STUDENT DATASET ANALYZER v1.0')
    print(BANNER)
    print()

    csvs = find_csvs(root)
    groups = group_by_dataset(root, csvs)
    if not groups:
        print('No CSV files found.')
        return

    print('Scanning directories...\n')
    for name in sorted(groups.keys()):
        print(f"[{CHECK}] {name}")
    print()

    group_infos = {}
    for name, paths in sorted(groups.items(), key=lambda x: x[0]):
        infos = []
        for p in paths:
            infos.append(summarize_dataframe(p))
        group_infos[name] = infos
        print_dataset_report(name, paths, infos)

    # OULAD join integrity checks
    if 'OULAD' in groups:
        try:
            checks = perform_oulad_join_checks(root, groups['OULAD'])
            if checks:
                print(BANNER)
                print('OULAD JOIN CHECKS')
                print(BANNER)
                for fname, c in checks.items():
                    print(f"{fname.ljust(24)} - unique_ids: {c['unique_ids']}  missing_in_studentInfo: {c['missing_in_studentInfo']} ({c['pct_missing']:.2%})")
                print()
        except Exception:
            pass

    overall_summary(group_infos)


if __name__ == '__main__':
    main()
