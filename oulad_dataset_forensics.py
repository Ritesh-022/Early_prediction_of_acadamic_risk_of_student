#!/usr/bin/env python3
"""OULAD Dataset Forensics

This script performs a deeper dataset audit for OULAD and the generated ML table.
It reports dataset health, target behavior, feature quality, leakage risk, and temporal activity.
"""
from pathlib import Path
import argparse
import csv
import sys
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder


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
        return ';' if ';' in sample and sample.count(';') > sample.count(',') else ','


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine='python', **kwargs)
    except Exception:
        pass
    delim = sniff_delimiter(path)
    try:
        return pd.read_csv(path, sep=delim, engine='python', **kwargs)
    except Exception:
        for sep in [',', ';', '\t', '|']:
            try:
                return pd.read_csv(path, sep=sep, engine='python', **kwargs)
            except Exception:
                continue
    return pd.read_csv(path, **kwargs)


def human_bytes(num_bytes: int) -> str:
    mb = num_bytes / (1024 * 1024)
    return f"{mb:.2f} MB"


def is_id_column(name: str) -> bool:
    low = name.lower()
    if low in {'id', 'id_student', 'id_assessment', 'id_site', 'stageid', 'gradeid', 'sectionid'}:
        return True
    return low.endswith('_id') or low.startswith('id_') or '_id_' in low


def detect_target_column(df: pd.DataFrame, explicit: Optional[str] = None) -> Tuple[Optional[str], Optional[pd.Series]]:
    if explicit and explicit in df.columns:
        return explicit, df[explicit]
    candidates = [
        'final_result', 'finalgrade', 'final_grade', 'grade', 'G3', 'GPA', 'passed',
        'result', 'pass', 'outcome'
    ]
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand], df[cols[cand]]
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique(dropna=True) <= 20:
            return c, df[c]
    return None, None


def summarize_health(df: pd.DataFrame, name: str) -> Dict[str, object]:
    info: Dict[str, object] = {}
    info['name'] = name
    info['rows'] = len(df)
    info['columns'] = len(df.columns)
    info['memory_MB'] = df.memory_usage(deep=True).sum() / (1024 * 1024)

    info['dtypes'] = df.dtypes.apply(lambda x: x.name).value_counts().to_dict()
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / max(1, len(df))) * 100
    info['missing'] = pd.DataFrame({'count': missing, 'percent': missing_pct})
    info['duplicate_rows'] = int(df.duplicated().sum())

    id_cols = [c for c in df.columns if is_id_column(c)]
    id_summary = {}
    for col in id_cols:
        unique = df[col].nunique(dropna=True)
        ids = len(df) - unique
        id_summary[col] = {
            'duplicates': int(ids),
            'unique_ratio': float(unique / max(1, len(df))),
            'expected_unique': unique >= len(df) * 0.99
        }
    info['id_columns'] = id_summary

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    info['constant_columns'] = constant_cols

    low_variance = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].nunique(dropna=True) <= 1:
            continue
        if df[col].var(skipna=True) < 1e-4:
            low_variance.append(col)
    info['low_variance_columns'] = low_variance

    impossible_values = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        if series.min() < 0 and any(k in col.lower() for k in ['count', 'num', 'total', 'activity', 'click', 'days', 'weeks', 'score', 'weight']):
            impossible_values[col] = {'min': float(series.min()), 'max': float(series.max())}
    info['impossible_values'] = impossible_values

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    high_cardinality = [c for c in categorical_cols if df[c].nunique(dropna=True) > 50 or df[c].nunique(dropna=True) / max(1, len(df)) > 0.5]
    info['high_cardinality'] = high_cardinality

    return info


def print_health_summary(info: Dict[str, object]) -> List[str]:
    lines = [f"Dataset: {info['name']}"]
    lines.append(f"Rows: {info['rows']}")
    lines.append(f"Columns: {info['columns']}")
    lines.append(f"Memory: {info['memory_MB']:.2f} MB")
    lines.append('')
    lines.append('Data types:')
    for dtype, count in info['dtypes'].items():
        lines.append(f"  {dtype}: {count}")
    lines.append('')
    lines.append(f"Duplicate rows: {info['duplicate_rows']}")
    lines.append(f"Constant columns: {len(info['constant_columns'])}")
    if info['constant_columns']:
        lines.append(f"  {', '.join(info['constant_columns'])}")
    lines.append(f"Low variance columns: {len(info['low_variance_columns'])}")
    if info['low_variance_columns']:
        lines.append(f"  {', '.join(info['low_variance_columns'])}")
    if info['impossible_values']:
        lines.append('Impossible values detected:')
        for col, stats in info['impossible_values'].items():
            lines.append(f"  {col}: min={stats['min']}, max={stats['max']}")
    if info['id_columns']:
        lines.append('ID-like columns:')
        for col, meta in info['id_columns'].items():
            lines.append(f"  {col}: duplicates={meta['duplicates']}, unique_ratio={meta['unique_ratio']:.3f}, expected_unique={meta['expected_unique']}")
    if info['high_cardinality']:
        lines.append('High-cardinality categorical columns:')
        lines.append(f"  {', '.join(info['high_cardinality'])}")
    lines.append('')
    return lines


def target_distribution(df: pd.DataFrame, target_col: str, group_col: Optional[str] = None) -> pd.DataFrame:
    if group_col and group_col in df.columns:
        grouped = df.groupby(group_col)[target_col].value_counts(normalize=True)
        result = grouped.rename('proportion').reset_index()
    else:
        result = df[target_col].value_counts(normalize=True).rename('proportion').reset_index()
        result.columns = [target_col, 'proportion']
    return result


def build_target_report(df: pd.DataFrame, target_col: str, module_col: str = 'code_module', presentation_col: str = 'code_presentation') -> Dict[str, pd.DataFrame]:
    report = {}
    report['overall'] = target_distribution(df, target_col)
    if module_col in df.columns:
        report['per_module'] = target_distribution(df, target_col, module_col)
    if presentation_col in df.columns:
        report['per_presentation'] = target_distribution(df, target_col, presentation_col)
    if module_col in df.columns and presentation_col in df.columns:
        combined = df.groupby([module_col, presentation_col])[target_col].value_counts(normalize=True).rename('proportion').reset_index()
        report['module_presentation'] = combined
    return report


def label_encode_series(series: pd.Series) -> pd.Series:
    encoder = LabelEncoder()
    return encoder.fit_transform(series.astype(str).fillna('missing'))


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    categories = categories.astype(str).fillna('missing')
    measurements = measurements.dropna()
    if measurements.empty:
        return 0.0
    cat_labels = categories.loc[measurements.index]
    category_groups = measurements.groupby(cat_labels)
    mean_total = measurements.mean()
    ss_between = sum(len(group) * (group.mean() - mean_total) ** 2 for _, group in category_groups)
    ss_total = ((measurements - mean_total) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    contingency = pd.crosstab(x.fillna('missing').astype(str), y.fillna('missing').astype(str))
    if contingency.size == 0:
        return 0.0
    chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    n = contingency.sum().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if rcorr <= 1 or kcorr <= 1 or denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def feature_recommendation(mutual_info: float) -> str:
    if mutual_info >= 0.20:
        return 'KEEP'
    if mutual_info >= 0.05:
        return 'OPTIONAL'
    return 'DROP'


def feature_quality(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    clean = df.copy()
    clean = clean.drop(columns=[target_col], errors='ignore')
    y = label_encode_series(df[target_col])
    numeric_cols = clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = clean.select_dtypes(include=['object', 'category']).columns.tolist()
    quality = []

    if not clean.columns.any():
        return pd.DataFrame(columns=['feature', 'type', 'mutual_info', 'f_score', 'f_pvalue', 'cramers_v', 'correlation_ratio', 'recommendation'])

    X = pd.DataFrame()
    discrete = []
    for col in clean.columns:
        if col in categorical_cols:
            X[col] = label_encode_series(clean[col])
            discrete.append(True)
        else:
            X[col] = clean[col].fillna(0)
            discrete.append(False)

    try:
        mi = mutual_info_classif(X.fillna(0), y, discrete_features=discrete, random_state=42)
    except Exception:
        mi = np.zeros(len(X.columns), dtype=float)

    f_scores = np.full(len(X.columns), np.nan, dtype=float)
    p_values = np.full(len(X.columns), np.nan, dtype=float)
    numeric_cols_for_test = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
    if numeric_cols_for_test:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                f_res = f_classif(X[numeric_cols_for_test].fillna(0), y)
            for idx, col in enumerate(numeric_cols_for_test):
                loc = list(X.columns).index(col)
                f_scores[loc] = float(f_res[0][idx])
                p_values[loc] = float(f_res[1][idx])
        except Exception:
            pass

    for i, col in enumerate(X.columns):
        col_type = 'categorical' if col in categorical_cols else 'numeric'
        cram = cramers_v(df[col], df[target_col]) if col in categorical_cols else np.nan
        corr_ratio = correlation_ratio(df[target_col], df[col]) if col in numeric_cols else np.nan
        quality.append({
            'feature': col,
            'type': col_type,
            'mutual_info': float(mi[i]),
            'f_score': float(f_scores[i]) if not np.isnan(f_scores[i]) else np.nan,
            'f_pvalue': float(p_values[i]) if not np.isnan(p_values[i]) else np.nan,
            'cramers_v': float(cram) if not np.isnan(cram) else np.nan,
            'correlation_ratio': float(corr_ratio) if not np.isnan(corr_ratio) else np.nan,
            'recommendation': feature_recommendation(float(mi[i]))
        })

    quality_df = pd.DataFrame(quality)
    quality_df = quality_df.sort_values(by='mutual_info', ascending=False)
    return quality_df


SAFE_FEATURES = {
    'gender', 'region', 'highest_education', 'imd_band', 'age_band',
    'num_of_prev_attempts', 'studied_credits', 'disability', 'registration_early_days'
}
RISKY_FEATURES = {
    'total_clicks', 'activity_count', 'num_assessments', 'avg_score',
    'weighted_score', 'days_active', 'active_weeks', 'avg_clicks_per_day',
    'inactivity_days', 'registration_early_days', 'clicks_per_active_week',
    'clicks_per_credit', 'assessments_per_week'
}
LEAKAGE_FEATURES = {
    'date_unregistration', 'date_unreg', 'date_unregistered',
    'final_result', 'finalgrade', 'final_grade', 'result', 'pass', 'fail',
    'distinction', 'outcome'
}


def classify_leakage(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    categories = []
    for col in df.columns:
        if col == target_col:
            categories.append('Target')
            continue
        low = col.lower()
        status = 'Safe'
        if col in SAFE_FEATURES or low in SAFE_FEATURES:
            status = 'Safe'
        elif col in LEAKAGE_FEATURES or low in LEAKAGE_FEATURES:
            status = 'Leakage'
        elif col in RISKY_FEATURES or low in RISKY_FEATURES:
            status = 'Risky'
        elif any(tok in low for tok in ['unreg', 'withdraw', 'late', 'final', 'grade', 'result', 'outcome']):
            status = 'Leakage'
        elif any(tok in low for tok in ['date', 'registration', 'last', 'first', 'week', 'total', 'num', 'count', 'avg', 'weighted', 'score', 'clicks']):
            status = 'Risky'
        categories.append(status)
    return pd.DataFrame({'feature': df.columns, 'leakage_status': categories})


def feature_redundancy(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    rows = []
    for a, b in combinations(numeric_cols, 2):
        if df[a].isna().all() or df[b].isna().all():
            continue
        corr = df[a].corr(df[b])
        if corr is not None and abs(corr) >= 0.85:
            rows.append({'feature_a': a, 'feature_b': b, 'metric': 'pearson', 'value': abs(float(corr))})
    for a, b in combinations(categorical_cols, 2):
        if df[a].isna().all() or df[b].isna().all():
            continue
        value = cramers_v(df[a], df[b])
        if value >= 0.6:
            rows.append({'feature_a': a, 'feature_b': b, 'metric': "cramers_v", 'value': float(value)})
    return pd.DataFrame(rows).sort_values(by='value', ascending=False)


def feature_quality_by_group(df: pd.DataFrame, target_col: str, group_col: str = 'code_module', min_rows: int = 100) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame(columns=['group', 'feature', 'type', 'mutual_info', 'f_score', 'f_pvalue', 'cramers_v', 'correlation_ratio', 'recommendation'])
    rows = []
    for group_value, subset in df.groupby(group_col):
        if len(subset) < min_rows:
            continue
        quality = feature_quality(subset, target_col)
        quality['group'] = group_value
        rows.append(quality)
    if not rows:
        return pd.DataFrame(columns=['group', 'feature', 'type', 'mutual_info', 'f_score', 'f_pvalue', 'cramers_v', 'correlation_ratio', 'recommendation'])
    result = pd.concat(rows, ignore_index=True)
    return result


def target_by_assessment_type(root: Path, raw_files: Dict[str, Path], target_col: str) -> pd.DataFrame:
    if 'studentAssessment.csv' not in raw_files or 'assessments.csv' not in raw_files or 'studentInfo.csv' not in raw_files:
        return pd.DataFrame()
    sa = read_csv_auto(raw_files['studentAssessment.csv'])
    ass = read_csv_auto(raw_files['assessments.csv'])
    student_info = read_csv_auto(raw_files['studentInfo.csv'])
    sa.columns = [c.strip() for c in sa.columns]
    ass.columns = [c.strip() for c in ass.columns]
    student_info.columns = [c.strip() for c in student_info.columns]
    if target_col not in student_info.columns:
        return pd.DataFrame()
    if 'id_assessment' not in sa.columns or 'id_assessment' not in ass.columns:
        return pd.DataFrame()
    merged = sa.merge(ass[['id_assessment', 'assessment_type']], on='id_assessment', how='left')
    merged = merged.merge(student_info[['id_student', 'code_module', 'code_presentation', target_col]], on='id_student', how='left')
    if 'assessment_type' not in merged.columns:
        return pd.DataFrame()
    result = merged.groupby(['assessment_type', target_col]).size().rename('count').reset_index()
    result['proportion'] = result.groupby('assessment_type')['count'].transform(lambda x: x / x.sum())
    return result


def find_oulad_files(root: Path) -> Dict[str, Path]:
    names = [
        'studentInfo.csv', 'studentVle.csv', 'studentAssessment.csv', 'assessments.csv',
        'vle.csv', 'studentRegistration.csv', 'courses.csv'
    ]
    found: Dict[str, Path] = {}
    for p in root.rglob('*.csv'):
        if p.name in names:
            found[p.name] = p
    return found


def temporal_activity_analysis(root: Path, output_dir: Path) -> List[str]:
    lines: List[str] = []
    files = find_oulad_files(root)
    if 'studentVle.csv' not in files or 'studentInfo.csv' not in files:
        lines.append('Temporal analysis skipped: raw studentVle.csv or studentInfo.csv not found.')
        return lines

    student_info = read_csv_auto(files['studentInfo.csv'])
    student_info.columns = [c.strip() for c in student_info.columns]
    if 'id_student' not in student_info.columns or 'final_result' not in student_info.columns:
        lines.append('Temporal analysis skipped: studentInfo missing id_student or final_result.')
        return lines

    vle = read_csv_auto(files['studentVle.csv'])
    vle.columns = [c.strip() for c in vle.columns]
    click_col = next((c for c in vle.columns if c.lower() in {'sum_click', 'sum_clicks', 'click', 'clicks', 'sum_click_count'}), None)
    week_col = next((c for c in vle.columns if c.lower() in {'week_from', 'week', 'weeknumber'}), None)
    if click_col is None or week_col is None:
        lines.append('Temporal analysis skipped: cannot detect click or week column in studentVle.')
        return lines

    vle = vle.rename(columns={click_col: 'clicks', week_col: 'week'})
    if 'code_module' not in vle.columns or 'code_presentation' not in vle.columns or 'id_student' not in vle.columns:
        lines.append('Temporal analysis skipped: studentVle missing required key columns.')
        return lines

    merged = vle.merge(student_info[['id_student', 'code_module', 'code_presentation', 'final_result']], on=['id_student', 'code_module', 'code_presentation'], how='left')
    merged['week'] = pd.to_numeric(merged['week'], errors='coerce').fillna(-1).astype(int)
    merged['clicks'] = pd.to_numeric(merged['clicks'], errors='coerce').fillna(0)

    weekly = merged.groupby(['week', 'final_result']).agg(
        student_count=('id_student', 'nunique'),
        total_clicks=('clicks', 'sum'),
        avg_clicks=('clicks', 'mean')
    ).reset_index().sort_values(['week', 'final_result'])
    weekly.to_csv(output_dir / 'temporal_clicks_by_week.csv', index=False)
    lines.append(f'Wrote temporal_clicks_by_week.csv ({len(weekly)} rows)')

    weekly_overall = merged.groupby('week').agg(
        student_count=('id_student', 'nunique'),
        total_clicks=('clicks', 'sum'),
        avg_clicks=('clicks', 'mean')
    ).reset_index().sort_values('week')
    weekly_overall.to_csv(output_dir / 'temporal_clicks_overall_by_week.csv', index=False)
    lines.append(f'Wrote temporal_clicks_overall_by_week.csv ({len(weekly_overall)} rows)')

    if 'activity_type' in merged.columns:
        activity_by_week = merged.groupby(['week', 'activity_type']).size().rename('events').reset_index()
        activity_by_week.to_csv(output_dir / 'temporal_activity_events_by_week.csv', index=False)
        lines.append(f'Wrote temporal_activity_events_by_week.csv ({len(activity_by_week)} rows)')

    return lines


def render_dataframe(df: pd.DataFrame, max_rows: int = 20) -> List[str]:
    lines = []
    if df.empty:
        lines.append('  <empty>')
        return lines
    preview = df.head(max_rows)
    lines.extend(preview.to_string(index=False).splitlines())
    if len(df) > max_rows:
        lines.append(f'  ... ({len(df)} rows total)')
    return lines


def plot_missing_values(missing_report: pd.DataFrame, output_dir: Path) -> None:
    if missing_report.empty:
        return
    plt.figure(figsize=(10, 6))
    missing_report = missing_report.sort_values('percent', ascending=False)
    plt.barh(missing_report['feature'].astype(str), missing_report['percent'], color='#4c72b0')
    plt.xlabel('Missing values (%)')
    plt.title('Missing Values by Feature')
    plt.tight_layout()
    plt.savefig(output_dir / 'missing_values.png', dpi=150)
    plt.close()


def plot_target_distribution(target_report: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    if 'overall' not in target_report or target_report['overall'].empty:
        return
    overall = target_report['overall']
    plt.figure(figsize=(8, 5))
    plt.bar(overall.iloc[:, 0].astype(str), overall['proportion'], color='#55a868')
    plt.xlabel(overall.columns[0])
    plt.ylabel('Proportion')
    plt.title('Target Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'target_distribution.png', dpi=150)
    plt.close()


def plot_feature_quality(quality_df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    if quality_df.empty:
        return
    plot_df = quality_df.sort_values('mutual_info', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df['feature'].astype(str), plot_df['mutual_info'], color='#c44e52')
    plt.gca().invert_yaxis()
    plt.xlabel('Mutual information')
    plt.title('Top Features by Mutual Information')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_feature_quality.png', dpi=150)
    plt.close()


def plot_temporal_clicks(output_dir: Path) -> None:
    path = output_dir / 'temporal_clicks_by_week.csv'
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or 'week' not in df.columns or 'total_clicks' not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    for label, subset in df.groupby('final_result'):
        plt.plot(subset['week'], subset['total_clicks'], marker='o', label=str(label))
    plt.xlabel('Week')
    plt.ylabel('Total Clicks')
    plt.title('Total Clicks by Week and Final Result')
    plt.legend(title='Final Result', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_clicks_by_week.png', dpi=150)
    plt.close()


def plot_temporal_activity_events(output_dir: Path) -> None:
    path = output_dir / 'temporal_activity_events_by_week.csv'
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or 'week' not in df.columns or 'events' not in df.columns:
        return
    pivot = df.pivot(index='week', columns='activity_type', values='events').fillna(0)
    plt.figure(figsize=(12, 6))
    pivot.plot(kind='area', stacked=True, ax=plt.gca())
    plt.xlabel('Week')
    plt.ylabel('Event Count')
    plt.title('Activity Event Counts by Week')
    plt.legend(title='Activity Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_activity_events_by_week.png', dpi=150)
    plt.close()


def write_text_report(lines: List[str], path: Path) -> None:
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run dataset forensics on OULAD or an OULAD ML table.')
    parser.add_argument('--input', default='oulad_ml_table.csv', help='ML table CSV input path')
    parser.add_argument('--raw-root', default='.', help='Root folder for raw OULAD CSV files')
    parser.add_argument('--output-dir', default='analysis', help='Directory to write forensic reports')
    parser.add_argument('--target', default='final_result', help='Target column name for classification analysis')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    raw_root = Path(args.raw_root)

    if not input_path.exists():
        print(f'Error: input file not found: {input_path}')
        sys.exit(1)

    print(f'Reading ML table from {input_path}')
    df = read_csv_auto(input_path)
    df.columns = [c.strip() for c in df.columns]

    target_col, target_series = detect_target_column(df, explicit=args.target)
    if target_col is None:
        print('Error: target column not found and could not be inferred.')
        sys.exit(1)
    print(f'Using target column: {target_col}')

    health = summarize_health(df, input_path.name)
    health_lines = print_health_summary(health)
    write_text_report(health_lines, out_dir / 'dataset_health.txt')
    print(f'Wrote dataset_health.txt')

    missing_report = health['missing'].reset_index().rename(columns={'index': 'feature'})
    missing_report.to_csv(out_dir / 'missing_values.csv', index=False)
    const_report = pd.DataFrame({'constant_columns': health['constant_columns']})
    const_report.to_csv(out_dir / 'constant_columns.csv', index=False)
    if health['impossible_values']:
        imm = pd.DataFrame.from_dict(health['impossible_values'], orient='index').reset_index().rename(columns={'index': 'feature'})
        imm.to_csv(out_dir / 'impossible_values.csv', index=False)

    print('Analyzing target distributions...')
    target_report = build_target_report(df, target_col)
    target_report['overall'].to_csv(out_dir / 'target_overall.csv', index=False)
    print('  Wrote target_overall.csv')
    if 'per_module' in target_report:
        target_report['per_module'].to_csv(out_dir / 'target_per_module.csv', index=False)
        print('  Wrote target_per_module.csv')
    if 'per_presentation' in target_report:
        target_report['per_presentation'].to_csv(out_dir / 'target_per_presentation.csv', index=False)
        print('  Wrote target_per_presentation.csv')
    if 'module_presentation' in target_report:
        target_report['module_presentation'].to_csv(out_dir / 'target_per_module_presentation.csv', index=False)
        print('  Wrote target_per_module_presentation.csv')

    by_assessment = target_by_assessment_type(raw_root, find_oulad_files(raw_root), target_col)
    if not by_assessment.empty:
        by_assessment.to_csv(out_dir / 'target_by_assessment_type.csv', index=False)
        print('  Wrote target_by_assessment_type.csv')

    print('Computing feature quality metrics...')
    quality_df = feature_quality(df, target_col)
    quality_df.to_csv(out_dir / 'feature_quality.csv', index=False)
    print('  Wrote feature_quality.csv')

    print('Computing feature redundancy...')
    redundancy_df = feature_redundancy(df)
    redundancy_df.to_csv(out_dir / 'feature_redundancy.csv', index=False)
    print('  Wrote feature_redundancy.csv')

    print('Detecting feature leakage risk...')
    leakage_df = classify_leakage(df, target_col)
    leakage_df.to_csv(out_dir / 'feature_leakage.csv', index=False)
    print('  Wrote feature_leakage.csv')

    print('Computing module-level feature quality...')
    module_quality_df = feature_quality_by_group(df, target_col, group_col='code_module')
    module_quality_df.to_csv(out_dir / 'feature_quality_per_module.csv', index=False)
    print('  Wrote feature_quality_per_module.csv')

    recommendation_df = quality_df[['feature', 'recommendation']].copy()
    recommendation_df.to_csv(out_dir / 'feature_recommendations.csv', index=False)
    print('  Wrote feature_recommendations.csv')

    missing_report = health['missing'].reset_index().rename(columns={'index': 'feature'})
    plot_missing_values(missing_report, out_dir)
    plot_target_distribution(target_report, out_dir)
    plot_feature_quality(quality_df, out_dir)

    summary_lines = [
        'Dataset Forensics Summary',
        '========================',
        '',
        *health_lines[:20],
        '---',
        'Target distribution preview:',
        *render_dataframe(target_report['overall'], max_rows=20),
        '---',
        'Top feature quality preview:',
        *render_dataframe(quality_df[['feature', 'type', 'mutual_info', 'recommendation']].head(20), max_rows=20),
        '---',
        'Leakage preview:',
        *render_dataframe(leakage_df.head(20), max_rows=20),
    ]

    time_lines = temporal_activity_analysis(raw_root, out_dir)
    summary_lines.extend(['---', 'Temporal analysis:'] + time_lines)

    plot_temporal_clicks(out_dir)
    plot_temporal_activity_events(out_dir)

    write_text_report(summary_lines, out_dir / 'forensics_summary.txt')
    print('Wrote forensics_summary.txt')
    print('\nSummary:')
    print('\n'.join(summary_lines[:30]))
    print('\nAnalysis files written to:', out_dir.resolve())


if __name__ == '__main__':
    main()
