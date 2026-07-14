#!/usr/bin/env python3
"""OULAD preprocessing pipeline

Builds a per-student-course ML table from the OULAD CSVs using memory-efficient aggregation.

Usage:
    python oulad_pipeline.py --root . --output oulad_ml_table.csv --chunksize 200000
"""
from pathlib import Path
import argparse
import csv
import re
import warnings
from typing import Dict, List

import pandas as pd
import numpy as np


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
        for sep in [';', ',', '\t', '|']:
            try:
                return pd.read_csv(path, sep=sep, engine='python', **kwargs)
            except Exception:
                continue
    return pd.read_csv(path, **kwargs)


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


def detect_click_col(cols: List[str]):
    candidates = ['sum_click', 'sum_clicks', 'click', 'clicks', 'sum_click_count', 'activity']
    for c in candidates:
        if c in cols:
            return c
    return None


def detect_date_col(cols: List[str]):
    # Prefer 'date' column specifically (OULAD studentVle uses 'date' as the day number)
    for c in cols:
        if c.lower() == 'date':
            return c
    for c in cols:
        low = c.lower()
        if 'date' in low or 'day' in low:
            return c
    return None


def detect_activity_type_col(cols: List[str]):
    candidates = ['activity_type', 'activity', 'type', 'vle_type']
    for c in cols:
        if c.lower() in candidates:
            return c
    for c in cols:
        low = c.lower()
        if 'activity' in low and 'type' in low:
            return c
    return None


def clean_column_name(name: str) -> str:
    value = re.sub(r'[^0-9a-zA-Z]+', '_', name.strip().lower())
    return re.sub(r'__+', '_', value).strip('_')


def week_slope_from_counts(row, cols: List[str]) -> float:
    values = row[cols].astype(float).fillna(0).values
    if np.all(values == 0):
        return 0.0
    x = np.arange(1, len(values) + 1)
    try:
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    except Exception:
        return 0.0


def aggregate_studentvle(path: Path, chunksize: int = 200_000, cutoff_day: int = None) -> pd.DataFrame:
    delim = sniff_delimiter(path)
    reader = pd.read_csv(path, sep=delim, chunksize=chunksize, low_memory=False)
    parts = []
    week_parts = []
    activity_parts = []
    for chunk in reader:
        chunk.columns = [c.strip() for c in chunk.columns]
        group_keys = [k for k in ['id_student', 'code_module', 'code_presentation'] if k in chunk.columns]
        if not group_keys:
            continue
        click_col = detect_click_col(list(chunk.columns))
        date_col = detect_date_col(list(chunk.columns))
        if date_col and date_col in chunk.columns:
            chunk['__day'] = pd.to_numeric(chunk[date_col], errors='coerce')
            if cutoff_day is not None:
                chunk = chunk[chunk['__day'] <= cutoff_day]
        else:
            chunk['__day'] = np.nan

        if click_col and click_col in chunk.columns:
            chunk['__clicks'] = pd.to_numeric(chunk[click_col], errors='coerce').fillna(0)
            activity_type_col = detect_activity_type_col(list(chunk.columns))
            if date_col and date_col in chunk.columns:
                chunk['week_num'] = np.where(chunk['__day'] < 0, 0, (chunk['__day'].fillna(0).astype(int) // 7) + 1)
                week_parts.append(chunk[group_keys + ['week_num', '__clicks']].rename(columns={'__clicks': 'clicks'}))
                if activity_type_col and activity_type_col in chunk.columns:
                    chunk['activity_type'] = chunk[activity_type_col].astype(str).fillna('missing')
                    activity_parts.append(chunk[group_keys + ['activity_type', '__clicks']].rename(columns={'__clicks': 'clicks'}))
                agg = chunk.groupby(group_keys).agg(
                    total_clicks=pd.NamedAgg(column='__clicks', aggfunc='sum'),
                    activity_count=pd.NamedAgg(column='__clicks', aggfunc='count'),
                    first_ts=pd.NamedAgg(column='__day', aggfunc='min'),
                    last_ts=pd.NamedAgg(column='__day', aggfunc='max'),
                    days_active=pd.NamedAgg(column='__day', aggfunc=lambda s: int(s.dropna().nunique()))
                )
            else:
                agg = chunk.groupby(group_keys).agg(
                    total_clicks=pd.NamedAgg(column='__clicks', aggfunc='sum'),
                    activity_count=pd.NamedAgg(column='__clicks', aggfunc='count')
                )
        else:
            if date_col and date_col in chunk.columns:
                chunk['__clicks'] = 1
                agg = chunk.groupby(group_keys).agg(
                    activity_count=pd.NamedAgg(column='__clicks', aggfunc='count'),
                    first_ts=pd.NamedAgg(column='__day', aggfunc='min'),
                    last_ts=pd.NamedAgg(column='__day', aggfunc='max'),
                    days_active=pd.NamedAgg(column='__day', aggfunc=lambda s: int(s.dropna().nunique()))
                )
            else:
                agg = chunk.groupby(group_keys).size().to_frame('activity_count')
        parts.append(agg)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts).reset_index()
    possible_keys = ['id_student', 'code_module', 'code_presentation']
    group_keys = [k for k in possible_keys if k in df.columns]
    if not group_keys:
        group_keys = df.columns[:3].tolist()

    agg_map = {}
    for c in df.columns:
        if c in group_keys:
            continue
        if c in ['first_ts']:
            agg_map[c] = 'min'
        elif c in ['last_ts']:
            agg_map[c] = 'max'
        else:
            if pd.api.types.is_numeric_dtype(df[c]):
                agg_map[c] = 'sum'
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                agg_map[c] = 'min'
            else:
                agg_map[c] = 'first'

    df = df.groupby(group_keys).agg(agg_map).reset_index()
    for col in ['first_ts', 'last_ts']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'first_ts' in df.columns and 'last_ts' in df.columns:
        df['inactivity_days'] = (df['last_ts'] - df['first_ts']).fillna(0).astype(int)
    # Drop correlated features: activity_count, days_active, avg_clicks_per_day are all
    # near-linear functions of total_clicks (corr ~0.9-1.0). Keep total_clicks only.
    for redundant in ['active_weeks', 'activity_count', 'days_active', 'avg_clicks_per_day']:
        if redundant in df.columns:
            df = df.drop(columns=[redundant])

    if week_parts:
        week_df = pd.concat(week_parts, ignore_index=True)
        week_df = week_df.groupby(group_keys + ['week_num'], as_index=False).agg(clicks=('clicks', 'sum'))
        weekly = week_df[week_df['week_num'].between(1, 12)]
        if not weekly.empty:
            week_pivot = weekly.pivot_table(index=group_keys, columns='week_num', values='clicks', fill_value=0).reset_index()
            week_pivot.columns = [f'week{int(c)}_clicks' if isinstance(c, int) else c for c in week_pivot.columns]
            df = df.merge(week_pivot, on=group_keys, how='left')
            for w in range(1, 13):
                col = f'week{w}_clicks'
                if col not in df.columns:
                    df[col] = 0
            week_cols = [f'week{w}_clicks' for w in range(1, 13)]
            df['click_variance'] = df[week_cols].var(axis=1)
            df['click_growth_rate'] = df[week_cols].apply(lambda row: week_slope_from_counts(row, week_cols), axis=1)
            gap_df = week_df[week_df['week_num'] > 0].groupby(group_keys)['week_num'].apply(
                lambda s: int(max(np.diff(np.sort(s.unique())) - 1, default=0))
            ).reset_index(name='longest_inactive_gap')
            df = df.merge(gap_df, on=group_keys, how='left')
            df['longest_inactive_gap'] = df['longest_inactive_gap'].fillna(0).astype(int)
            df['week_click_sum_1_12'] = df[week_cols].sum(axis=1)
            # Cumulative click features — smoother signal than individual weeks
            for cutoff in [2, 4, 6, 8]:
                df[f'clicks_until_week{cutoff}'] = df[[f'week{w}_clicks' for w in range(1, cutoff + 1)]].sum(axis=1)
        else:
            for w in range(1, 13):
                df[f'week{w}_clicks'] = 0
            df['click_variance'] = 0.0
            df['click_growth_rate'] = 0.0
            df['longest_inactive_gap'] = 0
            df['week_click_sum_1_12'] = 0
            for cutoff in [2, 4, 6, 8]:
                df[f'clicks_until_week{cutoff}'] = 0

    if activity_parts:
        act_df = pd.concat(activity_parts, ignore_index=True)
        # Only keep activity types that appear in the VLE data (not 'missing' placeholder)
        act_df = act_df[act_df['activity_type'] != 'missing']
        if not act_df.empty:
            act_totals = act_df.groupby(group_keys + ['activity_type'], as_index=False).agg(activity_clicks=('clicks', 'sum'))
            activity_pivot = act_totals.pivot_table(index=group_keys, columns='activity_type', values='activity_clicks', fill_value=0).reset_index()
            new_columns = []
            for c in activity_pivot.columns:
                if c in group_keys:
                    new_columns.append(c)
                else:
                    new_columns.append(f'activity_type_{clean_column_name(str(c))}')
            activity_pivot.columns = new_columns
            # Drop constant activity-type columns (zero variance across all students)
            act_type_cols = [c for c in activity_pivot.columns if c not in group_keys]
            non_constant = [c for c in act_type_cols if activity_pivot[c].nunique() > 1]
            activity_pivot = activity_pivot[group_keys + non_constant]
            df = df.merge(activity_pivot, on=group_keys, how='left')
            # activity_type_diversity: number of distinct activity types used per student
            diversity = act_df.groupby(group_keys)['activity_type'].nunique().reset_index(name='activity_type_diversity')
            df = df.merge(diversity, on=group_keys, how='left')
            df['activity_type_diversity'] = df['activity_type_diversity'].fillna(0).astype(int)

    return df


def compute_assessment_trend(group: pd.DataFrame, score_col: str) -> float:
    if score_col not in group.columns or 'date_submitted' not in group.columns:
        return np.nan
    scores = pd.to_numeric(group[score_col], errors='coerce').dropna()
    dates = pd.to_numeric(group.loc[scores.index, 'date_submitted'], errors='coerce').dropna()
    if len(scores) < 2 or dates.empty:
        return np.nan
    ordered = pd.DataFrame({'date': dates, 'score': scores.loc[dates.index].astype(float)})
    ordered = ordered.sort_values('date')
    if len(ordered) < 2:
        return 0.0
    first_score = ordered['score'].iloc[0]
    last_score = ordered['score'].iloc[-1]
    first_day = ordered['date'].iloc[0]
    last_day = ordered['date'].iloc[-1]
    if last_day == first_day:
        return float(last_score - first_score)
    return float((last_score - first_score) / (last_day - first_day))


def aggregate_assessments(student_assessment_path: Path, assessments_path: Path, cutoff_day: int = None) -> pd.DataFrame:
    sa = read_csv_auto(student_assessment_path)
    ass = read_csv_auto(assessments_path)
    sa.columns = [c.strip() for c in sa.columns]
    ass.columns = [c.strip() for c in ass.columns]

    if cutoff_day is not None and 'date_submitted' in sa.columns:
        sa['date_submitted'] = pd.to_numeric(sa['date_submitted'], errors='coerce')
        sa = sa[sa['date_submitted'] <= cutoff_day]

    total_assessments = None
    if 'id_assessment' in ass.columns:
        total_assessments = ass.groupby([k for k in ['code_module', 'code_presentation'] if k in ass.columns]).agg(total_assessments=('id_assessment', 'nunique')).reset_index()

    if 'id_assessment' in sa.columns and 'id_assessment' in ass.columns:
        cols_to_take = ['id_assessment']
        for optional in ['code_module', 'code_presentation', 'date', 'assessment_type', 'weight']:
            if optional in ass.columns:
                cols_to_take.append(optional)
        merged = sa.merge(ass[cols_to_take].drop_duplicates(), on='id_assessment', how='left')
    else:
        merged = sa

    group_keys = [k for k in ['id_student', 'code_module', 'code_presentation'] if k in merged.columns]
    if not group_keys:
        return pd.DataFrame()

    merged['date_submitted'] = pd.to_numeric(merged['date_submitted'], errors='coerce') if 'date_submitted' in merged.columns else np.nan

    score_col = None
    for cand in ['score', 'score_student', 'score_value', 'score_obtained']:
        if cand in merged.columns:
            score_col = cand
            break

    if score_col:
        merged[score_col] = pd.to_numeric(merged[score_col], errors='coerce')

    if score_col:
        agg = merged.groupby(group_keys).agg(
            avg_score=pd.NamedAgg(column=score_col, aggfunc='mean'),
            score_std=pd.NamedAgg(column=score_col, aggfunc=lambda s: s.std() if s.notnull().sum() >= 2 else np.nan),
            num_assessments=pd.NamedAgg(column=score_col, aggfunc=lambda s: int(s.notnull().sum())),
            first_assessment_day=pd.NamedAgg(column='date_submitted', aggfunc='min'),
            last_assessment_day=pd.NamedAgg(column='date_submitted', aggfunc='max')
        )
        agg['assessment_score_trend'] = merged.groupby(group_keys, group_keys=False).apply(
            lambda g: compute_assessment_trend(g, score_col)
        ).astype(float)
    else:
        agg = merged.groupby(group_keys).agg(
            num_assessments=pd.NamedAgg(column='id_assessment', aggfunc='count')
        )

    agg = agg.reset_index()
    join_keys = [k for k in ['code_module', 'code_presentation'] if total_assessments is not None and k in agg.columns and k in total_assessments.columns]
    if total_assessments is not None and join_keys:
        agg = agg.merge(total_assessments, on=join_keys, how='left')
        agg['assessment_completion_ratio'] = agg['num_assessments'] / agg['total_assessments'].replace(0, np.nan)
        agg['missed_assessments'] = agg['total_assessments'] - agg['num_assessments']
        agg['assessment_completion_ratio'] = agg['assessment_completion_ratio'].fillna(0)
        agg['missed_assessments'] = agg['missed_assessments'].fillna(0).astype(int)
    if 'date' in merged.columns and 'date_submitted' in merged.columns:
        late = merged[merged['date_submitted'] > pd.to_numeric(merged['date'], errors='coerce')]
        late_counts = late.groupby(group_keys).size().rename('late_submission_count').reset_index()
        agg = agg.merge(late_counts, on=group_keys, how='left')
        agg['late_submission_count'] = agg['late_submission_count'].fillna(0).astype(int)
    if 'first_assessment_day' in agg.columns and 'last_assessment_day' in agg.columns:
        agg['assessment_span_days'] = (agg['last_assessment_day'] - agg['first_assessment_day']).fillna(0).astype(int)
    return agg


def build_oulad_ml_table(root: Path, out_path: Path, chunksize: int = 200_000, week_cutoffs: List[int] = None):
    files = find_oulad_files(root)
    if 'studentInfo.csv' not in files:
        print('Missing studentInfo.csv in OULAD folder. Aborting.')
        return

    student_info = read_csv_auto(files['studentInfo.csv'])
    student_info.columns = [c.strip() for c in student_info.columns]
    print(f'Loaded studentInfo: {len(student_info)} rows')

    keys = [k for k in ['id_student', 'code_module', 'code_presentation'] if k in student_info.columns]
    if not keys:
        print('studentInfo missing id_student/code_module/code_presentation. Aborting.')
        return

    def build_table(cutoff_weeks: int = None) -> pd.DataFrame:
        vle_agg = None
        if 'studentVle.csv' in files:
            print('Aggregating studentVle{}...'.format(f' up to week {cutoff_weeks}' if cutoff_weeks else ''))
            cutoff_day = cutoff_weeks * 7 if cutoff_weeks is not None else None
            vle_agg = aggregate_studentvle(files['studentVle.csv'], chunksize=chunksize, cutoff_day=cutoff_day)
            print(f'studentVle aggregated rows: {len(vle_agg)}')

        assess_agg = None
        if 'studentAssessment.csv' in files and 'assessments.csv' in files:
            print('Aggregating assessments{}...'.format(f' up to week {cutoff_weeks}' if cutoff_weeks else ''))
            cutoff_day = cutoff_weeks * 7 if cutoff_weeks is not None else None
            assess_agg = aggregate_assessments(files['studentAssessment.csv'], files['assessments.csv'], cutoff_day=cutoff_day)
            print(f'assessment aggregates: {len(assess_agg)}')

        base = student_info.drop_duplicates(subset=keys).copy()
        if vle_agg is not None and not vle_agg.empty:
            base = base.merge(vle_agg, on=keys, how='left')
        if assess_agg is not None and not assess_agg.empty:
            base = base.merge(assess_agg, on=keys, how='left')
        if 'studentRegistration.csv' in files:
            reg = read_csv_auto(files['studentRegistration.csv'])
            reg.columns = [c.strip() for c in reg.columns]
            reg_keys = [k for k in ['id_student', 'code_module', 'code_presentation'] if k in reg.columns]
            if reg_keys:
                reg_small = reg.drop_duplicates(subset=reg_keys)
                base = base.merge(reg_small, on=reg_keys, how='left')

        for leak in ['date_unregistration', 'date_unreg', 'date_unregistered']:
            if leak in base.columns:
                base = base.drop(columns=[leak])

        for col in ['total_clicks', 'num_assessments', 'avg_score', 'weighted_score', 'inactivity_days', 'clicks_per_credit', 'click_variance', 'click_growth_rate'] + [f'clicks_until_week{c}' for c in [2, 4, 6, 8]]:
            if col in base.columns:
                base[col] = base[col].fillna(0)

        if 'date_registration' in base.columns:
            base['date_registration'] = pd.to_numeric(base['date_registration'], errors='coerce')
            base['registration_early_days'] = base['date_registration'].fillna(0).astype(int)
            base = base.drop(columns=['date_registration'], errors='ignore')

        # registration_delay_category removed: redundant with registration_early_days (numeric)

        # clicks_per_active_week removed: it is a linear rescaling of avg_clicks_per_day (corr=1.0)
        if 'total_clicks' in base.columns and 'studied_credits' in base.columns:
            base['clicks_per_credit'] = base['total_clicks'] / base['studied_credits'].replace(0, np.nan)
        if 'num_assessments' in base.columns and 'studied_credits' in base.columns:
            base['credits_per_attempt'] = base['studied_credits'] / base['num_assessments'].replace(0, np.nan)

        return base

    weekly_tables = []
    main_table = build_table(cutoff_weeks=None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    main_table.to_csv(out_path, index=False)
    print(f'Wrote ML table to {out_path} (rows={len(main_table)})')

    if week_cutoffs:
        for week in sorted(set(week_cutoffs)):
            suffix = f'_week{week}'
            week_path = out_path.parent / f'{out_path.stem}{suffix}{out_path.suffix}'
            week_table = build_table(cutoff_weeks=week)
            week_table.to_csv(week_path, index=False)
            print(f'Wrote ML table for week {week} to {week_path} (rows={len(week_table)})')

            cutoff_day = week * 7
            for col in ['first_ts', 'last_ts', 'first_assessment_day', 'last_assessment_day']:
                if col in week_table.columns:
                    if week_table[col].dropna().gt(cutoff_day).any():
                        print(f'WARNING: week {week} table has {col} > {cutoff_day}. Check cutoff logic for leakage.')


def main():
    parser = argparse.ArgumentParser(description='Build OULAD ML table')
    parser.add_argument('--root', default='.', help='Workspace root')
    parser.add_argument('--output', default='oulad_ml_table.csv', help='Output CSV file')
    parser.add_argument('--chunksize', type=int, default=200000, help='Chunk size for large files')
    parser.add_argument('--week-cutoffs', default='', help='Comma-separated list of week cutoffs to export partial ML tables (e.g. 2,4,6,8)')
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    week_cutoffs = [int(x.strip()) for x in args.week_cutoffs.split(',') if x.strip().isdigit()]
    build_oulad_ml_table(root, out, chunksize=args.chunksize, week_cutoffs=week_cutoffs)


if __name__ == '__main__':
    main()
