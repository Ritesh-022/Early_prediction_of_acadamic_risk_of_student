#!/usr/bin/env python3
"""
OULAD Preprocessing Pipeline V2 — Rich Temporal Features
=========================================================
Extracts day-level and week-level behavioural patterns from raw OULAD files.

New temporal features designed to distinguish Fail vs Withdrawn:
  - first_active_day, last_active_day, last_active_week
  - silence_onset_day, silence_onset_week (day/week student went inactive)
  - consecutive_inactive_weeks, days_since_last_activity
  - engagement_slope, engagement_decay, engagement_acceleration
  - early_activity_ratio, late_activity_ratio
  - week_of_peak_activity, peak_week_clicks
  - resource-type features (forum_clicks, quiz_clicks, content_clicks, etc.)
  - assessment submission timing (first/last submission day, submission delays)

Usage:
    python oulad_pipeline_v2.py --root . --output oulad_ml_table_v2.csv
"""
from pathlib import Path
import argparse
import csv
import warnings
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# FILE I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sniff_delimiter(path: Path) -> str:
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            sample = ''.join([next(f) for _ in range(10)])
    except Exception:
        return ','
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[',', ';', '\t', '|'])
        return dialect.delimiter
    except Exception:
        return ','


def read_csv_auto(path: Path, **kw) -> pd.DataFrame:
    for sep in [None, ',', ';', '\t']:
        try:
            engine = "python" if sep is None else None
            df = pd.read_csv(path, sep=sep, engine=engine, low_memory=False, **kw)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, low_memory=False, **kw)


def find_oulad_files(root: Path) -> Dict[str, Path]:
    names = ['studentInfo.csv', 'studentVle.csv', 'studentAssessment.csv',
             'assessments.csv', 'vle.csv', 'studentRegistration.csv', 'courses.csv']
    found = {}
    for p in root.rglob('*.csv'):
        if p.name in names:
            found[p.name] = p
    return found

# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _slope(arr: np.ndarray) -> float:
    """Linear regression slope over positions 0..n-1."""
    n = len(arr)
    if n < 2 or np.all(arr == 0):
        return 0.0
    x = np.arange(n, dtype=float)
    try:
        return float(np.polyfit(x, arr, 1)[0])
    except Exception:
        return 0.0


def _find_silence_onset(weekly_clicks: np.ndarray, min_silence: int = 2) -> int:
    """
    Return the 1-indexed week at which the student goes permanently silent
    (min_silence consecutive zero-click weeks that extend to the end).
    Returns 0 if the student never goes permanently silent.
    """
    n = len(weekly_clicks)
    for start in range(n - min_silence + 1):
        if np.all(weekly_clicks[start:] == 0):
            return start + 1   # 1-indexed week
    return 0   # never went permanently silent


def _consec_inactive_weeks(weekly_clicks: np.ndarray) -> int:
    """Max consecutive zero-click weeks."""
    max_run = cur = 0
    for v in weekly_clicks:
        if v == 0:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def _activity_quartile_ratios(weekly_clicks: np.ndarray):
    """
    Split the week array into early (Q1), middle (Q2+Q3), late (Q4) thirds.
    Returns (early_ratio, mid_ratio, late_ratio) normalised to total.
    """
    total = weekly_clicks.sum()
    if total == 0 or len(weekly_clicks) < 4:
        return 0.0, 0.0, 0.0
    n  = len(weekly_clicks)
    q1 = max(1, n // 4)
    q4 = max(1, n // 4)
    early = weekly_clicks[:q1].sum()
    late  = weekly_clicks[-q4:].sum()
    mid   = total - early - late
    return early / total, mid / total, late / total


# ══════════════════════════════════════════════════════════════════════════════
# VLE AGGREGATION — full temporal features
# ══════════════════════════════════════════════════════════════════════════════

# Map activity_type → feature group
ACTIVITY_GROUPS = {
    "forumng":     "forum",
    "forum":       "forum",
    "quiz":        "quiz",
    "oucontent":   "content",
    "resource":    "resource",
    "homepage":    "homepage",
    "subpage":     "subpage",
    "url":         "url",
    "externalquiz":"quiz",
    "glossary":    "content",
    "page":        "content",
    "dataplus":    "content",
    "ouwiki":      "content",
    "folder":      "resource",
    "questionnaire":"quiz",
    "ouelluminate":"forum",
    "sharedsubpage":"subpage",
    "repeatactivity":"content",
    "htmlactivity": "content",
}


def aggregate_vle_temporal(vle_path: Path,
                            vle_meta_path: Optional[Path],
                            chunksize: int = 200_000,
                            cutoff_day: Optional[int] = None) -> pd.DataFrame:
    """
    Single-pass chunked aggregation over studentVle.csv.
    Returns one row per (id_student, code_module, code_presentation).
    """
    # Load VLE metadata for activity_type join
    act_type_map: Dict = {}
    if vle_meta_path and vle_meta_path.exists():
        vle_meta = read_csv_auto(vle_meta_path)
        vle_meta.columns = [c.strip() for c in vle_meta.columns]
        if 'id_site' in vle_meta.columns and 'activity_type' in vle_meta.columns:
            act_type_map = dict(zip(vle_meta['id_site'],
                                    vle_meta['activity_type'].str.lower().str.strip()))
    print(f"  VLE metadata: {len(act_type_map)} sites mapped to activity types")

    # ── streaming aggregation ──────────────────────────────────────────────
    GROUP_KEYS = ['id_student', 'code_module', 'code_presentation']

    # per-day click accumulator: {group_key → {day: clicks}}
    day_acc:  Dict = {}  # group → {day: total_clicks}
    site_acc: Dict = {}  # group → set of id_site

    # activity-type click accumulators
    activity_group_acc: Dict = {}  # group → {group_name: clicks}

    print("  Streaming studentVle.csv ...")
    delim = sniff_delimiter(vle_path)
    reader = pd.read_csv(vle_path, sep=delim, chunksize=chunksize, low_memory=False)

    total_rows = 0
    for chunk_idx, chunk in enumerate(reader):
        chunk.columns = [c.strip() for c in chunk.columns]

        # identify columns
        date_col  = next((c for c in chunk.columns if c.lower() == 'date'), None)
        click_col = next((c for c in chunk.columns
                          if c.lower() in ('sum_click','sum_clicks','clicks','click')), None)
        site_col  = next((c for c in chunk.columns if c.lower() == 'id_site'), None)

        if date_col is None or click_col is None:
            continue

        chunk['__day']   = pd.to_numeric(chunk[date_col],  errors='coerce')
        chunk['__click'] = pd.to_numeric(chunk[click_col], errors='coerce').fillna(0)

        if cutoff_day is not None:
            chunk = chunk[chunk['__day'] <= cutoff_day]
        if chunk.empty:
            continue

        # group keys present in this chunk
        gkeys = [k for k in GROUP_KEYS if k in chunk.columns]
        if not gkeys:
            continue

        # join activity_type from meta
        if site_col and act_type_map:
            chunk['__act_group'] = chunk[site_col].map(act_type_map).map(
                ACTIVITY_GROUPS).fillna('other')
        else:
            chunk['__act_group'] = 'other'

        # iteration via numpy arrays (faster than itertuples with __ columns)
        key_arr   = chunk[gkeys].values
        day_arr   = chunk['__day'].values
        click_arr = chunk['__click'].values
        atype_arr = chunk['__act_group'].values
        site_arr  = chunk[site_col].values if site_col else None

        for i in range(len(chunk)):
            key   = tuple(key_arr[i])
            day   = day_arr[i]
            click = click_arr[i]
            atype = atype_arr[i]

            # day-level accumulator
            if key not in day_acc:
                day_acc[key]            = {}
                activity_group_acc[key] = {}
                if site_col:
                    site_acc[key] = set()

            if pd.notna(day):
                d = int(day)
                day_acc[key][d] = day_acc[key].get(d, 0) + click

            # activity group
            if atype and pd.notna(atype):
                activity_group_acc[key][atype] = activity_group_acc[key].get(atype, 0) + click

            # unique sites
            if site_col and site_arr is not None:
                site_val = site_arr[i]
                if pd.notna(site_val):
                    site_acc[key].add(int(site_val))

        total_rows += len(chunk)
        if (chunk_idx + 1) % 10 == 0:
            print(f"    ... processed {total_rows:,} rows ({len(day_acc):,} students)")

    print(f"  Total rows processed: {total_rows:,}  |  Unique students: {len(day_acc):,}")

    # ── build feature rows ─────────────────────────────────────────────────
    # capture the group keys structure for unpacking later
    # keys from the first entry (all entries have the same key structure)
    if not day_acc:
        return pd.DataFrame()
    sample_key = next(iter(day_acc.keys()))
    gkeys_order = GROUP_KEYS[:len(sample_key)]  # safe assumption: same order as GROUP_KEYS

    rows = []
    for key, day_dict in day_acc.items():
        if not day_dict:
            continue

        days  = sorted(day_dict.keys())
        clicks_by_day = np.array([day_dict[d] for d in days], dtype=float)

        # basic temporal bounds
        first_day = min(days)
        last_day  = max(days)
        active_days  = len(days)
        total_clicks = float(clicks_by_day.sum())

        # week-level aggregation (week = day // 7, week 0 = days 0–6, etc.)
        # negative days = pre-course; map to week 0 or keep negative week
        week_dict: Dict[int, float] = {}
        for d, c in zip(days, clicks_by_day):
            w = int(d) // 7 if int(d) >= 0 else -1   # pre-course = week -1
            week_dict[w] = week_dict.get(w, 0.0) + c

        course_weeks = [w for w in week_dict if w >= 0]
        first_week   = min(course_weeks) if course_weeks else -1
        last_week    = max(course_weeks) if course_weeks else -1
        active_weeks = len(set(course_weeks))

        # weekly click array for weeks 0..last_week
        max_week = max(course_weeks) if course_weeks else 0
        weekly_arr = np.array([week_dict.get(w, 0.0)
                                for w in range(max(0, min(course_weeks, default=0)),
                                               max_week + 1)
                                ] if course_weeks else [0.0], dtype=float)

        # silence onset
        silence_onset_week = _find_silence_onset(weekly_arr, min_silence=2)
        consec_inactive    = _consec_inactive_weeks(weekly_arr)

        # engagement slope over weeks
        eng_slope = _slope(weekly_arr)

        # early/mid/late activity ratios
        early_ratio, mid_ratio, late_ratio = _activity_quartile_ratios(weekly_arr)

        # peak week
        if len(weekly_arr) > 0 and weekly_arr.max() > 0:
            peak_week_idx = int(np.argmax(weekly_arr))
            peak_week     = (first_week + peak_week_idx) if course_weeks else -1
            peak_clicks   = float(weekly_arr.max())
        else:
            peak_week = -1
            peak_clicks = 0.0

        # pre-course activity (negative days)
        precourse_clicks = float(sum(c for d, c in zip(days, clicks_by_day) if d < 0))

        # unique resources
        unique_sites = len(site_acc.get(key, set()))

        # activity-type breakdown
        ag = activity_group_acc.get(key, {})
        forum_clicks    = float(ag.get('forum',    0))
        quiz_clicks     = float(ag.get('quiz',     0))
        content_clicks  = float(ag.get('content',  0))
        resource_clicks = float(ag.get('resource', 0))
        homepage_clicks = float(ag.get('homepage', 0))
        subpage_clicks  = float(ag.get('subpage',  0))
        url_clicks      = float(ag.get('url',      0))
        other_clicks    = float(ag.get('other',    0))
        unique_act_types= int(len([v for v in ag.values() if v > 0]))

        tc = total_clicks if total_clicks > 0 else np.nan
        forum_ratio    = forum_clicks    / tc if tc else 0.0
        quiz_ratio     = quiz_clicks     / tc if tc else 0.0
        content_ratio  = content_clicks  / tc if tc else 0.0
        resource_ratio = resource_clicks / tc if tc else 0.0

        # silence: days from last_active_day to course end
        # we don't know course length from VLE alone; use last_week as proxy
        # this will be joined later to get proper days_since_last_activity
        # with respect to module end date

        row_dict = {}
        # unpack tuple key: (id_student, code_module, code_presentation)
        for k_name, k_val in zip(gkeys_order, key):
            row_dict[k_name] = k_val
        row_dict.update({
            # ── basic engagement ────────────────────────────────────────────
            'total_clicks_v2':      total_clicks,
            'active_days':          float(active_days),
            'active_weeks_v2':      float(active_weeks),
            'first_active_day':     float(first_day),
            'last_active_day':      float(last_day),
            'first_active_week':    float(first_week),
            'last_active_week':     float(last_week),
            'precourse_clicks':     precourse_clicks,
            # ── inactivity / silence ─────────────────────────────────────────
            'silence_onset_week':        float(silence_onset_week),
            'went_silent':               float(1 if silence_onset_week > 0 else 0),
            'consec_inactive_weeks':     float(consec_inactive),
            'days_active_span':          float(last_day - first_day) if last_day > first_day else 0.0,
            'active_day_density':        float(active_days) / max(1.0, float(last_day - first_day)),
            # ── engagement trend ──────────────────────────────────────────────
            'engagement_slope':          eng_slope,
            'early_activity_ratio':      early_ratio,
            'mid_activity_ratio':        mid_ratio,
            'late_activity_ratio':       late_ratio,
            'peak_week':                 float(peak_week),
            'peak_week_clicks':          peak_clicks,
            # ── resource diversity ────────────────────────────────────────────
            'unique_resources_accessed': float(unique_sites),
            'resource_diversity':        float(unique_sites) / max(1.0, total_clicks),
            'unique_act_types':          float(unique_act_types),
            # ── activity type breakdown ────────────────────────────────────────
            'forum_clicks':             forum_clicks,
            'quiz_clicks':              quiz_clicks,
            'content_clicks':           content_clicks,
            'resource_clicks':          resource_clicks,
            'homepage_clicks':          homepage_clicks,
            'subpage_clicks':           subpage_clicks,
            'url_clicks':               url_clicks,
            'forum_ratio':              forum_ratio,
            'quiz_ratio':               quiz_ratio,
            'content_ratio':            content_ratio,
            'resource_ratio':           resource_ratio,
        })
        rows.append(row_dict)

    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT AGGREGATION — temporal submission features
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_assessments_temporal(sa_path: Path, assess_path: Path,
                                     cutoff_day: Optional[int] = None) -> pd.DataFrame:
    """
    Join studentAssessment + assessments.
    Returns temporal submission features per student.
    """
    sa  = read_csv_auto(sa_path)
    ass = read_csv_auto(assess_path)
    sa.columns  = [c.strip() for c in sa.columns]
    ass.columns = [c.strip() for c in ass.columns]

    # apply day cutoff to submissions
    if 'date_submitted' in sa.columns and cutoff_day is not None:
        sa['date_submitted'] = pd.to_numeric(sa['date_submitted'], errors='coerce')
        sa = sa[sa['date_submitted'] <= cutoff_day]

    # merge to get due dates + assessment type
    if 'id_assessment' in sa.columns and 'id_assessment' in ass.columns:
        cols_to_merge = ['id_assessment', 'date', 'assessment_type', 'weight']
        cols_to_merge = [c for c in cols_to_merge if c in ass.columns]
        merged = sa.merge(ass[cols_to_merge].drop_duplicates(), on='id_assessment', how='left')
        merged = merged.rename(columns={'date': 'due_date'})
    else:
        merged = sa.copy()
        merged['due_date']        = np.nan
        merged['assessment_type'] = 'unknown'
        merged['weight']          = np.nan

    # normalise columns
    if 'date_submitted' in merged.columns:
        merged['date_submitted'] = pd.to_numeric(merged['date_submitted'], errors='coerce')
    else:
        merged['date_submitted'] = np.nan
    if 'score' in merged.columns:
        merged['score'] = pd.to_numeric(merged['score'], errors='coerce')
    else:
        merged['score'] = np.nan
    if 'due_date' in merged.columns:
        merged['due_date'] = pd.to_numeric(merged['due_date'], errors='coerce')
    if 'weight' in merged.columns:
        merged['weight'] = pd.to_numeric(merged['weight'], errors='coerce')

    # submission delay = submitted - due (negative = early, positive = late)
    merged['submission_delay'] = merged['date_submitted'] - merged['due_date']

    group_keys = ['id_student']  # studentAssessment has no module/presentation columns
    if 'id_student' not in merged.columns:
        return pd.DataFrame()

    # aggregate per student
    def agg_func(g):
        sub = g['date_submitted'].dropna()
        scores = g['score'].dropna()
        delays = g['submission_delay'].dropna()
        weights= g['weight'].dropna()

        first_sub = float(sub.min()) if len(sub) > 0 else np.nan
        last_sub  = float(sub.max()) if len(sub) > 0 else np.nan
        n_sub     = len(sub)
        n_total   = len(g)
        submit_ratio = n_sub / n_total if n_total > 0 else 0.0

        # score trend (first vs last)
        if len(scores) >= 2:
            first_score = float(scores.iloc[0])
            last_score  = float(scores.iloc[-1])
            score_trend = last_score - first_score
            score_vol   = float(scores.std())
        elif len(scores) == 1:
            first_score = float(scores.iloc[0])
            last_score  = first_score
            score_trend = 0.0
            score_vol   = 0.0
        else:
            first_score = np.nan
            last_score  = np.nan
            score_trend = 0.0
            score_vol   = 0.0

        avg_score = float(scores.mean()) if len(scores) > 0 else np.nan
        max_score = float(scores.max()) if len(scores) > 0 else np.nan
        min_score = float(scores.min()) if len(scores) > 0 else np.nan

        # submission timing
        late_count    = int((delays > 0).sum())
        early_count   = int((delays < 0).sum())
        avg_delay     = float(delays.mean()) if len(delays) > 0 else 0.0
        max_delay     = float(delays.max()) if len(delays) > 0 else 0.0

        # assessment span
        assess_span   = last_sub - first_sub if pd.notna(first_sub) and pd.notna(last_sub) else 0.0

        # TMA vs CMA (if available)
        if 'assessment_type' in g.columns:
            tma_count = int((g['assessment_type'].str.lower() == 'tma').sum())
            cma_count = int((g['assessment_type'].str.lower() == 'cma').sum())
            first_tma_submitted = int((g[g['assessment_type'].str.lower() == 'tma']
                                        ['date_submitted'].notna().any()))
        else:
            tma_count = 0
            cma_count = 0
            first_tma_submitted = 0

        # weighted average score
        if 'weight' in g.columns and len(scores) > 0 and len(weights) > 0:
            valid = g[g['score'].notna() & g['weight'].notna()]
            if len(valid) > 0:
                weighted_avg = (valid['score'] * valid['weight']).sum() / valid['weight'].sum()
            else:
                weighted_avg = avg_score
        else:
            weighted_avg = avg_score

        return pd.Series({
            'first_submission_day':     first_sub,
            'last_submission_day':      last_sub,
            'n_submissions':            float(n_sub),
            'n_assessments_available':  float(n_total),
            'submission_ratio':         submit_ratio,
            'avg_score_v2':             avg_score,
            'weighted_avg_score':       weighted_avg,
            'max_score':                max_score,
            'min_score':                min_score,
            'first_assessment_score':   first_score,
            'last_assessment_score':    last_score,
            'score_trend':              score_trend,
            'score_volatility':         score_vol,
            'late_submission_count_v2': float(late_count),
            'early_submission_count':   float(early_count),
            'avg_submission_delay':     avg_delay,
            'max_submission_delay':     max_delay,
            'assessment_submission_span': assess_span,
            'tma_count':                float(tma_count),
            'cma_count':                float(cma_count),
            'first_tma_submitted':      float(first_tma_submitted),
        })

    # aggregate per student using iteration for robustness
    agg_rows = []
    for gkey, g in merged.groupby(group_keys):
        row_dict = dict(zip(group_keys, gkey if isinstance(gkey, tuple) else [gkey]))
        row_dict.update(agg_func(g).to_dict())
        agg_rows.append(row_dict)
    result = pd.DataFrame(agg_rows)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_oulad_ml_table_v2(root: Path, out_path: Path, chunksize: int = 200_000):
    files = find_oulad_files(root)
    if 'studentInfo.csv' not in files:
        raise FileNotFoundError("studentInfo.csv not found.")

    print("\n" + "="*65)
    print("  OULAD ML Table V2 Builder — Rich Temporal Features")
    print("="*65)

    # Load base student info
    student_info = read_csv_auto(files['studentInfo.csv'])
    student_info.columns = [c.strip() for c in student_info.columns]
    print(f"\n  studentInfo: {len(student_info):,} rows")

    keys = ['id_student', 'code_module', 'code_presentation']
    base = student_info.drop_duplicates(subset=keys).copy()

    # ── VLE temporal features ─────────────────────────────────────────────────
    if 'studentVle.csv' in files:
        print(f"\n  Processing studentVle.csv ...")
        vle_path = files['studentVle.csv']
        vle_meta_path = files.get('vle.csv')
        vle_agg = aggregate_vle_temporal(vle_path, vle_meta_path, chunksize=chunksize)
        print(f"  VLE aggregated: {len(vle_agg):,} rows")
        base = base.merge(vle_agg, on=keys, how='left')
    else:
        print("\n  studentVle.csv not found — skipping VLE features.")

    # ── Assessment temporal features ──────────────────────────────────────────
    if 'studentAssessment.csv' in files and 'assessments.csv' in files:
        print(f"\n  Processing assessments ...")
        assess_agg = aggregate_assessments_temporal(
            files['studentAssessment.csv'], files['assessments.csv'])
        print(f"  Assessments aggregated: {len(assess_agg):,} rows")
        base = base.merge(assess_agg, on=['id_student'], how='left')
    else:
        print("\n  Assessment files not found — skipping assessment features.")

    # ── Registration features (early registration = positive predictor) ───────
    if 'studentRegistration.csv' in files:
        reg = read_csv_auto(files['studentRegistration.csv'])
        reg.columns = [c.strip() for c in reg.columns]
        reg_keys = [k for k in keys if k in reg.columns]
        if reg_keys:
            reg_small = reg.drop_duplicates(subset=reg_keys)
            if 'date_registration' in reg_small.columns:
                reg_small['registration_early_days_v2'] = pd.to_numeric(
                    reg_small['date_registration'], errors='coerce').fillna(0).astype(int)
                reg_small = reg_small.drop(columns=['date_registration'], errors='ignore')
            base = base.merge(reg_small, on=reg_keys, how='left')
            print(f"  Registration features merged.")

    # ── Drop leakage columns ──────────────────────────────────────────────────
    leakage = ['date_unregistration', 'date_unreg', 'date_unregistered']
    base = base.drop(columns=[c for c in leakage if c in base.columns], errors='ignore')

    # ── Fill NaNs with 0 for activity/assessment-derived features ────────────
    zero_fill_patterns = ['_clicks', '_ratio', '_count', 'n_', 'avg_', 'submission',
                          'score', 'active', 'unique', 'diversity', 'silence', 'consec',
                          'days_', 'weeks_', 'tma', 'cma', 'went_']
    num_cols = base.select_dtypes(include='number').columns
    for col in num_cols:
        if any(p in col.lower() for p in zero_fill_patterns):
            base[col] = base[col].fillna(0)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out_path, index=False)
    print(f"\n{'='*65}")
    print(f"  ML Table V2 saved → {out_path}")
    print(f"  Shape: {base.shape}")
    print(f"{'='*65}")

    # ── Summary report ────────────────────────────────────────────────────────
    print(f"\n  New Temporal Features Added:")
    temporal_keywords = ['first_active', 'last_active', 'silence', 'slope', 'peak',
                         'consec', 'early_activity', 'late_activity', 'diversity',
                         'forum_', 'quiz_', 'content_', 'resource_',
                         'submission_day', 'score_trend', 'submission_delay', 'tma']
    new_feats = [c for c in base.columns if any(k in c.lower() for k in temporal_keywords)]
    for i, feat in enumerate(sorted(new_feats), 1):
        print(f"    {i:2}. {feat}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OULAD ML Table V2 Builder")
    parser.add_argument("--root", default=".", help="Workspace root")
    parser.add_argument("--output", default="oulad_ml_table_v2.csv")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out  = Path(args.output).expanduser().resolve()
    build_oulad_ml_table_v2(root, out, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
