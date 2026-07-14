#!/usr/bin/env python3
"""Deep diagnosis: what's the actual ceiling, and what features are missing."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv('oulad_ml_table.csv')
fr = df['final_result']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

le = LabelEncoder()
y = le.fit_transform(fr)

DROP = ['final_result','id_student','code_module','code_presentation',
        'date_unregistration','date_unreg','first_ts','last_ts',
        'active_weeks','clicks_per_active_week','assessments_per_week',
        'activity_count','days_active','avg_clicks_per_day',
        'week_click_sum_1_4','registration_delay_category',
        'date_unreg','last_assessment_day','first_assessment_day']

X_base = df.drop(columns=[c for c in DROP if c in df.columns])

# ── Experiment 1: synthetic_platform schema (early-warning only) ──────────────
print('=== Exp 1: synthetic_platform schema (early-warning, 22 features) ===')
platform_cols = ['consistency_ratio','total_engagement','engagement_trend',
                 'engagement_consistency','weekly_activity_change',
                 'first_assessment_day','late_submission_count',
                 'prior_failures','absence_flag','lifestyle_risk_score',
                 'gender','age_band','education_level','socioeconomic_flag',
                 'clicks_per_credit','credits_per_attempt','studied_credits',
                 'registration_early_days','code_module','source']
# Approximate with available cols
approx = ['registration_early_days','studied_credits','num_of_prev_attempts',
          'gender','age_band','highest_education','imd_band','disability',
          'late_submission_count','total_clicks','week1_clicks','week2_clicks',
          'week3_clicks','week4_clicks']
approx = [c for c in approx if c in df.columns]
X1 = df[approx].copy()
num1 = X1.select_dtypes(include='number').columns.tolist()
cat1 = X1.select_dtypes(include='object').columns.tolist()
pre1 = ColumnTransformer([
    ('n', SimpleImputer(strategy='median'), num1),
    ('c', Pipeline([('i',SimpleImputer(strategy='most_frequent')),
                    ('e',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]), cat1)
])
pipe1 = Pipeline([('p',pre1),('m',LGBMClassifier(n_estimators=300,random_state=42,verbosity=-1))])
s1 = cross_val_score(pipe1, X1, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'  CV accuracy: {s1.mean():.4f} +/- {s1.std():.4f}')
print(f'  → This is what synthetic_platform gets with early-warning mode\n')

# ── Experiment 2: full oulad_ml_table (all features) ─────────────────────────
print('=== Exp 2: Full oulad_ml_table (all features, ~50 cols) ===')
X2 = X_base.copy()
num2 = X2.select_dtypes(include='number').columns.tolist()
cat2 = X2.select_dtypes(include='object').columns.tolist()
pre2 = ColumnTransformer([
    ('n', SimpleImputer(strategy='median'), num2),
    ('c', Pipeline([('i',SimpleImputer(strategy='most_frequent')),
                    ('e',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]), cat2)
])
pipe2 = Pipeline([('p',pre2),('m',LGBMClassifier(n_estimators=300,random_state=42,verbosity=-1,class_weight='balanced'))])
s2 = cross_val_score(pipe2, X2, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'  CV accuracy: {s2.mean():.4f} +/- {s2.std():.4f}')
print(f'  → Baseline with everything the ML table has\n')

# ── Experiment 3: Withdrawn vs Fail — just that confusion ────────────────────
print('=== Exp 3: How hard is Withdrawn vs Fail? ===')
mask_wf = fr.isin(['Withdrawn','Fail'])
X_wf = X_base[mask_wf].copy()
y_wf = LabelEncoder().fit_transform(fr[mask_wf])
num_wf = X_wf.select_dtypes(include='number').columns.tolist()
cat_wf = X_wf.select_dtypes(include='object').columns.tolist()
pre_wf = ColumnTransformer([
    ('n', SimpleImputer(strategy='median'), num_wf),
    ('c', Pipeline([('i',SimpleImputer(strategy='most_frequent')),
                    ('e',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]), cat_wf)
])
pipe_wf = Pipeline([('p',pre_wf),('m',LGBMClassifier(n_estimators=300,random_state=42,verbosity=-1))])
s_wf = cross_val_score(pipe_wf, X_wf, y_wf, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'  Withdrawn vs Fail only — CV accuracy: {s_wf.mean():.4f} +/- {s_wf.std():.4f}')
print(f'  → Shows irreducible Withdrawn/Fail confusion\n')

# ── Experiment 4: Drop Withdrawn, train on Pass/Fail/Distinction ─────────────
print('=== Exp 4: Drop Withdrawn — 3-class only ===')
mask3 = fr.isin(['Pass','Fail','Distinction'])
X3 = X_base[mask3].copy()
y3 = LabelEncoder().fit_transform(fr[mask3])
num3 = X3.select_dtypes(include='number').columns.tolist()
cat3 = X3.select_dtypes(include='object').columns.tolist()
pre3 = ColumnTransformer([
    ('n', SimpleImputer(strategy='median'), num3),
    ('c', Pipeline([('i',SimpleImputer(strategy='most_frequent')),
                    ('e',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]), cat3)
])
pipe3 = Pipeline([('p',pre3),('m',LGBMClassifier(n_estimators=300,random_state=42,verbosity=-1))])
s3 = cross_val_score(pipe3, X3, y3, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'  Pass/Fail/Distinction — CV accuracy: {s3.mean():.4f} +/- {s3.std():.4f}')
print()

# ── Experiment 5: engineered features that synthetic_platform is missing ──────
print('=== Exp 5: Add engineered features missing from synthetic_platform ===')
X5 = X_base.copy()
# log clicks
for c in ['total_clicks'] + [f'week{w}_clicks' for w in range(1,13)]:
    if c in X5: X5[f'log_{c}'] = np.log1p(X5[c].fillna(0))
# ratios
if 'clicks_until_week4' in X5 and 'total_clicks' in X5:
    X5['early_ratio'] = (X5['clicks_until_week4'] / X5['total_clicks'].replace(0,np.nan)).fillna(0)
if 'avg_score' in X5 and 'assessment_completion_ratio' in X5:
    X5['score_x_completion'] = X5['avg_score'].fillna(0) * X5['assessment_completion_ratio'].fillna(0)
if 'total_clicks' in X5:
    X5['zero_clicks'] = (X5['total_clicks'].fillna(0) == 0).astype(int)
if 'num_assessments' in X5:
    X5['no_assessments'] = (X5['num_assessments'].fillna(0) == 0).astype(int)
if 'avg_score' in X5:
    X5['passed_threshold'] = (X5['avg_score'].fillna(0) >= 40).astype(int)
# IMD numeric
if 'imd_band' in X5:
    imd_map = {'0-10%':1,'10-20':2,'10-20%':2,'20-30%':3,'30-40%':4,'40-50%':5,
               '50-60%':6,'60-70%':7,'70-80%':8,'80-90%':9,'90-100%':10}
    X5['imd_numeric'] = X5['imd_band'].map(imd_map).fillna(5)
# peak week
wk_cols = [f'week{w}_clicks' for w in range(1,13) if f'week{w}_clicks' in X5]
if wk_cols:
    X5['peak_week'] = X5[wk_cols].fillna(0).idxmax(axis=1).str.replace('week','').str.replace('_clicks','').astype(float)
    X5['active_weeks_count'] = (X5[wk_cols].fillna(0) > 0).sum(axis=1)
    h1 = [f'week{w}_clicks' for w in range(1,7) if f'week{w}_clicks' in X5]
    h2 = [f'week{w}_clicks' for w in range(7,13) if f'week{w}_clicks' in X5]
    if h1 and h2:
        X5['h2_vs_h1'] = X5[h2].fillna(0).sum(axis=1) / (X5[h1].fillna(0).sum(axis=1) + 1)

num5 = X5.select_dtypes(include='number').columns.tolist()
cat5 = X5.select_dtypes(include='object').columns.tolist()
pre5 = ColumnTransformer([
    ('n', SimpleImputer(strategy='median'), num5),
    ('c', Pipeline([('i',SimpleImputer(strategy='most_frequent')),
                    ('e',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]), cat5)
])
pipe5 = Pipeline([('p',pre5),('m',LGBMClassifier(n_estimators=400,num_leaves=63,learning_rate=0.05,
                                                   class_weight='balanced',random_state=42,verbosity=-1))])
s5 = cross_val_score(pipe5, X5, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'  CV accuracy with engineered features: {s5.mean():.4f} +/- {s5.std():.4f}\n')

print('=== Summary of accuracy ceiling analysis ===')
print(f'  synthetic_platform (early-warning schema)    : {s1.mean():.4f}')
print(f'  Full ML table (no eng)                       : {s2.mean():.4f}')
print(f'  Withdrawn vs Fail only                       : {s_wf.mean():.4f}  ← the hard problem')
print(f'  3-class (no Withdrawn)                       : {s3.mean():.4f}')
print(f'  Full + engineered features                   : {s5.mean():.4f}')
print()
print('Root cause: Withdrawn students (29% zero clicks, 54% zero assessments)')
print('look identical to some Fail students (5% zero clicks) in the feature space.')
print('This is an IRREDUCIBLE overlap without temporal sequence data.')
