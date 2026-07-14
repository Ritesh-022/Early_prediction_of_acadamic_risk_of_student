#!/usr/bin/env python3
"""Diagnose why 4-class OULAD caps at 75% and what can push it higher."""
import pandas as pd
import numpy as np

df = pd.read_csv('oulad_ml_table.csv')
print('=== OULAD ML Table ===')
print('Shape:', df.shape)

print('\nTarget distribution:')
print(df['final_result'].value_counts())
print()

tc = df['total_clicks'].fillna(0)
na = df['num_assessments'].fillna(0)
fr = df['final_result']

print('Zero clicks:', (tc == 0).sum(), '({:.1f}%)'.format((tc==0).mean()*100))
print('Zero assessments:', (na == 0).sum(), '({:.1f}%)'.format((na==0).mean()*100))
print()

print('avg_score missing:', df['avg_score'].isna().sum(), '({:.1f}%)'.format(df['avg_score'].isna().mean()*100))
print()

# Separability: withdrawn vs fail without leakage
print('=== Withdrawn vs Fail separability ===')
for label in ['Withdrawn', 'Fail', 'Pass', 'Distinction']:
    mask = fr == label
    zero_c = (tc[mask] == 0).sum()
    zero_a = (na[mask] == 0).sum()
    n = mask.sum()
    print(f'{label}: n={n}  zero_clicks={zero_c}({zero_c/n*100:.0f}%)  zero_assessments={zero_a}({zero_a/n*100:.0f}%)')

print()
# Overlap between Fail and Withdrawn in click space
w = df[fr == 'Withdrawn']['total_clicks'].fillna(0)
f = df[fr == 'Fail']['total_clicks'].fillna(0)
print('Withdrawn clicks — median:', w.median(), ' mean:', round(w.mean(),1), ' max:', w.max())
print('Fail clicks      — median:', f.median(), ' mean:', round(f.mean(),1), ' max:', f.max())

print()
# Avg score overlap
w_s = df[fr == 'Withdrawn']['avg_score'].fillna(0)
f_s = df[fr == 'Fail']['avg_score'].fillna(0)
print('Withdrawn avg_score — median:', w_s.median(), ' mean:', round(w_s.mean(),1))
print('Fail avg_score      — median:', f_s.median(), ' mean:', round(f_s.mean(),1))

print()
# Assessment completion
w_ac = df[fr == 'Withdrawn']['assessment_completion_ratio'].fillna(0)
f_ac = df[fr == 'Fail']['assessment_completion_ratio'].fillna(0)
print('Withdrawn assessment_completion — median:', w_ac.median(), ' mean:', round(w_ac.mean(),2))
print('Fail assessment_completion      — median:', f_ac.median(), ' mean:', round(f_ac.mean(),2))

print()
print('=== Feature coverage per class ===')
key_features = ['total_clicks', 'avg_score', 'assessment_completion_ratio',
                'inactivity_days', 'studied_credits', 'num_of_prev_attempts']
for feat in key_features:
    if feat not in df.columns:
        continue
    print(f'\n{feat}:')
    for label in ['Distinction', 'Pass', 'Fail', 'Withdrawn']:
        subset = df[fr == label][feat].fillna(0)
        print(f'  {label:<12}: mean={subset.mean():.1f}  median={subset.median():.1f}  std={subset.std():.1f}')

print()
print('=== Missing data per class (avg_score) ===')
for label in ['Distinction', 'Pass', 'Fail', 'Withdrawn']:
    mask = fr == label
    miss = df[mask]['avg_score'].isna().mean() * 100
    print(f'  {label:<12}: {miss:.1f}% missing avg_score')

print()
print('=== synthetic_platform confusion: source leakage? ===')
# In synthetic_platform the "source" column is in top SHAP — check its predictive power
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df['final_result'])
    # just use total_clicks vs zero to estimate Withdrawn/Fail boundary
    X_simple = df[['total_clicks', 'avg_score', 'assessment_completion_ratio']].fillna(0)
    from sklearn.model_selection import cross_val_score
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    scores = cross_val_score(dt, X_simple, y, cv=5, scoring='accuracy')
    print('3-feature DecisionTree CV accuracy:', round(scores.mean(), 4))
except Exception as e:
    print('Skipped:', e)
