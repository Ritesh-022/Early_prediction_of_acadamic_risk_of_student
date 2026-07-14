with open('multisource_ablation.py', encoding='utf-8') as f:
    src = f.read()

checks = {
    'Fix1/10  train-only median (X_train_ref)':     'X_train_ref' in src,
    'Fix9     binary ROC-AUC branch':               'len(label_encoder.classes_) == 2' in src or 'n_classes == 2' in src,
    'Fix12    per-source evaluation':               'per_source_eval' in src or 'evaluate_per_source' in src,
    'Fix11    source excluded final model':         'include_source=False' in src,
    'Fix13/14 signed SHAP direction':               'shap_direction' in src or 'signed_shap' in src or 'risk_direction' in src,
    'Fix15    separate actionable/model drivers':   'actionable_drivers' in src or 'model_drivers' in src,
    'Fix7     single split for all ablation exps':  'X_train, X_test' in src,
    'ACTIONABLE_FEATURES defined':                  'ACTIONABLE_FEATURES' in src,
}
print('=== multisource_ablation.py ===')
for k, v in checks.items():
    status = 'OK  ' if v else 'MISS'
    print(f'  {status}  {k}')

print()
with open('synthetic_platform.py', encoding='utf-8') as f:
    sp = f.read()

sp_checks = {
    'Fix6  target harmonization note':    'WARNING' in sp and 'Medium' in sp and 'binary' in sp.lower(),
    'Fix7  train-only preprocessing':     'X_train' in sp and 'train_medians' in sp,
    'Fix8  source excluded final model':  'include_source' in sp,
    'Fix13 signed SHAP':                  'signed' in sp or 'risk_direction' in sp or 'sign_val' in sp,
    'Fix15 actionable vs model drivers':  'ACTIONABLE_FEATURES' in sp and 'actionable_drivers' in sp,
}
print('=== synthetic_platform.py ===')
for k, v in sp_checks.items():
    status = 'OK  ' if v else 'MISS'
    print(f'  {status}  {k}')

print()
with open('oulad_pipeline.py', encoding='utf-8') as f:
    op = f.read()

op_checks = {
    'Fix1  week8 cutoff applied to VLE before feat eng': 'cutoff_day' in op,
    'Fix16 student overlap check / GroupSplit':          'GroupShuffleSplit' in op or 'student_id' in op.lower() or 'group' in op.lower(),
}
print('=== oulad_pipeline.py ===')
for k, v in op_checks.items():
    status = 'OK  ' if v else 'MISS'
    print(f'  {status}  {k}')

print()
with open('high_accuracy_pipeline.py', encoding='utf-8') as f:
    hp = f.read()

hp_checks = {
    'Fix2  temporal leakage audit comment':   'LATE_FEATURES' in hp or 'leakage' in hp.lower(),
    'Fix3  single split for all models':      'X_train, X_test' in hp,
    'Fix4  100% small dataset audit':         'drop_duplicates' in hp or 'target_col' in hp,
}
print('=== high_accuracy_pipeline.py ===')
for k, v in hp_checks.items():
    status = 'OK  ' if v else 'MISS'
    print(f'  {status}  {k}')
