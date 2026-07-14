#!/usr/bin/env python3
"""Simple baseline training script for the OULAD ML table

Reads the `oulad_ml_table.csv` produced by `oulad_pipeline.py`, does lightweight preprocessing,
trains a RandomForest baseline (scikit-learn) and prints evaluation metrics.

Usage:
    python oulad_baseline.py --input oulad_ml_table.csv --target final_result
"""
from pathlib import Path
import argparse
import joblib
import logging
import random

import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix, classification_report
try:
    import shap
except Exception:
    shap = None
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


# Features only observable at or near end-of-course.
# Highly predictive but NOT safe for early-warning prediction.
LATE_FEATURES = {
    'assessment_completion_ratio',
    'last_ts',
    'last_assessment_day',
    'assessment_span_days',
    'inactivity_days',
    'num_assessments',
    'missed_assessments',
    'total_assessments',
    'week_click_sum_1_12',   # sum of all 12 weeks = full-semester aggregate
    'click_growth_rate',
    'click_variance',
    'longest_inactive_gap',
    'avg_score',             # mean over all submitted assessments — end-of-course
    'score_std',             # std over all submitted assessments — end-of-course
    'assessment_score_trend',# trend over all submitted assessments — end-of-course
}

# Derived slope/trend features — legitimately negative; exclude from impossible-value checks.
TREND_FEATURES = {'click_growth_rate', 'assessment_score_trend', 'score_std'}

# Composite unique key for OULAD — id_student alone is NOT unique across modules/presentations.
OULAD_COMPOSITE_KEY = ['id_student', 'code_module', 'code_presentation']


def classify_features(feature_names):
    """Classify features into KEEP (early/safe), OPTIONAL (mid-course), DROP (late/end-of-course)."""
    keep, optional, drop = [], [], []
    for f in feature_names:
        base = f.split('__')[-1]
        if base in LATE_FEATURES:
            drop.append(f)
        elif base.startswith('week') and '_clicks' in base:
            try:
                week_num = int(base.replace('week', '').replace('_clicks', ''))
                (keep if week_num <= 4 else optional).append(f)
            except ValueError:
                optional.append(f)
        else:
            keep.append(f)
    return keep, optional, drop


def simple_preprocess(df: pd.DataFrame, target_col: str, mode: str = 'benchmark'):
    """
    mode='benchmark'     : all features included (full-course evaluation).
    mode='early-warning' : late/end-of-course features dropped before training.
    """
    df = df.copy()
    df = df.dropna(axis=1, how='all')
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in input")

    for col in ['total_clicks', 'activity_count', 'num_assessments', 'days_active', 'avg_clicks_per_day', 'inactivity_days']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Hard leakage: unregistration date directly reveals withdrawal outcome
    for leak in ['date_unregistration', 'date_unreg', 'date_unregistered', 'weighted_score']:
        if leak in df.columns:
            df = df.drop(columns=[leak])

    # Redundant correlated columns dropped in pipeline; guard here too
    for redundant in ['active_weeks', 'clicks_per_active_week', 'assessments_per_week',
                      'activity_count', 'days_active', 'avg_clicks_per_day',
                      'week_click_sum_1_4', 'registration_delay_category']:
        if redundant in df.columns:
            df = df.drop(columns=[redundant])

    dropped_late = []
    if mode == 'early-warning':
        dropped_late = [c for c in df.columns if c in LATE_FEATURES]
        if dropped_late:
            df = df.drop(columns=dropped_late)

    drop_cols = ['id_student', 'id_assessment', 'id_site', 'first_ts', 'last_ts']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]
    mask = y.notnull()
    return X.loc[mask], y.loc[mask], dropped_late


def build_preprocessor(model_name: str, numeric_cols, categorical_cols):
    transformers = []
    numeric_steps = [('impute', SimpleImputer(strategy='median'))]
    if model_name == 'logistic_regression':
        numeric_steps.append(('scale', StandardScaler()))
    if numeric_cols:
        transformers.append(('num', Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        )
    return ColumnTransformer(transformers, remainder='drop')


def get_param_distributions(model_name: str):
    if model_name == 'random_forest':
        return {
            'clf__n_estimators': randint(100, 500),
            'clf__max_depth': [None, 10, 20, 30, 40],
            'clf__min_samples_split': randint(2, 10),
            'clf__min_samples_leaf': randint(1, 8),
            'clf__max_features': ['sqrt', 'log2', None],
        }
    if model_name == 'xgboost':
        return {
            'clf__learning_rate': uniform(0.01, 0.29),
            'clf__max_depth': randint(3, 12),
            'clf__subsample': uniform(0.5, 0.5),
            'clf__colsample_bytree': uniform(0.5, 0.5),
            'clf__gamma': uniform(0, 5),
            'clf__min_child_weight': uniform(1, 10),
            'clf__reg_alpha': uniform(0, 1),
            'clf__reg_lambda': uniform(0, 1),
        }
    if model_name == 'lightgbm':
        return {
            'clf__num_leaves': randint(20, 128),
            'clf__learning_rate': uniform(0.01, 0.29),
            'clf__max_depth': [None, 10, 20, 30],
            'clf__feature_fraction': uniform(0.5, 0.5),
            'clf__min_child_samples': randint(5, 50),
            'clf__subsample': uniform(0.5, 0.5),
            'clf__colsample_bytree': uniform(0.5, 0.5),
            'clf__reg_alpha': uniform(0, 1),
            'clf__reg_lambda': uniform(0, 1),
        }
    if model_name == 'catboost':
        return {
            'clf__depth': randint(4, 10),
            'clf__learning_rate': uniform(0.01, 0.29),
            'clf__l2_leaf_reg': uniform(1, 10),
            'clf__bagging_temperature': uniform(0, 1),
            'clf__random_strength': uniform(0, 1),
            'clf__border_count': randint(32, 128),
            'clf__grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
            'clf__iterations': randint(100, 500),
        }
    return None


def tune_pipeline(pipe: Pipeline, model_name: str, X, y, cv, n_iter: int, n_jobs: int, random_state: int, verbose: bool):
    param_distributions = get_param_distributions(model_name)
    if not param_distributions:
        return pipe, None
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='f1_macro',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=2 if verbose else 0,
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def print_feature_importance(pipe: Pipeline, feature_names, top_n: int = 20):
    clf = pipe.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        coef = clf.coef_
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)
    else:
        return None

    if len(importances) != len(feature_names):
        feature_names = [f'feature_{i}' for i in range(len(importances))]

    order = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], float(importances[i])) for i in order]


def clean_feature_names(feature_names):
    cleaned = []
    for name in feature_names:
        if isinstance(name, str):
            cleaned.append(
                name.replace('cat__', '')
                    .replace('num__', '')
                    .replace('preprocessor__', '')
            )
        else:
            cleaned.append(name)
    return cleaned


def compute_shap_importance(pipe: Pipeline, X, feature_names, top_n: int = 20):
    if shap is None:
        return None
    clf = pipe.named_steps['clf']
    try:
        X_transformed = pipe.named_steps['preprocessor'].transform(X)
    except Exception:
        return None
    try:
        tree_classes = tuple(c for c in (RandomForestClassifier, DecisionTreeClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier) if c is not None)
        if isinstance(clf, tree_classes):
            explainer = shap.TreeExplainer(clf)
        else:
            explainer = shap.Explainer(clf, X_transformed, feature_names=feature_names)
        shap_values = explainer(X_transformed)
        if hasattr(shap_values, 'values'):
            values = shap_values.values
        elif isinstance(shap_values, np.ndarray):
            values = shap_values
        elif isinstance(shap_values, (list, tuple)) and len(shap_values) > 0:
            if hasattr(shap_values[0], 'values'):
                values = np.array([sv.values for sv in shap_values])
            else:
                values = np.array(shap_values)
        else:
            return None

        abs_values = np.abs(values)
        if abs_values.ndim == 3:
            if abs_values.shape[-1] == len(feature_names):
                importances = np.mean(abs_values, axis=(0, 1))
            else:
                importances = np.mean(abs_values, axis=(0, 2))
        elif abs_values.ndim == 2:
            importances = np.mean(abs_values, axis=0)
        else:
            importances = np.mean(abs_values, axis=0)

        order = np.argsort(importances)[::-1][:top_n]
        return [(feature_names[i], float(importances[i])) for i in order]
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Train a simple baseline on oulad_ml_table.csv')
    parser.add_argument('--input', default='oulad_ml_table.csv')
    parser.add_argument('--target', default='final_result')
    parser.add_argument(
        '--model', default='rf',
        help='Model to use: rf, logreg, dt, xgb, lgb, catboost, compare, or all'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of the data held out for final testing')
    parser.add_argument('--cv-splits', type=int, default=5, help='Maximum number of stratified folds for cross-validation')
    parser.add_argument('--cv-repeats', type=int, default=2, help='Number of repeats for repeated stratified CV')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs for training and tuning')
    parser.add_argument('--tune', action='store_true', help='Perform randomized hyperparameter search')
    parser.add_argument('--tune-iters', type=int, default=25, help='Number of iterations for randomized hyperparameter search')
    parser.add_argument('--class-weight', choices=['none', 'balanced'], default='none', help='Class weight strategy for supported models')
    parser.add_argument('--shap-sample', type=int, default=500, help='Sample size for SHAP explanations')
    parser.add_argument('--verbose', action='store_true', help='Verbose output for tuning and progress')
    parser.add_argument('--output-model', default=None, help='Optional path to save the final model pipeline as a .pkl file')
    parser.add_argument('--output-encoder', default=None, help='Optional path to save the fitted LabelEncoder as a .pkl file')
    parser.add_argument('--output-dir', default=None, help='Optional directory to save feature importance and SHAP importance CSVs')
    parser.add_argument(
        '--mode', choices=['benchmark', 'early-warning'], default='benchmark',
        help='benchmark: all features (full-course). early-warning: drop late/end-of-course features.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    np.random.seed(args.seed)
    random.seed(args.seed)

    p = Path(args.input)
    if not p.exists():
        parser.error(f"Input file not found: {p}")

    df = pd.read_csv(p)
    df = df.drop_duplicates()

    mode = args.mode
    try:
        X, y, dropped_late = simple_preprocess(df, args.target, mode=mode)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return

    if len(X) < 10:
        logger.error('Not enough rows after preprocessing to train.')
        return

    logger.info('=' * 60)
    if mode == 'benchmark':
        logger.info('MODE: benchmark  (all features — full-course evaluation)')
        logger.info('  Late/end-of-course features ARE included.')
        logger.info('  Expected accuracy: 74-80%%  |  NOT suitable for deployment.')
    else:
        logger.info('MODE: early-warning  (late features removed)')
        logger.info('  Dropped %d late features: %s', len(dropped_late), dropped_late)
        logger.info('  Expected accuracy: 55-65%%  |  Realistic prediction setting.')
    logger.info('=' * 60)

    logger.info('\nClass distribution: %s', y.value_counts().to_dict())

    # Feature safety report
    keep_feats, optional_feats, drop_feats = classify_features(X.columns.tolist())
    logger.info('\nFeature safety report:')
    logger.info('  KEEP     (early/safe)    : %d', len(keep_feats))
    logger.info('  OPTIONAL (weeks 5-12)    : %d', len(optional_feats))
    logger.info('  DROP     (end-of-course) : %d  %s',
                len(drop_feats), drop_feats if drop_feats else '(none — clean)')

    # Composite key note
    composite_present = [c for c in OULAD_COMPOSITE_KEY if c in df.columns]
    if len(composite_present) == 3:
        logger.info('\nComposite key: (id_student, code_module, code_presentation)')

    # Encode string class labels to numeric targets for all estimators
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y), index=y.index)

    # Build preprocessing transformer based on model type
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    class_weight = 'balanced' if args.class_weight == 'balanced' else None
    models = {
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=args.n_jobs, class_weight=class_weight),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=args.seed, class_weight=class_weight),
        'decision_tree': DecisionTreeClassifier(random_state=args.seed, class_weight=class_weight),
    }
    if XGBClassifier is not None:
        xgb_kwargs = {'use_label_encoder': False, 'eval_metric': 'mlogloss', 'random_state': args.seed, 'n_jobs': args.n_jobs}
        models['xgboost'] = XGBClassifier(**xgb_kwargs)
    if LGBMClassifier is not None:
        models['lightgbm'] = LGBMClassifier(random_state=args.seed, n_jobs=args.n_jobs, class_weight=class_weight, verbosity=-1)
    if CatBoostClassifier is not None:
        cat_kwargs = {'verbose': 0, 'random_state': args.seed}
        if class_weight == 'balanced':
            cat_kwargs['auto_class_weights'] = 'Balanced'
        models['catboost'] = CatBoostClassifier(**cat_kwargs)

    aliases = {
        'rf': 'random_forest',
        'randomforest': 'random_forest',
        'logreg': 'logistic_regression',
        'dt': 'decision_tree',
        'xgb': 'xgboost',
        'lgb': 'lightgbm',
        'cb': 'catboost',
        'all': 'compare',
    }
    model_name = args.model.lower()
    if model_name in aliases:
        model_name = aliases[model_name]
    if model_name == 'compare':
        selected_models = models
    else:
        selected_models = {model_name: models[model_name]} if model_name in models else None
        if selected_models is None:
            if model_name == 'catboost' and CatBoostClassifier is None:
                print('CatBoost is not installed. Install it with `pip install catboost` and rerun.')
            else:
                print(f"Model {args.model} is not available. Available: {', '.join(models.keys())}")
            return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    min_class_size = y_train.value_counts().min()
    cv_splits = max(2, min(args.cv_splits, min_class_size))
    if cv_splits < 2:
        parser.error('Need at least 2 members in each class for stratified cross-validation.')

    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=args.cv_repeats, random_state=args.seed)
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
        'balanced_accuracy': 'balanced_accuracy',
    }

    for name, model in selected_models.items():
        preprocessor = build_preprocessor(name, numeric_cols, categorical_cols)
        pipe = Pipeline([('preprocessor', preprocessor), ('clf', model)])
        logger.info(f'\nEvaluating {name}')
        results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=args.n_jobs, error_score='raise')
        for metric, vals in results.items():
            if metric.startswith('test_'):
                logger.info(f'  {metric[5:]:<15}: {vals.mean():.4f} ± {vals.std():.4f}')

        if args.tune:
            pipe, best_params = tune_pipeline(pipe, name, X_train, y_train, cv, n_iter=args.tune_iters, n_jobs=args.n_jobs, random_state=args.seed, verbose=args.verbose)
            if best_params is not None:
                logger.info('  Best params: %s', best_params)

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        try:
            probas = pipe.predict_proba(X_test)
        except Exception:
            probas = None

        logger.info('  Final test accuracy: %0.4f', accuracy_score(y_test, preds))
        logger.info('  Final test f1_macro: %0.4f', f1_score(y_test, preds, average='macro'))
        logger.info('  Final test f1_weighted: %0.4f', f1_score(y_test, preds, average='weighted'))
        logger.info('  Balanced accuracy: %0.4f', balanced_accuracy_score(y_test, preds))
        logger.info('  Cohen kappa: %0.4f', cohen_kappa_score(y_test, preds))
        logger.info('  MCC: %0.4f', matthews_corrcoef(y_test, preds))
        logger.info('  Confusion matrix:')
        logger.info('\n%s', confusion_matrix(y_test, preds))
        logger.info('  Classification Report:')
        logger.info(
            '\n%s',
            classification_report(y_test, preds, target_names=label_encoder.classes_)
        )

        if probas is not None:
            try:
                from sklearn.preprocessing import label_binarize
                classes = np.unique(y_test)
                y_bin = label_binarize(y_test, classes=classes)
                from sklearn.metrics import roc_auc_score
                roc_auc_macro = roc_auc_score(y_bin, probas, average='macro', multi_class='ovr')
                roc_auc_per_class = roc_auc_score(y_bin, probas, average=None, multi_class='ovr')
                logger.info('  ROC AUC OVR: %0.4f', roc_auc_macro)
                for cls_name, cls_auc in zip(label_encoder.classes_, roc_auc_per_class):
                    logger.info('    ROC AUC %s: %0.4f', cls_name, cls_auc)
            except Exception:
                logger.info('  ROC AUC OVR: not available for this model or fold configuration')

        if args.output_model is not None:
            output_path = Path(args.output_model)
            if model_name == 'compare':
                output_path = output_path.parent / f"{output_path.stem}_{name}.pkl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, output_path)
            logger.info(f'  Saved model pipeline to {output_path}')

        if args.output_encoder is not None:
            encoder_path = Path(args.output_encoder)
            if model_name == 'compare':
                encoder_path = encoder_path.parent / f"{encoder_path.stem}_{name}.pkl"
            encoder_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(label_encoder, encoder_path)
            logger.info(f'  Saved label encoder to {encoder_path}')

        feature_names = []
        try:
            feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()
        except Exception:
            feature_names = [f'feature_{i}' for i in range(pipe.named_steps['preprocessor'].transform(X_train).shape[1])]
        feature_names = clean_feature_names(feature_names)

        importance = print_feature_importance(pipe, feature_names)
        if importance is not None:
            logger.info('  Top feature importances:')
            for feat, val in importance[:20]:
                logger.info('    %s: %0.6f', feat, val)

        if shap is not None:
            X_shap = X_test.sample(min(len(X_test), args.shap_sample), random_state=args.seed)
            shap_imp = compute_shap_importance(pipe, X_shap, feature_names)
            if shap_imp is not None:
                logger.info('  Top SHAP feature importances:')
                for feat, val in shap_imp[:20]:
                    logger.info('    %s: %0.6f', feat, val)
                if args.output_dir is not None:
                    out_dir = Path(args.output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    shap_path = out_dir / f'shap_importance_{name}.csv'
                    pd.DataFrame(shap_imp, columns=['feature', 'importance']).to_csv(shap_path, index=False)
                    logger.info('  Saved SHAP importance to %s', shap_path)
            else:
                logger.info('  SHAP explanations not available for this model.')

        if importance is not None and args.output_dir is not None:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            importance_path = out_dir / f'feature_importance_{name}.csv'
            pd.DataFrame(importance, columns=['feature', 'importance']).to_csv(importance_path, index=False)
            logger.info('  Saved feature importance to %s', importance_path)

    logger.info('\nDone.')


if __name__ == '__main__':
    main()
