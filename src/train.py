"""
=============================================================
src/train.py — Étape 2 du pipeline DVC
=============================================================
Entraînement des modèles de classification :
  - Vectorisation TF-IDF
  - Gestion du déséquilibre (class_weight)
  - Entraînement de plusieurs modèles
  - Sauvegarde du meilleur modèle

Usage :
    python src/train.py --input data/tweets_suspect_preprocessed.csv \
                        --model-dir models/ \
                        --params params.yaml
=============================================================
"""

import argparse
import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)


def load_params(params_file: str) -> dict:
    with open(params_file, 'r') as f:
        return yaml.safe_load(f)


def build_models(params: dict) -> dict:
    """Construit les pipelines de modèles à entraîner."""
    tfidf_params = params.get('vectorizer', {})

    tfidf = TfidfVectorizer(
        max_features  = tfidf_params.get('max_features', 10000),
        ngram_range   = tuple(tfidf_params.get('ngram_range', [1, 2])),
        sublinear_tf  = tfidf_params.get('sublinear_tf', True),
        min_df        = tfidf_params.get('min_df', 2),
    )

    train_params = params.get('train', {})
    cw = train_params.get('class_weight', 'balanced')

    models = {
        'LogisticRegression': Pipeline([
            ('tfidf', tfidf),
            ('clf',   LogisticRegression(
                C=train_params.get('lr_C', 1.0),
                max_iter=1000,
                class_weight=cw,
                random_state=train_params.get('random_state', 42)
            ))
        ]),
        'NaiveBayes': Pipeline([
            ('tfidf',  tfidf),
            ('scaler', MaxAbsScaler()),
            ('clf',    MultinomialNB(alpha=train_params.get('nb_alpha', 0.1)))
        ]),
        'LinearSVC': Pipeline([
            ('tfidf', tfidf),
            ('clf',   LinearSVC(
                C=train_params.get('svc_C', 1.0),
                class_weight=cw,
                max_iter=2000,
                random_state=train_params.get('random_state', 42)
            ))
        ]),
        'RandomForest': Pipeline([
            ('tfidf', tfidf),
            ('clf',   RandomForestClassifier(
                n_estimators=train_params.get('rf_n_estimators', 100),
                class_weight=cw,
                random_state=train_params.get('random_state', 42),
                n_jobs=-1
            ))
        ]),
    }
    return models


def main(input_path: str, model_dir: str, params_file: str,
         metrics_path: str = 'metrics/train_metrics.json'):

    print("=" * 60)
    print("  ÉTAPE 2 — ENTRAÎNEMENT DES MODÈLES")
    print("=" * 60)

    params = load_params(params_file)
    train_params = params.get('train', {})
    test_size    = train_params.get('test_size', 0.2)
    random_state = train_params.get('random_state', 42)
    cv_folds     = train_params.get('cv_folds', 5)

    # ── Chargement ───────────────────────────────────────────────
    print(f"\n  Chargement de : {input_path}")
    df = pd.read_csv(input_path)
    df['message_clean'] = df['message_clean'].fillna('')
    X = df['message_clean']
    y = df['label']
    print(f"  → {len(df)} tweets | Classes : {dict(y.value_counts())}")

    # ── Split train/test ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n  Split : {len(X_train)} train / {len(X_test)} test (stratifié)")

    # ── Entraînement et évaluation ───────────────────────────────
    models = build_models(params)
    results = {}
    best_f1  = -1
    best_name = None

    print(f"\n  Validation croisée ({cv_folds} folds) + évaluation sur test :\n")
    print(f"  {'Modèle':<22} {'CV F1':>8} {'Test F1':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8}")
    print("  " + "-" * 64)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for name, pipeline in models.items():
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=skf, scoring='f1', n_jobs=-1)
        # Entraînement final
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        cv_f1 = cv_scores.mean()

        results[name] = {
            'cv_f1_mean': round(cv_f1, 4),
            'cv_f1_std' : round(cv_scores.std(), 4),
            'accuracy'  : round(acc,  4),
            'precision' : round(prec, 4),
            'recall'    : round(rec,  4),
            'f1_score'  : round(f1,   4),
        }

        print(f"  {name:<22} {cv_f1:>8.4f} {f1:>8.4f} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f}")

        if f1 > best_f1:
            best_f1   = f1
            best_name = name
            best_pipeline = pipeline

    print("  " + "-" * 64)
    print(f"\n  Meilleur modèle : {best_name} (F1 = {best_f1:.4f})")

    # ── Rapport de classification du meilleur modèle ─────────────
    y_pred_best = best_pipeline.predict(X_test)
    print(f"\n  Rapport de classification — {best_name} :")
    print(classification_report(y_test, y_pred_best,
                                target_names=['Normal', 'Suspect']))

    # ── Sauvegarde du meilleur modèle ────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': best_pipeline, 'name': best_name}, f)
    print(f"   Meilleur modèle sauvegardé : {model_path}")

    # Sauvegarde de tous les modèles
    all_models_path = os.path.join(model_dir, 'all_models.pkl')
    trained_models = {name: pipeline for name, pipeline in models.items()}
    with open(all_models_path, 'wb') as f:
        pickle.dump(trained_models, f)

    # Sauvegarde du split test pour l'étape d'évaluation
    test_data_path = os.path.join(model_dir, 'test_data.pkl')
    with open(test_data_path, 'wb') as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)

    # ── Métriques ────────────────────────────────────────────────
    metrics = {
        'best_model'    : best_name,
        'best_f1_score' : round(best_f1, 4),
        'train_size'    : int(len(X_train)),
        'test_size'     : int(len(X_test)),
        'cv_folds'      : cv_folds,
        'models'        : results
    }
    os.makedirs('metrics', exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Métriques sauvegardées : {metrics_path}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entraînement des modèles')
    parser.add_argument('--input',      default='data/tweets_suspect_preprocessed.csv')
    parser.add_argument('--model-dir',  default='models/')
    parser.add_argument('--params',     default='params.yaml')
    parser.add_argument('--metrics',    default='metrics/train_metrics.json')
    args = parser.parse_args()

    main(args.input, args.model_dir, args.params, args.metrics)
