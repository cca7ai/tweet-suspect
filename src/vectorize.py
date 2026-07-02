"""
=============================================================
src/vectorize.py — Étape 1.5 du pipeline DVC
=============================================================
Représentation numérique des tweets via plusieurs approches :
  - Bag of Words (CountVectorizer)
  - TF-IDF (unigrammes + bigrammes)
  - Word2Vec (vecteurs moyens par tweet)

Sauvegarde les matrices de features pour la phase train.

Usage :
    python src/vectorize.py --input data/tweets_preprocessed.csv
                             --output-dir data/
                             --params params.yaml
=============================================================
"""

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec


def load_params(params_file):
    with open(params_file) as f:
        return yaml.safe_load(f)


# ── Bag of Words ─────────────────────────────────────────────────────────────
def build_bow(corpus_train, corpus_all, params):
    print("  [BoW] Entraînement CountVectorizer...")
    bow_params = params.get('bow', {})
    vec = CountVectorizer(
        max_features = bow_params.get('max_features', 5000),
        ngram_range  = tuple(bow_params.get('ngram_range', [1, 1])),
        min_df       = bow_params.get('min_df', 2),
    )
    vec.fit(corpus_train)
    X = vec.transform(corpus_all)
    print(f"  [BoW] Matrice : {X.shape} | Vocabulaire : {len(vec.vocabulary_)}")
    return vec, X


# ── TF-IDF ───────────────────────────────────────────────────────────────────
def build_tfidf(corpus_train, corpus_all, params):
    print("  [TF-IDF] Entraînement TfidfVectorizer...")
    tfidf_params = params.get('vectorizer', {})
    vec = TfidfVectorizer(
        max_features = tfidf_params.get('max_features', 10000),
        ngram_range  = tuple(tfidf_params.get('ngram_range', [1, 2])),
        sublinear_tf = tfidf_params.get('sublinear_tf', True),
        min_df       = tfidf_params.get('min_df', 2),
    )
    vec.fit(corpus_train)
    X = vec.transform(corpus_all)
    print(f"  [TF-IDF] Matrice : {X.shape} | Vocabulaire : {len(vec.vocabulary_)}")
    return vec, X


# ── Word2Vec ─────────────────────────────────────────────────────────────────
def build_word2vec(corpus_train, corpus_all, params):
    print("  [Word2Vec] Entraînement du modèle...")
    w2v_params = params.get('word2vec', {})
    vector_size = w2v_params.get('vector_size', 100)
    window      = w2v_params.get('window', 5)
    min_count   = w2v_params.get('min_count', 2)
    epochs      = w2v_params.get('epochs', 20)
    sg          = w2v_params.get('sg', 1)   # 1=Skip-gram, 0=CBOW

    # Tokenisation
    sentences_train = [text.split() for text in corpus_train]
    sentences_all   = [text.split() for text in corpus_all]

    # Entraînement
    model = Word2Vec(
        sentences   = sentences_train,
        vector_size = vector_size,
        window      = window,
        min_count   = min_count,
        epochs      = epochs,
        sg          = sg,
        workers     = 4,
        seed        = 42
    )

    vocab_size = len(model.wv)
    print(f"  [Word2Vec] Vocabulaire appris : {vocab_size} mots | dim={vector_size}")

    # Vecteur moyen par tweet (mean pooling)
    def tweet_vector(tokens):
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if not vecs:
            return np.zeros(vector_size)
        return np.mean(vecs, axis=0)

    X = np.vstack([tweet_vector(tokens) for tokens in sentences_all])
    print(f"  [Word2Vec] Matrice finale : {X.shape}")
    return model, X


def main(input_path, output_dir, params_file,
         metrics_path='metrics/vectorize_metrics.json'):

    print("=" * 60)
    print("  ÉTAPE 1.5 — REPRÉSENTATION NUMÉRIQUE DES DONNÉES")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    params = load_params(params_file)
    random_state = params.get('train', {}).get('random_state', 42)

    # ── Chargement ───────────────────────────────────────────────
    df = pd.read_csv(input_path)
    df['message_clean'] = df['message_clean'].fillna('')
    corpus = df['message_clean'].tolist()
    labels = df['label'].values
    print(f"\n  {len(corpus)} tweets chargés\n")

    # Split train pour fit des vectoriseurs (éviter le data leakage)
    from sklearn.model_selection import train_test_split
    idx = list(range(len(corpus)))
    idx_train, _ = train_test_split(idx, test_size=0.2,
                                    random_state=random_state, stratify=labels)
    corpus_train = [corpus[i] for i in idx_train]

    # ── BoW ──────────────────────────────────────────────────────
    bow_vec, X_bow = build_bow(corpus_train, corpus, params)

    # ── TF-IDF ───────────────────────────────────────────────────
    tfidf_vec, X_tfidf = build_tfidf(corpus_train, corpus, params)

    # ── Word2Vec ─────────────────────────────────────────────────
    w2v_model, X_w2v = build_word2vec(corpus_train, corpus, params)

    # ── Sauvegarde ───────────────────────────────────────────────
    print("\n  Sauvegarde des artefacts...")

    artifacts = {
        'bow_vectorizer'   : bow_vec,
        'tfidf_vectorizer' : tfidf_vec,
        'w2v_model'        : w2v_model,
        'X_bow'            : X_bow,
        'X_tfidf'          : X_tfidf,
        'X_w2v'            : X_w2v,
        'labels'           : labels,
    }

    out_path = os.path.join(output_dir, 'vectorized_data.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(artifacts, f)

    w2v_model.save(os.path.join(output_dir, 'word2vec.model'))
    print(f"  Artefacts sauvegardés : {out_path}")

    # ── Métriques ────────────────────────────────────────────────
    metrics = {
        'bow'    : {'shape': list(X_bow.shape),   'vocab': len(bow_vec.vocabulary_)},
        'tfidf'  : {'shape': list(X_tfidf.shape), 'vocab': len(tfidf_vec.vocabulary_)},
        'word2vec': {
            'shape'     : list(X_w2v.shape),
            'vocab'     : len(w2v_model.wv),
            'vector_size': w2v_model.vector_size
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  Métriques : {metrics_path}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',      default='data/tweets_preprocessed.csv')
    parser.add_argument('--output-dir', default='data/')
    parser.add_argument('--params',     default='params.yaml')
    parser.add_argument('--metrics',    default='metrics/vectorize_metrics.json')
    args = parser.parse_args()
    main(args.input, args.output_dir, args.params, args.metrics)
