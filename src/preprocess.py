"""
=============================================================
src/preprocess.py — Étape 1 du pipeline DVC
=============================================================
Prétraitement des tweets bruts :
  - Nettoyage du texte (URLs, mentions, caractères spéciaux)
  - Suppression des stop words (avec préservation des négations)
  - Lemmatisation
  - Extraction de features textuelles
  - Sauvegarde du dataset nettoyé

Usage :
    python src/preprocess.py --input data/tweets.csv \
                              --output data/tweets_suspect_preprocessed.csv \
                              --params params.yaml
=============================================================
"""

import argparse
import os
import re
import json
import time
import yaml
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Téléchargement des ressources NLTK ──────────────────────────────────────
for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)


# ── Initialisation ───────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()

PRESERVE_NEGATIONS = {
    'no', 'not', 'never', 'nothing', 'nobody', 'nowhere',
    'none', 'nor', 'neither', 'cannot', "can't", "won't",
    'against', 'hate', 'bad', 'worst'
}


def load_params(params_file: str) -> dict:
    """Charge les paramètres depuis params.yaml."""
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params.get('preprocess', {})


def build_stopwords(language: str = 'english',
                    preserve_negations: bool = True) -> set:
    """Construit l'ensemble de stop words personnalisé."""
    sw = set(stopwords.words(language))
    if preserve_negations:
        sw -= PRESERVE_NEGATIONS
    return sw


def preprocess_tweet(text: str,
                     stop_words: set,
                     lemmatize: bool = True,
                     min_token_len: int = 2) -> str:
    """
    Pipeline de nettoyage d'un tweet.

    Étapes :
      1. Conversion en minuscules
      2. Suppression des URLs
      3. Suppression des mentions (@user)
      4. Nettoyage des hashtags (#mot → mot)
      5. Suppression des caractères spéciaux et chiffres
      6. Normalisation des espaces
      7. Filtrage des stop words + longueur minimale
      8. Lemmatisation (verbes + noms)
    """
    if pd.isna(text):
        return ''

    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    tokens = [t for t in tokens
              if t not in stop_words and len(t) > min_token_len]

    if lemmatize:
        tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
        tokens = [lemmatizer.lemmatize(t, pos='n') for t in tokens]

    return ' '.join(tokens)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les features textuelles brutes."""
    df = df.copy()
    df['tweet_length']      = df['message'].apply(len)
    df['word_count']        = df['message'].apply(lambda x: len(str(x).split()))
    df['char_count']        = df['message'].apply(lambda x: len(str(x).replace(' ', '')))
    df['hashtag_count']     = df['message'].apply(lambda x: len(re.findall(r'#\w+', str(x))))
    df['mention_count']     = df['message'].apply(lambda x: len(re.findall(r'@\w+', str(x))))
    df['url_count']         = df['message'].apply(lambda x: len(re.findall(r'http\S+|www\.\S+', str(x))))
    df['uppercase_ratio']   = df['message'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
    df['exclamation_count'] = df['message'].apply(lambda x: str(x).count('!'))
    df['question_count']    = df['message'].apply(lambda x: str(x).count('?'))
    return df


def main(input_path: str, output_path: str, params_file: str,
         metrics_path: str = 'metrics/preprocess_metrics.json'):

    print("=" * 60)
    print("  ÉTAPE 1 — PRÉTRAITEMENT DES DONNÉES")
    print("=" * 60)

    # ── Chargement des paramètres ────────────────────────────────
    params = load_params(params_file)
    language          = params.get('language', 'english')
    lemmatize         = params.get('lemmatize', True)
    preserve_neg      = params.get('preserve_negations', True)
    min_token_len     = params.get('min_token_length', 2)

    print(f"\n  Paramètres chargés depuis {params_file} :")
    print(f"    language          : {language}")
    print(f"    lemmatize         : {lemmatize}")
    print(f"    preserve_negations: {preserve_neg}")
    print(f"    min_token_length  : {min_token_len}")

    # ── Chargement des données ───────────────────────────────────
    print(f"\n  Chargement de : {input_path}")
    df = pd.read_csv(input_path)
    n_raw = len(df)
    print(f"  → {n_raw} tweets chargés")

    # ── Nettoyage initial ────────────────────────────────────────
    df = df.dropna(subset=['message']).drop_duplicates(subset='message').reset_index(drop=True)
    n_clean = len(df)
    print(f"  → {n_raw - n_clean} doublons/NaN supprimés → {n_clean} tweets restants")

    # ── Extraction des features brutes ───────────────────────────
    print("\n  Extraction des features textuelles...")
    df = extract_features(df)

    # ── Prétraitement du texte ───────────────────────────────────
    print("  Prétraitement du texte en cours...")
    stop_words = build_stopwords(language, preserve_neg)
    start = time.time()
    df['message_clean'] = df['message'].apply(
        lambda x: preprocess_tweet(x, stop_words, lemmatize, min_token_len)
    )
    elapsed = time.time() - start

    # Supprimer les tweets vides après nettoyage
    n_before_empty = len(df)
    df = df[df['message_clean'].str.strip() != ''].reset_index(drop=True)
    n_empty = n_before_empty - len(df)

    df['clean_word_count']   = df['message_clean'].apply(lambda x: len(str(x).split()))
    df['clean_tweet_length'] = df['message_clean'].apply(len)
    df['vocab_reduction']    = 1 - (df['clean_word_count'] / df['word_count'].replace(0, 1))

    print(f"  → Terminé en {elapsed:.2f}s ({n_empty} tweets vides supprimés)")

    # ── Sauvegarde ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    cols_to_save = [
        'message', 'message_clean', 'label',
        'tweet_length', 'word_count', 'clean_word_count',
        'hashtag_count', 'mention_count', 'url_count',
        'uppercase_ratio', 'exclamation_count', 'question_count'
    ]
    df[cols_to_save].to_csv(output_path, index=False)
    print(f"\n  Dataset prétraité sauvegardé : {output_path}")
    print(f"     {len(df)} tweets | {len(cols_to_save)} colonnes")

    # ── Métriques de prétraitement ───────────────────────────────
    vocab_raw   = set(' '.join(df['message']).lower().split())
    vocab_clean = set(' '.join(df['message_clean']).split())
    metrics = {
        "n_raw_tweets"       : int(n_raw),
        "n_processed_tweets" : int(len(df)),
        "n_removed_tweets"   : int(n_raw - len(df)),
        "vocab_raw_size"     : int(len(vocab_raw)),
        "vocab_clean_size"   : int(len(vocab_clean)),
        "vocab_reduction_pct": round((1 - len(vocab_clean)/len(vocab_raw)) * 100, 2),
        "avg_words_before"   : round(df['word_count'].mean(), 2),
        "avg_words_after"    : round(df['clean_word_count'].mean(), 2),
        "class_balance"      : {
            "normal" : int((df['label'] == 0).sum()),
            "suspect": int((df['label'] == 1).sum())
        },
        "processing_time_s"  : round(elapsed, 3)
    }

    os.makedirs('metrics', exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Métriques sauvegardées : {metrics_path}")
    print(f"     vocab_reduction : {metrics['vocab_reduction_pct']}%")
    print(f"     mots/tweet : {metrics['avg_words_before']} → {metrics['avg_words_after']}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prétraitement des tweets')
    parser.add_argument('--input',   default='data/tweets.csv',
                        help='Chemin du dataset brut')
    parser.add_argument('--output',  default='data/tweets_suspect_preprocessed.csv',
                        help='Chemin du dataset prétraité')
    parser.add_argument('--params',  default='params.yaml',
                        help='Fichier de paramètres YAML')
    parser.add_argument('--metrics', default='metrics/preprocess_metrics.json',
                        help='Fichier de sortie des métriques')
    args = parser.parse_args()

    main(args.input, args.output, args.params, args.metrics)
