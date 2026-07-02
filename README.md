# 🐦 Détection de Tweets Suspects — Pipeline ML avec DVC

> Projet d'examen final — Construction de Modèles et leur Déploiement

---

## 📋 Description

Ce projet implémente un pipeline complet de Machine Learning pour la **classification automatique de tweets suspects**, en intégrant les bonnes pratiques MLOps via **Git** et **DVC** (Data Version Control).

---

## 🗂️ Structure du projet

```
tweet_suspect/
├── data/
│   ├── tweets.csv              # Dataset brut (versionné par DVC)
│   ├── tweets.csv.dvc          # Pointeur DVC du dataset
│   └── tweets_preprocessed.csv # Données prétraitées (généré par pipeline)
│
├── src/
│   ├── preprocess.py           # Étape 1 : prétraitement du texte
│   ├── train.py                # Étape 2 : entraînement des modèles
│   └── evaluate.py             # Étape 3 : évaluation et figures
│
├── models/                     # Modèles entraînés (gérés par DVC)
│   ├── best_model.pkl
│   ├── all_models.pkl
│   └── test_data.pkl
│
├── notebooks/
│   └── partie1_eda_preprocessing.ipynb  # Analyse exploratoire
│
├── reports/
│   └── figures/                # Figures générées automatiquement
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── models_comparison.png
│
├── metrics/                    # Métriques DVC (JSON)
│   ├── preprocess_metrics.json
│   ├── train_metrics.json
│   └── metrics.json
│
├── dvc.yaml                    # Définition du pipeline DVC
├── params.yaml                 # Paramètres centraux (hyperparamètres)
├── requirements.txt            # Dépendances Python
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-username>/tweet-suspect.git
cd tweet-suspect

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger les données depuis le remote DVC
dvc pull
```

---

## 🚀 Reproduire les résultats

```bash
# Exécuter le pipeline complet (ou uniquement les étapes modifiées)
dvc repro

# Visualiser le graphe de dépendances
dvc dag

# Afficher les métriques
dvc metrics show

# Comparer deux versions
dvc metrics diff HEAD~1
```

---

## 📊 Pipeline DVC

```
data/tweets.csv
      │
      ▼
┌─────────────┐
│  preprocess  │  src/preprocess.py + params.yaml [preprocess]
└─────────────┘
      │
      ▼  data/tweets_preprocessed.csv
┌─────────────┐
│    train     │  src/train.py + params.yaml [vectorizer, train]
└─────────────┘
      │
      ▼  models/best_model.pkl
┌─────────────┐
│   evaluate   │  src/evaluate.py
└─────────────┘
      │
      ▼  metrics/metrics.json + reports/figures/
```

---

## 🔧 Paramètres (`params.yaml`)

| Section | Paramètre | Valeur par défaut | Description |
|---------|-----------|-------------------|-------------|
| `preprocess` | `lemmatize` | `true` | Activer la lemmatisation |
| `preprocess` | `preserve_negations` | `true` | Conserver les mots négatifs |
| `vectorizer` | `max_features` | `10000` | Taille du vocabulaire TF-IDF |
| `vectorizer` | `ngram_range` | `[1, 2]` | Unigrammes + bigrammes |
| `train` | `class_weight` | `balanced` | Gestion du déséquilibre |
| `train` | `cv_folds` | `5` | Folds pour la validation croisée |
| `train` | `test_size` | `0.2` | Proportion du set de test |

Modifier `params.yaml` puis relancer `dvc repro` pour ré-entraîner automatiquement.

---

## 📈 Modèles comparés

| Modèle | Représentation |
|--------|---------------|
| Logistic Regression | TF-IDF (1-2 grammes) |
| Naive Bayes | TF-IDF + MaxAbsScaler |
| LinearSVC | TF-IDF (1-2 grammes) |
| Random Forest | TF-IDF (1-2 grammes) |

---

## 📦 Versionner une nouvelle expérience

```bash
# Modifier les paramètres
nano params.yaml

# Ré-exécuter le pipeline
dvc repro

# Comparer les métriques
dvc metrics diff

# Committer la nouvelle version
git add dvc.lock params.yaml
git commit -m "exp: augmentation max_features à 15000"
git tag exp-v2

# Pousser vers le remote
git push
dvc push
```

---

## 🌐 Déploiement

### Option A — Interface Streamlit
```bash
streamlit run app/streamlit_app.py
```

### Option B — API FastAPI
```bash
uvicorn app.api:app --reload --port 8000
# Documentation : http://localhost:8000/docs
```

---

## 👤 Auteur

Projet réalisé dans le cadre du cours **Construction de Modèles et leur Déploiement**.
