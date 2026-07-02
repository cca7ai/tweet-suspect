"""
=============================================================
app/streamlit_app.py — Partie 7 : Déploiement Streamlit
=============================================================
Application de détection de tweets suspects.
Charge le modèle optimisé (Partie 6) et permet :
  - Saisir un tweet
  - Obtenir la prédiction (Normal / Suspect)
  - Afficher la probabilité associée
  - Analyser un batch de tweets (fichier CSV)

Lancement :
    streamlit run app/streamlit_app.py
=============================================================
"""

import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import time

# ── Téléchargement NLTK silencieux ───────────────────────────
for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)


# ════════════════════════════════════════════════════════════
# Configuration de la page
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "Détection de Tweets Suspects",
    page_icon   = "🐦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ════════════════════════════════════════════════════════════
# CSS personnalisé
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Fond principal */
    .main { background-color: #F8F9FA; }

    /* Carte résultat */
    .result-card {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .result-suspect {
        background: linear-gradient(135deg, #FDEDEC, #FADBD8);
        border-left: 6px solid #E74C3C;
        color: #922B21;
    }
    .result-normal {
        background: linear-gradient(135deg, #EAFAF1, #D5F5E3);
        border-left: 6px solid #2ECC71;
        color: #1E8449;
    }

    /* Métriques */
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #7F8C8D; margin-top: 0.2rem; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #2C3E50, #3498DB);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }

    /* Tweet card */
    .tweet-display {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-style: italic;
        color: #2C3E50;
        margin: 0.5rem 0;
    }

    /* Badge */
    .badge-suspect {
        background: #E74C3C; color: white;
        padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.85rem;
    }
    .badge-normal {
        background: #2ECC71; color: white;
        padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Chargement du modèle
# ════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    """Charge le modèle optimisé depuis le fichier pkl."""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models', 'best_model_optimized.pkl'
    )
    if not os.path.exists(model_path):
        # Fallback : chercher dans le dossier courant
        model_path = 'models/best_model_optimized.pkl'

    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    return artifact


# ════════════════════════════════════════════════════════════
# Pipeline de prétraitement (identique Partie 1)
# ════════════════════════════════════════════════════════════
PRESERVE_NEGATIONS = {
    'no', 'not', 'never', 'nothing', 'nobody', 'nowhere',
    'none', 'nor', 'neither', 'cannot', "can't", "won't",
    'against', 'hate', 'bad', 'worst'
}

@st.cache_resource
def load_nlp_tools():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - PRESERVE_NEGATIONS
    return lemmatizer, stop_words


def preprocess_tweet(text: str) -> str:
    """Même pipeline que src/preprocess.py."""
    lemmatizer, stop_words = load_nlp_tools()
    if not text or str(text).strip() == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split()
              if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    tokens = [lemmatizer.lemmatize(t, pos='n') for t in tokens]
    return ' '.join(tokens)


def predict_tweet(text: str, artifact: dict) -> dict:
    """Prétraite et classifie un tweet."""
    model  = artifact['model']
    tfidf  = artifact['tfidf']

    cleaned = preprocess_tweet(text)
    if not cleaned:
        return {
            'label': -1, 'label_name': 'Vide',
            'proba_suspect': 0.0, 'proba_normal': 0.0,
            'cleaned': cleaned, 'original': text
        }

    X = tfidf.transform([cleaned])

    # Certains modèles nécessitent des matrices denses
    model_name = artifact.get('model_name', '')
    if any(k in model_name for k in ['SVM RBF', 'Forest', 'Boosting', 'XGB']):
        X = X.toarray()

    pred = model.predict(X)[0]

    try:
        proba = model.predict_proba(X)[0]
        p_suspect = float(proba[1])
        p_normal  = float(proba[0])
    except AttributeError:
        raw = model.decision_function(X)[0]
        p_suspect = float(1 / (1 + np.exp(-raw)))
        p_normal  = 1 - p_suspect

    return {
        'label'        : int(pred),
        'label_name'   : 'Suspect' if pred == 1 else 'Normal',
        'proba_suspect': round(p_suspect * 100, 2),
        'proba_normal' : round(p_normal  * 100, 2),
        'cleaned'      : cleaned,
        'original'     : text,
    }


def make_proba_gauge(p_suspect: float):
    """Crée un gauge matplotlib pour la probabilité."""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Barre de fond
    ax.barh(0, 100, color='#E8F8F5', height=0.5, edgecolor='#BDC3C7')
    # Barre colorée
    color = '#E74C3C' if p_suspect >= 50 else '#2ECC71'
    ax.barh(0, p_suspect, color=color, height=0.5, alpha=0.85)
    # Seuil 50%
    ax.axvline(x=50, color='#7F8C8D', linewidth=1.5, linestyle='--')
    ax.text(50, 0.35, '50%', ha='center', va='bottom', fontsize=9, color='#7F8C8D')

    ax.text(p_suspect / 2 if p_suspect > 15 else p_suspect + 2,
            0, f'{p_suspect:.1f}%',
            ha='center', va='center', fontweight='bold',
            fontsize=14, color='white' if p_suspect > 20 else color)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.7)
    ax.axis('off')
    ax.set_title('Probabilité — Suspect', fontsize=11, pad=5)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/twitter.png", width=60)
    st.title("⚙️ Paramètres")
    st.divider()

    # Infos modèle
    try:
        artifact = load_model()
        st.success("✅ Modèle chargé")
        st.markdown(f"**Modèle :** `{artifact.get('model_name', 'N/A')}`")
        st.markdown(f"**Méthode :** `{artifact.get('method', 'N/A')}`")
        metrics = artifact.get('test_metrics', {})
        if metrics:
            st.markdown(f"**F1-Score :** `{metrics.get('f1', 'N/A'):.4f}`")
            st.markdown(f"**AUC-ROC :** `{metrics.get('auc', 'N/A'):.4f}`")
            st.markdown(f"**Accuracy :** `{metrics.get('accuracy', 'N/A'):.4f}`")
        model_loaded = True
    except FileNotFoundError:
        st.error("❌ Modèle introuvable\n\n`models/best_model_optimized.pkl`\n\nLancez d'abord la Partie 6.")
        model_loaded = False

    st.divider()

    # Seuil de décision
    threshold = st.slider(
        "Seuil de décision",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Au-dessus de ce seuil → Suspect"
    )
    st.caption(f"Seuil actuel : **{threshold*100:.0f}%**")

    st.divider()
    st.markdown("### Guide")
    st.markdown("""
- **Normal** : tweet inoffensif
- **Suspect** : contenu potentiellement haineux, trompeur ou offensant

**Probabilité > seuil** → classé Suspect
    """)

    st.divider()
    st.markdown("*Projet : Construction de Modèles et leur Déploiement*")


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <h1 style="margin:0; font-size:2rem;">🐦 Détection de Tweets Suspects</h1>
    <p style="margin:0.5rem 0 0 0; opacity:0.85;">
        Classification automatique de tweets par Machine Learning —
        Examen Final MLOps
    </p>
</div>
""", unsafe_allow_html=True)


if not model_loaded:
    st.error("⛔ Le modèle n'est pas chargé. Exécutez le notebook Partie 6 d'abord.")
    st.stop()

artifact = load_model()

# ════════════════════════════════════════════════════════════
# ONGLETS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "Analyser un tweet",
    "Analyser un fichier CSV",
    "Informations du modèle"
])


# ────────────────────────────────────────────────────────────
# TAB 1 — Analyser un tweet
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Entrez un tweet à analyser")

    # Exemples cliquables
    st.markdown("**Exemples rapides :**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    examples = {
        "🚨 Suspect"  : "BREAKING: fake news spreading everywhere, don't believe mainstream media they're all LYING!!! Wake up people!!!",
        "✅ Normal"   : "Just had the most amazing breakfast this morning, feeling so energized and ready for the day!",
        "⚠️ Ambigu"  : "I want to kill this exam, I'm so stressed out about everything today.",
    }
    if col_ex1.button("🚨 Exemple suspect",  use_container_width=True):
        st.session_state['tweet_input'] = examples["🚨 Suspect"]
    if col_ex2.button("✅ Exemple normal",   use_container_width=True):
        st.session_state['tweet_input'] = examples["✅ Normal"]
    if col_ex3.button("⚠️ Exemple ambigu",  use_container_width=True):
        st.session_state['tweet_input'] = examples["⚠️ Ambigu"]

    # Zone de saisie
    tweet_text = st.text_area(
        label       = "Tweet",
        value       = st.session_state.get('tweet_input', ''),
        placeholder = "Entrez votre tweet ici...",
        height      = 120,
        key         = "tweet_input",
        label_visibility = "collapsed"
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    predict_btn = col_btn1.button("Analyser", type="primary",
                                   use_container_width=True)
    col_btn2.button("Effacer", on_click=lambda: st.session_state.update(
                    {'tweet_input': ''}), use_container_width=False)

    if predict_btn and tweet_text.strip():
        with st.spinner("Analyse en cours..."):
            time.sleep(0.3)  # effet visuel
            result = predict_tweet(tweet_text, artifact)

        # Appliquer le seuil personnalisé
        is_suspect = result['proba_suspect'] >= threshold * 100
        label_display = "Suspect" if is_suspect else "Normal"
        card_class    = "result-suspect" if is_suspect else "result-normal"
        icon          = "🚨" if is_suspect else "✅"

        st.markdown("---")

        # ── Résultat principal
        st.markdown(f"""
        <div class="result-card {card_class}">
            {icon} Ce tweet est classé comme : <strong>{label_display}</strong>
        </div>
        """, unsafe_allow_html=True)

        # ── Métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color:{'#E74C3C' if is_suspect else '#2ECC71'}">
                    {label_display}
                </div>
                <div class="metric-label">Prédiction</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color:#E74C3C">
                    {result['proba_suspect']:.1f}%
                </div>
                <div class="metric-label">P(Suspect)</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color:#2ECC71">
                    {result['proba_normal']:.1f}%
                </div>
                <div class="metric-label">P(Normal)</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color:#3498DB">
                    {threshold*100:.0f}%
                </div>
                <div class="metric-label">Seuil</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge de probabilité
        col_gauge, col_detail = st.columns([1, 1])
        with col_gauge:
            fig_gauge = make_proba_gauge(result['proba_suspect'])
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close()

        with col_detail:
            st.markdown("**Détail du traitement**")
            st.markdown(f"""
            <div class="tweet-display">
                <strong>Original :</strong><br>{tweet_text}
            </div>
            <div class="tweet-display" style="background:#F4F6F7">
                <strong>Après prétraitement :</strong><br>
                {result['cleaned'] if result['cleaned'] else '<em>(texte vide après nettoyage)</em>'}
            </div>
            """, unsafe_allow_html=True)

            # Tokens
            if result['cleaned']:
                tokens = result['cleaned'].split()
                st.markdown(f"**Tokens ({len(tokens)}) :** " +
                            " · ".join([f"`{t}`" for t in tokens[:15]]) +
                            (" ..." if len(tokens) > 15 else ""))

    elif predict_btn:
        st.warning("⚠️ Veuillez entrer un tweet.")


# ────────────────────────────────────────────────────────────
# TAB 2 — Analyser un fichier CSV
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Analyse en batch — fichier CSV")
    st.info("Le fichier doit contenir une colonne `message` avec les tweets à analyser.")

    uploaded_file = st.file_uploader(
        "Choisir un fichier CSV", type=['csv'],
        help="Format attendu : colonne 'message' (et optionnellement 'label')"
    )

    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.markdown(f"**{len(df_upload)} tweets chargés**")
        st.dataframe(df_upload.head(5), use_container_width=True)

        if 'message' not in df_upload.columns:
            st.error("❌ Colonne 'message' introuvable dans le CSV.")
        else:
            if st.button("Lancer l'analyse", type="primary"):
                progress = st.progress(0, text="Analyse en cours...")
                results  = []

                for i, row in enumerate(df_upload['message'].fillna('').tolist()):
                    res = predict_tweet(str(row), artifact)
                    is_s = res['proba_suspect'] >= threshold * 100
                    results.append({
                        'Tweet'          : str(row)[:80] + '...' if len(str(row)) > 80 else str(row),
                        'Prédiction'     : '🚨 Suspect' if is_s else '✅ Normal',
                        'P(Suspect) %'   : res['proba_suspect'],
                        'P(Normal) %'    : res['proba_normal'],
                        'Texte nettoyé'  : res['cleaned'],
                    })
                    progress.progress((i+1) / len(df_upload),
                                      text=f"Analyse : {i+1}/{len(df_upload)}")

                progress.empty()
                results_df = pd.DataFrame(results)

                # Statistiques
                n_suspect = (results_df['Prédiction'] == '🚨 Suspect').sum()
                n_normal  = (results_df['Prédiction'] == '✅ Normal').sum()

                st.markdown("---")
                st.markdown("### Résultats")

                col_s, col_n, col_t = st.columns(3)
                col_s.metric("🚨 Suspects", n_suspect,
                             f"{n_suspect/len(results_df)*100:.1f}%")
                col_n.metric("✅ Normaux",  n_normal,
                             f"{n_normal/len(results_df)*100:.1f}%")
                col_t.metric("📝 Total",    len(results_df))

                # Graphique distribution
                fig_dist, ax = plt.subplots(figsize=(8, 3))
                ax.hist(results_df['P(Suspect) %'], bins=20,
                        color='#E74C3C', alpha=0.7, edgecolor='white')
                ax.axvline(x=threshold*100, color='#2C3E50',
                           linestyle='--', linewidth=2,
                           label=f'Seuil ({threshold*100:.0f}%)')
                ax.set_xlabel('Probabilité Suspect (%)')
                ax.set_ylabel('Nombre de tweets')
                ax.set_title('Distribution des probabilités')
                ax.legend()
                fig_dist.patch.set_facecolor('white')
                st.pyplot(fig_dist, use_container_width=True)
                plt.close()

                # Tableau complet
                st.dataframe(
                    results_df.style.apply(
                        lambda col: [
                            'background-color: #FADBD8' if '🚨' in str(v)
                            else 'background-color: #D5F5E3'
                            for v in col
                        ] if col.name == 'Prédiction' else ['']*len(col),
                        axis=0
                    ),
                    use_container_width=True
                )

                # Export CSV
                csv_out = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Télécharger les résultats (CSV)",
                    data     = csv_out,
                    file_name= "predictions_tweets.csv",
                    mime     = "text/csv",
                )


# ────────────────────────────────────────────────────────────
# TAB 3 — Informations du modèle
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Informations sur le modèle déployé")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("#### Modèle")
        info_data = {
            "Algorithme"    : artifact.get('model_name', 'N/A'),
            "Optimisation"  : artifact.get('method', 'N/A'),
            "Vectorisation" : "TF-IDF (ngram 1-2, max 10 000 features)",
            "Équilibrage"   : "Class Weights (balanced)",
        }
        for k, v in info_data.items():
            st.markdown(f"**{k}** : `{v}`")

        st.markdown("#### Hyperparamètres optimaux")
        best_params = artifact.get('best_params', {})
        if best_params:
            for k, v in best_params.items():
                st.markdown(f"- `{k}` = **{v}**")
        else:
            st.markdown("*Non disponible*")

    with col_info2:
        st.markdown("#### Performances (test set)")
        metrics = artifact.get('test_metrics', {})
        metric_display = {
            'F1-Score' : metrics.get('f1', 'N/A'),
            'AUC-ROC'  : metrics.get('auc', 'N/A'),
            'Accuracy' : metrics.get('accuracy', 'N/A'),
            'Precision': metrics.get('precision', 'N/A'),
            'Recall'   : metrics.get('recall', 'N/A'),
        }
        for metric, val in metric_display.items():
            if isinstance(val, float):
                bar_val = val
                color   = '#2ECC71' if val >= 0.9 else '#F39C12' if val >= 0.8 else '#E74C3C'
                st.markdown(f"**{metric}** : `{val:.4f}`")
                st.progress(val, text="")
            else:
                st.markdown(f"**{metric}** : `{val}`")

    st.divider()
    st.markdown("#### Pipeline complet")
    st.markdown("""
    ```
    Tweet brut
        │
        ▼  Prétraitement (src/preprocess.py)
        │  → minuscules, URLs, mentions, stop words, lemmatisation
        │
        ▼  Vectorisation TF-IDF
        │  → max_features=10000, ngram_range=(1,2), sublinear_tf=True
        │
        ▼  Modèle optimisé
        │  → SVM RBF (C, gamma optimisés par Grid/Random Search)
        │
        ▼  Probabilité + Seuil
        └→ Normal (0) ou Suspect (1)
    ```
    """)

    st.divider()
    st.markdown("#### Structure du projet")
    st.code("""
tweet_suspect/
├── app/
│   └── streamlit_app.py        ← Cette application
├── data/
│   ├── tweets.csv              (versionné DVC)
│   └── tweets_preprocessed.csv
├── models/
│   └── best_model_optimized.pkl  ← Modèle chargé
├── src/
│   ├── preprocess.py
│   ├── vectorize.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   ├── partie1_eda_preprocessing.ipynb
│   ├── partie3_representation.ipynb
│   ├── partie4_modeles.ipynb
│   ├── partie5_validation.ipynb
│   └── partie6_optimisation.ipynb
├── dvc.yaml
├── params.yaml
└── requirements.txt
    """, language="")
