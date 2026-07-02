"""
=============================================================
src/evaluate.py — Étape 3 du pipeline DVC
=============================================================
Évaluation complète du meilleur modèle :
  - Métriques détaillées (Accuracy, Precision, Recall, F1)
  - Matrice de confusion
  - Courbe ROC et AUC
  - Sauvegarde des métriques DVC (metrics.json)
  - Génération des figures d'évaluation

Usage :
    python src/evaluate.py --model-dir models/ \
                           --params params.yaml \
                           --output-dir reports/figures/
=============================================================
"""

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    ConfusionMatrixDisplay
)


COLORS = {'suspect': '#E74C3C', 'normal': '#2ECC71', 'accent': '#3498DB'}


def load_artifacts(model_dir: str):
    """Charge le modèle et les données de test."""
    with open(os.path.join(model_dir, 'best_model.pkl'), 'rb') as f:
        artifact = pickle.load(f)
    with open(os.path.join(model_dir, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    return artifact, test_data


def plot_confusion_matrix(y_test, y_pred, model_name: str, output_dir: str):
    """Génère et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Matrice de Confusion — {model_name}', fontsize=13, fontweight='bold')

    for ax, matrix, title, fmt in [
        (axes[0], cm,      'Valeurs absolues', 'd'),
        (axes[1], cm_norm, 'Valeurs normalisées (%)', '.2%')
    ]:
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                      display_labels=['Normal', 'Suspect'])
        disp.plot(ax=ax, colorbar=True,
                  cmap='Blues' if fmt == 'd' else 'RdYlGn',
                  values_format=fmt)
        ax.set_title(title, fontweight='bold', pad=12)
        ax.set_xlabel('Classe Prédite', fontweight='bold')
        ax.set_ylabel('Classe Réelle', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Matrice de confusion : {path}")
    return cm


def plot_roc_curve(model_name: str, y_test, scores, output_dir: str) -> float:
    """Génère et sauvegarde la courbe ROC."""
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    # Point optimal (Youden)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color=COLORS['suspect'], lw=2.5,
            label=f'Courbe ROC (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS['suspect'])
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Classifieur aléatoire (AUC = 0.50)')

    # Point optimal
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], s=120,
               color=COLORS['accent'], zorder=5,
               label=f'Seuil optimal = {optimal_threshold:.3f}')
    ax.annotate(f'  Seuil={optimal_threshold:.3f}\n  FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f}',
                xy=(fpr[optimal_idx], tpr[optimal_idx]),
                xytext=(fpr[optimal_idx] + 0.08, tpr[optimal_idx] - 0.08),
                fontsize=9, color=COLORS['accent'])

    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    ax.set_title(f'Courbe ROC — {model_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.grid(True, alpha=0.3)

    # Texte AUC
    ax.text(0.55, 0.15, f'AUC = {roc_auc:.4f}', fontsize=14,
            fontweight='bold', color=COLORS['suspect'],
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Courbe ROC (AUC={roc_auc:.4f}) : {path}")
    return roc_auc


def plot_models_comparison(train_metrics_path: str, output_dir: str):
    """Graphique de comparaison des modèles."""
    if not os.path.exists(train_metrics_path):
        return

    with open(train_metrics_path) as f:
        data = json.load(f)

    models_data = data.get('models', {})
    if not models_data:
        return

    model_names = list(models_data.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels   = ['Accuracy', 'Précision', 'Rappel', 'F1-Score']
    colors_list     = [COLORS['accent'], '#9B59B6', '#F39C12', COLORS['suspect']]

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Comparaison des Modèles — Métriques de Classification',
                 fontsize=13, fontweight='bold')

    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors_list)):
        values = [models_data[m][metric] for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=label,
                      color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xlabel('Modèle', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(model_names) - 0.1, 0.81, 'Seuil 80%', fontsize=8, color='gray')

    plt.tight_layout()
    path = os.path.join(output_dir, 'models_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Comparaison des modèles : {path}")


def main(model_dir: str, output_dir: str,
         dvc_metrics_path: str = 'metrics/metrics.json',
         train_metrics_path: str = 'metrics/train_metrics.json'):

    print("=" * 60)
    print("  ÉTAPE 3 — ÉVALUATION DU MODÈLE")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    # ── Chargement ───────────────────────────────────────────────
    artifact, test_data = load_artifacts(model_dir)
    model      = artifact['model']
    model_name = artifact['name']
    X_test     = test_data['X_test']
    y_test     = test_data['y_test']

    print(f"\n  Modèle évalué : {model_name}")
    print(f"  Taille du test set : {len(y_test)}")

    # ── Prédictions ──────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # Scores de probabilité ou de décision pour ROC
    try:
        scores = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        scores = model.decision_function(X_test)
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    # ── Métriques ────────────────────────────────────────────────
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  Métriques finales :")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1-Score  : {f1:.4f}")

    print(f"\n  Rapport de classification complet :")
    print(classification_report(y_test, y_pred,
                                target_names=['Normal (0)', 'Suspect (1)']))

    # ── Figures ──────────────────────────────────────────────────
    print("  Génération des figures...")
    cm      = plot_confusion_matrix(y_test, y_pred, model_name, output_dir)
    roc_auc = plot_roc_curve(model_name, y_test, scores, output_dir)
    plot_models_comparison(train_metrics_path, output_dir)

    # ── Sauvegarde métriques DVC ─────────────────────────────────
    dvc_metrics = {
        "accuracy" : round(acc,     4),
        "precision": round(prec,    4),
        "recall"   : round(rec,     4),
        "f1_score" : round(f1,      4),
        "auc_roc"  : round(roc_auc, 4),
        "model"    : model_name,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1])
        }
    }

    with open(dvc_metrics_path, 'w') as f:
        json.dump(dvc_metrics, f, indent=2)

    print(f"\n  Métriques DVC sauvegardées : {dvc_metrics_path}")
    print(f"     AUC-ROC : {roc_auc:.4f}")
    print(f"     F1-Score : {f1:.4f}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Évaluation du modèle')
    parser.add_argument('--model-dir',  default='models/')
    parser.add_argument('--output-dir', default='reports/figures/')
    parser.add_argument('--metrics',    default='metrics/metrics.json')
    parser.add_argument('--train-metrics', default='metrics/train_metrics.json')
    args = parser.parse_args()

    main(args.model_dir, args.output_dir, args.metrics, args.train_metrics)
