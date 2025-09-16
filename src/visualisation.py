# src/visualisation.py
"""
Графики сравнения метрик, ROC-кривые и confusion matrix.
Сохраняет картинки в reports/figures/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
sns.set(style="whitegrid")


def plot_metrics_bar(df_metrics: pd.DataFrame, save_path="reports/figures/model_metrics_bar.png"):
    """
    df_metrics: DataFrame index=model, columns = accuracy, precision, recall, f1, roc_auc (optional)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = df_metrics.reset_index().melt(id_vars='model', var_name='metric', value_name='value')
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='value', hue='metric', data=df)
    plt.ylim(0,1)
    plt.title("Сравнение метрик моделей")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_roc_curves(results: dict, X_test, y_test, save_path="reports/figures/roc_curves.png"):
    """Строит ROC-кривые для всех моделей (если есть predict_proba)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8,6))
    for name, metrics in results.items():
        y_proba = metrics.get('y_proba', None)
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(cm, model_name="model", save_path="reports/figures/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_feature_importance(pipeline, model_name="model", top_n=20, save_path="reports/figures/feature_importance.png"):
    """
    Попытка извлечь имена признаков из pipeline (ColumnTransformer + OneHotEncoder)
    и затем взять feature_importances_ у базовой модели (если есть).
    Работает для RandomForest, DecisionTree
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        clf = pipeline.named_steps['clf']
    except Exception:
        print("Pipeline должен содержать named_steps 'preprocessor' и 'clf'")
        return

    # извлечь имена признаков
    feature_names = []
    try:
        # Для sklearn >= 1.0
        num_cols = preprocessor.transformers_[0][2]
        cat_pipe = preprocessor.transformers_[1][1]
        cat_cols = preprocessor.transformers_[1][2]
        # имена категориальных после OneHotEncoder
        ohe = cat_pipe.named_steps['onehot']
        cat_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names = list(num_cols) + cat_names
    except Exception:
        # fallback: придумаем упрощённый список
        feature_names = [f"f{i}" for i in range(getattr(clf, "n_features_in_", 0))]

    # получаем importances
    try:
        importances = clf.feature_importances_
    except Exception:
        print("У модели нет feature_importances_")
        return

    feat_imp = pd.Series(importances, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title(f"Feature importances: {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
