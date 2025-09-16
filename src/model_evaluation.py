# src/model_evaluation.py
"""
Оценка обученных моделей: accuracy, precision, recall, f1, roc_auc, confusion matrix.
Возвращает словарь с метриками и сохраняет CSV.
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    # Для ROC AUC нужно predict_proba
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        # если модель не поддерживает predict_proba
        y_proba = None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_proba': y_proba
    }
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    else:
        metrics['roc_auc'] = None

    return metrics


def evaluate_models_dict(trained_pipelines: dict, X_test, y_test):
    results = {}
    for name, pipeline in trained_pipelines.items():
        print(f"Оценка метрик {name} ...")
        results[name] = evaluate_model(pipeline, X_test, y_test)
    # Сохраним таблицу метрик
    rows = []
    for name, m in results.items():
        rows.append({
            'model': name,
            'accuracy': m['accuracy'],
            'precision': m['precision'],
            'recall': m['recall'],
            'f1': m['f1'],
            'roc_auc': m.get('roc_auc', None)
        })
    df_metrics = pd.DataFrame(rows).set_index('model')
    os.makedirs('reports', exist_ok=True)
    df_metrics.to_csv('reports/metrics_summary.csv')
    print("Метрики сохранены в reports/metrics_summary.csv")
    return results, df_metrics
