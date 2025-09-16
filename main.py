# main.py

from src.data_preprocessing import load_raw, clean_basic, fill_missing_basic, save_processed, train_test_split_df
from src.model_training import build_model_pipelines, save_pipeline, tune_models
from src.model_evaluation import evaluate_models_dict
from src.visualisation import plot_metrics_bar, plot_roc_curves, plot_confusion_matrix, plot_feature_importance

import os
import pandas as pd


def main():
    # 1. Загрузка и очистка данных
    df = load_raw("data/raw/Telco-Customer-Churn.csv")
    df = clean_basic(df)
    df = fill_missing_basic(df)
    save_processed(df, path="data/processed/telco_churn_processed.csv")
    print("Данные загружены и предобработаны. Размерность:", df.shape)

    # 2. Разделение на тренировочные и тестовые данные
    X_train, X_test, y_train, y_test = train_test_split_df(df, target_col='Churn', test_size=0.2)
    print("Разделение Train/test:", X_train.shape, X_test.shape)

    # 3. Создание pipeline и train
    pipelines = build_model_pipelines(X_train)
    trained = tune_models(pipelines, X_train, y_train)

    # 4. Оценка метрик
    results, df_metrics = evaluate_models_dict(trained, X_test, y_test)

    # 5. Визуализация
    plot_metrics_bar(df_metrics, save_path="reports/figures/model_metrics_bar.png")
    plot_roc_curves(results, X_test, y_test, save_path="reports/figures/roc_curves.png")
    # confusion matrices per model
    for name, metrics in results.items():
        cm = metrics['confusion_matrix']
        plot_confusion_matrix(cm, model_name=name, save_path=f"reports/figures/cm_{name}.png")
        # feature importance (если имеется)
        try:
            plot_feature_importance(trained[name], model_name=name, save_path=f"reports/figures/fi_{name}.png")
        except Exception as e:
            print(f"Feature importance for {name} failed: {e}")

    # 6. Сохраняем лучшую модель (по roc_auc или f1)
    # выбираем по roc_auc если есть, иначе по f1
    best_model_name = None
    best_score = -1
    for name, metrics in results.items():
        score = metrics.get('roc_auc')
        if score is None:
            score = metrics.get('f1', 0)
        if score is None:
            score = 0
        if score > best_score:
            best_score = score
            best_model_name = name

    print(f"Лучшая модель: {best_model_name} (score = {best_score:.4f})")
    # сохраняем весь pipeline
    save_pipeline(trained[best_model_name], path=f"models/{best_model_name}_pipeline.pkl")


if __name__ == "__main__":
    main()
