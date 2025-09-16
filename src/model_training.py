# src/model_training.py


import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')

    return preprocessor, numeric_features, categorical_features


def build_model_pipelines(X_train: pd.DataFrame):
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    pipelines = {}

    # Логистическая регрессия
    lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ])
    pipelines['LogisticRegression'] = lr

    # Дерево решений
    dt = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', DecisionTreeClassifier(class_weight='balanced', random_state=42))
    ])
    pipelines['DecisionTree'] = dt

    # Случайный лес
    rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42))
    ])
    pipelines['RandomForest'] = rf
    return pipelines


def train_all(pipelines: dict, X_train, y_train):
    """Обучает все пайплайны и возвращает dict обученных моделей."""
    trained = {}
    for name, pipe in pipelines.items():
        print(f"Обучение {name} ...")
        pipe.fit(X_train, y_train)
        trained[name] = pipe
    return trained


def save_pipeline(pipeline, path="models/best_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Пайплайн сохранен в {path}")


def tune_models(pipelines: dict, X_train, y_train, cv=3):
    """Подбирает гиперпараметры для логистической регрессии, дерева решений и случайного леса."""
    param_grids = {
        'LogisticRegression': {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs', 'saga']
        },
        'DecisionTree': {
            'clf__max_depth': [3, 5, 10, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 5]
        },
        'RandomForest': {
            'clf__n_estimators': [200, 500],
            'clf__max_depth': [5, 10, None],
            'clf__max_features': ['sqrt', 'log2'],
            'clf__min_samples_leaf': [1, 2, 5]
        }
    }

    tuned = {}
    for name, pipe in pipelines.items():
        print(f"🔍 Подбор параметров для {name} ...")
        grid = GridSearchCV(pipe, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        print(f"✅ {name}: лучшие параметры {grid.best_params_}, лучший score {grid.best_score_:.4f}")
        tuned[name] = grid.best_estimator_
    return tuned

