# src/data_preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_raw(path="data/raw/Telco-Customer-Churn.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_csv(path)
    return df


def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # убираем customerID (не признак)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # убираем пробелы в строковых колонках
    str_cols = df.select_dtypes(include=['object']).columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()

    # TotalCharges может быть object с пробелами -> приводим к float
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace("", np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Преобразуем Churn в 0/1
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1}).astype(int)

    return df


def fill_missing_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'TotalCharges' in df.columns:
        # попытка эмпирического заполнения: MonthlyCharges * tenure
        mask = df['TotalCharges'].isnull()
        if mask.any():
            est = (df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']).fillna(0)
            median_val = est.median()
            df.loc[mask, 'TotalCharges'] = est.fillna(median_val)
        # если всё ещё NaN — заменить на 0
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    return df


def save_processed(df: pd.DataFrame, path="data/processed/telco_churn_processed.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def get_features_target(df: pd.DataFrame, target_col='Churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def train_test_split_df(df: pd.DataFrame, target_col='Churn', test_size=0.2, random_state=42, stratify=True):
    X, y = get_features_target(df, target_col)
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
