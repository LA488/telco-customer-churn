# app/main.py

import os
import sys
import joblib
import pandas as pd
import ttkbootstrap as tb
from ttkbootstrap.constants import *


def resource_path(relative_path: str) -> str:
    """
    Возвращает абсолютный путь как при обычном запуске,
    так и внутри PyInstaller (.exe).
    """
    if hasattr(sys, "_MEIPASS"):
        # Когда запускается .exe, PyInstaller распаковывает всё во временную папку
        base_path = sys._MEIPASS
    else:
        # Когда запускается как обычный python app/main.py
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_path, relative_path)


MODEL_PATH = resource_path(os.path.join("models", "RandomForest_pipeline.pkl"))


class TelcoChurnApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Telco Churn Predictor")
        self.root.geometry("600x400")

        # 🔹 Модель и статус
        self.model = None
        self.status_var = tb.StringVar(value="")
        self.load_model()

        # Разделы интерфейса
        self.create_personal_section()
        self.create_services_section()
        self.create_payment_section()
        self.create_buttons()
        self.create_status_bar()

    def load_model(self):
        """Загружаем модель RandomForest"""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.status_var.set(f"✅ Модель загружена\n{MODEL_PATH}")
            except Exception as e:
                self.status_var.set(f"❌ Ошибка загрузки: {e}")
        else:
            self.status_var.set(f"⚠ Модель не найдена: {MODEL_PATH}")


    def create_personal_section(self):
        frame = tb.LabelFrame(self.root, text="Личные данные", bootstyle=PRIMARY)
        frame.pack(fill="x", padx=10, pady=5)

        self.vars = {
            "gender": tb.StringVar(value="Male"),
            "SeniorCitizen": tb.BooleanVar(value=False),
            "Partner": tb.BooleanVar(value=False),
            "Dependents": tb.BooleanVar(value=False),
            "tenure": tb.StringVar(value="12"),
            "MonthlyCharges": tb.StringVar(value="50"),
            "TotalCharges": tb.StringVar(value="600"),
        }

        tb.Label(frame, text="Gender:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tb.Combobox(frame, textvariable=self.vars["gender"],
                    values=["Male", "Female"], width=10).grid(row=0, column=1)

        tb.Checkbutton(frame, text="Senior Citizen",
                       variable=self.vars["SeniorCitizen"]).grid(row=0, column=2, padx=5)

        tb.Checkbutton(frame, text="Partner", variable=self.vars["Partner"]).grid(row=0, column=3, padx=5)
        tb.Checkbutton(frame, text="Dependents", variable=self.vars["Dependents"]).grid(row=0, column=4, padx=5)

        tb.Label(frame, text="Tenure:").grid(row=1, column=0, sticky="w", padx=5)
        tb.Entry(frame, textvariable=self.vars["tenure"], width=10).grid(row=1, column=1)

        tb.Label(frame, text="MonthlyCharges:").grid(row=1, column=2, sticky="w", padx=5)
        tb.Entry(frame, textvariable=self.vars["MonthlyCharges"], width=10).grid(row=1, column=3)

        tb.Label(frame, text="TotalCharges:").grid(row=1, column=4, sticky="w", padx=5)
        tb.Entry(frame, textvariable=self.vars["TotalCharges"], width=10).grid(row=1, column=5)

    def create_services_section(self):
        frame = tb.LabelFrame(self.root, text="Услуги", bootstyle=SUCCESS)
        frame.pack(fill="x", padx=10, pady=5)

        service_features = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        for i, feature in enumerate(service_features):
            var = tb.BooleanVar(value=False)
            self.vars[feature] = var
            tb.Checkbutton(frame, text=feature, variable=var).grid(row=i // 3, column=i % 3, sticky="w",
                                                                   padx=10, pady=2)

    def create_payment_section(self):
        frame = tb.LabelFrame(self.root, text="Оплата", bootstyle=WARNING)
        frame.pack(fill="x", padx=10, pady=5)

        self.vars["Contract"] = tb.StringVar(value="Month-to-month")
        self.vars["PaperlessBilling"] = tb.BooleanVar(value=False)
        self.vars["PaymentMethod"] = tb.StringVar(value="Electronic check")

        tb.Label(frame, text="Contract:").grid(row=0, column=0, padx=5, sticky="w")
        tb.Combobox(frame, textvariable=self.vars["Contract"],
                    values=["Month-to-month", "One year", "Two year"], width=15).grid(row=0, column=1)

        tb.Checkbutton(frame, text="PaperlessBilling",
                       variable=self.vars["PaperlessBilling"]).grid(row=0, column=2, padx=5)

        tb.Label(frame, text="PaymentMethod:").grid(row=0, column=3, padx=5, sticky="w")
        tb.Combobox(frame, textvariable=self.vars["PaymentMethod"],
                    values=[
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"
                    ],
                    width=20).grid(row=0, column=4)

    def create_buttons(self):
        frame = tb.Frame(self.root)
        frame.pack(fill="x", padx=10, pady=10)

        tb.Button(frame, text="Предсказать", bootstyle=SUCCESS,
                  command=self.show_data).pack(side="left", padx=5)
        tb.Button(frame, text="Сбросить", bootstyle=SECONDARY,
                  command=self.reset).pack(side="left", padx=5)

        self.result_label = tb.Label(self.root, text="",
                                     font=("Arial", 14), bootstyle=INFO)
        self.result_label.pack(pady=10)

    def create_status_bar(self):
        status_label = tb.Label(self.root, textvariable=self.status_var,
                                bootstyle=INFO, anchor="w")
        status_label.pack(side="bottom", fill="x", padx=5, pady=5)

    def reset(self):
        for var in self.vars.values():
            if isinstance(var, (tb.StringVar, tb.BooleanVar)):
                var.set(False if isinstance(var, tb.BooleanVar) else "")

    def show_data(self):
        """Формируем данные и делаем предсказание"""
        data = {}
        for key, var in self.vars.items():
            if isinstance(var, tb.BooleanVar):
                if key == "SeniorCitizen":
                    data[key] = 1 if var.get() else 0
                else:
                    data[key] = "Yes" if var.get() else "No"
            else:
                data[key] = var.get()

        if self.model:
            try:
                df = pd.DataFrame([data])
                pred = self.model.predict(df)[0]
                proba = self.model.predict_proba(df)[0][1]

                result = f"Предсказание: {'⚠ Уйдёт' if pred == 1 else '✅ Останется'}\n" \
                         f"Вероятность ухода: {proba * 100:.2f}%"
            except Exception as e:
                result = f"❌ Ошибка предсказания: {e}"
        else:
            result = f"ДЕМО-режим\n{data}"

        self.result_label.config(text=result)


if __name__ == "__main__":
    root = tb.Window(themename="flatly")
    app = TelcoChurnApp(root)
    root.mainloop()
