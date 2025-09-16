# Telco Customer Churn Prediction / Прогнозирование оттока клиентов Telco
  
| Колонка            | Перевод |
|---------------------|---------|
| `customerID`        | Идентификатор клиента |
| `gender`            | Пол |
| `SeniorCitizen`     | Пенсионер (1 — да, 0 — нет) |
| `Partner`           | Наличие супруга/супруги |
| `Dependents`        | Наличие иждивенцев |
| `tenure`            | Количество месяцев, которые клиент пользуется услугами |
| `PhoneService`      | Наличие телефонной связи |
| `MultipleLines`     | Наличие нескольких телефонных линий |
| `InternetService`   | Тип интернет-сервиса (DSL, Fiber optic, None) |
| `OnlineSecurity`    | Онлайн-безопасность |
| `OnlineBackup`      | Онлайн-резервное копирование |
| `DeviceProtection`  | Защита устройства |
| `TechSupport`       | Техническая поддержка |
| `StreamingTV`       | Стриминг ТВ |
| `StreamingMovies`   | Стриминг фильмов |
| `Contract`          | Тип контракта (месячный, годовой, двухгодовой) |
| `PaperlessBilling`  | Безбумажная оплата |
| `PaymentMethod`     | Способ оплаты |
| `MonthlyCharges`    | Ежемесячные расходы |
| `TotalCharges`      | Общие расходы |
| `Churn`             | Отток клиента (Yes — ушёл, No — остался) |

---

## Описание датасета / Dataset Description

Датасет содержит информацию о **7043 клиентах телекоммуникационной компании**.  
Задача — **предсказать вероятность оттока клиентов** (`Churn`) на основе демографических данных, типа контракта, предоставляемых услуг и расходов клиента.  

The dataset contains information about 7043 customers of a telecom company.
The goal is to predict customer churn probability (Churn) based on demographics, contract type, provided services, and customer expenses.


##  Цели проекта / Project Goals

- Выполнить исследовательский анализ данных (EDA)  
- Подготовить данные для моделей (очистка, кодирование категориальных признаков, масштабирование числовых)  
- Построить и обучить модели машинного обучения  
- Сравнить метрики и выбрать наилучший алгоритм  
- Разработать приложение для демонстрации работы модели  
---
- Perform Exploratory Data Analysis (EDA)
- Prepare data for models (cleaning, encoding categorical features, scaling numerical features)
- Build and train machine learning models
- Compare metrics and select the best algorithm
- Develop an application to demonstrate the model
---

## Используемый стек / Tech Stack

- **Python** 3.12
- **pandas, numpy** — работа с данными / data processing
- **matplotlib, seaborn** — визуализация / visualization
- **scikit-learn** — модели / machine learning
- **joblib** — сохранение пайплайнов / model persistence
- **PyInstaller** — упаковка приложения в .exe / packaging into .exe
- **tkinter** — GUI-приложение / GUI application  

---

## Структура проекта / Project Structure

```bash
│
├── app/                     # Приложение / Application
│   ├── dist/                # Скомпилированные бинарники / Compiled binaries
│   │   └── TelcoApp.exe     # Исполняемый файл / Executable
│   └── main.py              # Запуск приложения / App entry point
│
├── data/                    # Данные / Data
│   ├── raw/                 # Исходные данные / Raw data
│   │   └── Telco-Customer-Churn.csv
│   └── processed/           # Обработанные данные / Processed data
│       └── telco_churn_processed.csv
│
├── models/                  # Сохранённые модели / Saved models
│   └── RandomForest_pipeline.pkl
│
├── notebook/                # Ноутбуки / Notebooks
│   └── eda_telco.ipynb      # EDA
│
├── reports/                 # Отчёты / Reports
│   ├── figures/             # Визуализации / Figures
│   └── metrics_summary.csv  # Метрики / Metrics summary
│
├── src/                     # Основные скрипты / Source code
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── utils.py
│   └── visualisation.py
│
├── main.py                  # Главный скрипт / Main pipeline script
├── README.md                # Документация / Documentation
├── requirements.txt         # Зависимости / Dependencies
└── TelcoApp.spec            # Сборка / Build config

```

---

## Описание файлов  

### 🔹 `app/`  
- **main.py** — точка входа для запуска приложения.  
- **TelcoApp.exe** — собранное приложение для Windows.  

### 🔹 `data/`  
- **raw/Telco-Customer-Churn.csv** — исходные данные.  
- **processed/telco_churn_processed.csv** — очищенные и предобработанные данные.  

### 🔹 `models/`  
- **RandomForest_pipeline.pkl** — обученный пайплайн (препроцессинг + модель).  

### 🔹 `notebook/`  
- **eda_telco.ipynb** — исследовательский анализ данных (графики, корреляции, выводы).  

### 🔹 `reports/`  
- **figures/** — сохранённые графики визуализаций.  
- **metrics_summary.csv** — таблица с результатами метрик для моделей.  

### 🔹 `src/`  
- **data_preprocessing.py** — функции для загрузки и очистки данных.  
- **model_training.py** — обучение моделей (логистическая регрессия, дерево решений, случайный лес).  
- **model_evaluation.py** — метрики и визуализации качества моделей.  
- **utils.py** — вспомогательные функции (например, сохранение файлов, логирование).  
- **visualisation.py** — графики (матрицы ошибок, ROC-кривые, распределения признаков).  

### 🔹 Корень проекта  
- **main.py** — последовательный запуск пайплайна (EDA → обработка → обучение → оценка).  
- **requirements.txt** — список необходимых библиотек.  
- **TelcoApp.spec** — конфигурация для сборки приложения через PyInstaller.  

---

## Запуск проекта / Run the Project

1. Установить зависимости / Install dependencies:  
```bash
pip install -r requirements.txt
```
2. Запустить анализ и обучение моделей / Train models:
```bash
python main.py
```
3. Для запуска готового приложения / Run GUI app:
```
./dist/TelcoApp/TelcoApp.exe
```
4. Для установки приложения / Build .exe:
```bash
pyinstaller TelcoApp.spec
```



## Установка и запуск / Installation & Usage


1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/username/telco-churn.git
   cd telco-churn
   ```
2. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Запустить обучение моделей:
   ```bash
   python main.py
   ```
4. Запустить Tkinter-приложение:
   ```bash
   python app/main.py
   ```


## Результаты 
- Лучшая модель: **Random Forest** с ROC-AUC ≈ 0.84.  
- Все визуализации сохранены в `reports/figures/`.  
- GUI-приложение позволяет вручную вводить данные клиента и предсказывать вероятность его оттока.  

---

## 👤 Автор 
**Lim Arthur Sergeyevich**  

