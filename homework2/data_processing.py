# data_processing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def count_missing(df):
    """
    Возвращает Series с количеством пропущенных значений в каждом столбце.
    """
    return df.isnull().sum()

def missing_report(df):
    """
    Формирует отчёт о пропущенных значениях: столбец, количество, процент.
    Возвращает DataFrame с отчётом.
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    report = pd.DataFrame({
        'Пропущено': missing_count,
        'Процент': missing_percent
    })
    report = report[report['Пропущено'] > 0].sort_values('Пропущено', ascending=False)
    return report

def fill_missing(df, method='mean', columns=None):
    """
    Заполняет пропущенные значения в указанных столбцах.
    method: 'mean' — среднее, 'median' — медиана, 'mode' — мода.
    columns: список столбцов. Если None — обрабатываются все числовые (для mean/median) или все (для mode).
    Возвращает копию DataFrame с заполненными пропусками.
    """
    df_filled = df.copy()

    if columns is None:
        if method in ['mean', 'median']:
            columns = df_filled.select_dtypes(include=[np.number]).columns.tolist()
        else:  # mode
            columns = df_filled.columns.tolist()

    for col in columns:
        if col not in df_filled.columns:
            print(f"Столбец {col} не найден, пропускаем.")
            continue

        if method == 'mean':
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
            else:
                print(f"Столбец {col} не числовой, для mean пропущены значения не заполнены.")
        elif method == 'median':
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col].fillna(df_filled[col].median(), inplace=True)
            else:
                print(f"Столбец {col} не числовой, для median пропуски не заполнены.")
        elif method == 'mode':
            mode_val = df_filled[col].mode()
            if not mode_val.empty:
                df_filled[col].fillna(mode_val[0], inplace=True)
            else:
                print(f"Не удалось вычислить моду для {col}")
        else:
            print(f"Метод {method} не поддерживается.")
    return df_filled