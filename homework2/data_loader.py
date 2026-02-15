# data_loader.py
import pandas as pd
import requests
from io import StringIO
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_csv(filepath):
    """
    Загружает данные из CSV файла.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"CSV файл {filepath} успешно загружен. Форма: {df.shape}")
        return df
    except Exception as e:
        print(f"Ошибка при загрузке CSV: {e}")
        return None

def load_json(filepath):
    """
    Загружает данные из JSON файла.
    Предполагается, что JSON содержит список записей или объект, который можно нормализовать.
    """
    try:
        df = pd.read_json(filepath)
        print(f"JSON файл {filepath} успешно загружен. Форма: {df.shape}")
        return df
    except Exception as e:
        print(f"Ошибка при загрузке JSON: {e}")
        return None

def load_from_api(url, params=None):
    """
    Загружает данные из API, возвращающего JSON.
    Можно передать параметры запроса (params).
    """
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # Если API возвращает список записей
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Некоторые API оборачивают данные в поле, например 'results'
            # Попробуем взять первый ключ, содержащий список
            for key, value in data.items():
                if isinstance(value, list):
                    df = pd.DataFrame(value)
                    break
            else:
                df = pd.json_normalize(data)
        else:
            df = pd.DataFrame()
        print(f"Данные из API {url} успешно загружены. Форма: {df.shape}")
        return df
    except Exception as e:
        print(f"Ошибка при загрузке из API: {e}")
        return None