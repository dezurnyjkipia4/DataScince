import pandas as pd
import numpy as np

def convert_to_numeric(df, verbose=True):
      df_numeric = df
      conversion_info = {}
    
    # Расширенные словари для преобразования
      binary_mappings = {
        '	automatic': 1, 'manual': 0,
      }
    
      ordinal_mappings = {
        'Standart': 0,'basic': 1,'premium': 2,
        'Sci-Fi': 0, 'Documentary': 1, 'Comedy': 2, 'Action': 3, 'Horror': 4, 'Comedy': 5, 'Drama': 6, 'Horror': 7, 'Romance': 8, 'Sci-Fi': 9, 'Thriller': 10
      }
    
      for col in df.columns:
        original_dtype = df[col].dtype
        
        # Пропускаем уже числовые колонки
        if pd.api.types.is_numeric_dtype(df[col]):
            if verbose:
                print(f"{col}: уже числовая")
            conversion_info[col] = {'type': 'numeric', 'original': original_dtype}
            continue
        
        # Получаем уникальные значения (без пропусков)
        unique_vals = df[col].dropna().unique()
        unique_lower = set(str(v).lower() for v in unique_vals)
        
        # 1. Проверка на бинарные значения
        if len(unique_lower) <= 2 and all(v in binary_mappings or v in ['yes', 'no'] for v in unique_lower):
            # Создаем маппинг для этой колонки
            col_mapping = {}
            for val in unique_vals:
                val_lower = str(val).lower()
                if val_lower in binary_mappings:
                    col_mapping[val] = binary_mappings[val_lower]
                elif val_lower == 'yes':
                    col_mapping[val] = 1
                elif val_lower == 'no':
                    col_mapping[val] = 0
                else:
                    col_mapping[val] = 0  # по умолчанию
            
            df_numeric[col] = df[col].map(col_mapping)
            conversion_info[col] = {
                'type': 'binary',
                'mapping': col_mapping,
                'unique_original': list(unique_vals)
            }
            if verbose:
                print(f"{col}: бинарная -> {col_mapping}")
        
        # 2. Проверка на порядковые значения
        elif any(v in ordinal_mappings for v in unique_lower):
            col_mapping = {}
            for val in unique_vals:
                val_lower = str(val).lower()
                if val_lower in ordinal_mappings:
                    col_mapping[val] = ordinal_mappings[val_lower]
                else:
                    # Если значение не найдено, используем среднее
                    col_mapping[val] = 2
            
            df_numeric[col] = df[col].map(col_mapping)
            conversion_info[col] = {
                'type': 'ordinal',
                'mapping': col_mapping,
                'unique_original': list(unique_vals)
            }
            if verbose:
                print(f"{col}: порядковая -> {col_mapping}")
        
        # 3. Обычные категориальные - используем Label Encoding
        else:
            # Сортируем уникальные значения для согласованности
            sorted_vals = sorted(unique_vals, key=lambda x: str(x))
            col_mapping = {val: i for i, val in enumerate(sorted_vals)}
            
            df_numeric[col] = df[col].map(col_mapping)
            conversion_info[col] = {
                'type': 'categorical',
                'mapping': col_mapping,
                'unique_original': list(unique_vals)
            }
            if verbose:
                print(f"{col}: категориальная -> {len(sorted_vals)} категорий")
        
        print(df_numeric.head())
      return df_numeric

def fill_missing(df, strategy='mean', columns=None) -> None:
        """
        Заполнение пропущенных значений.

        Параметры:
        - strategy: Стратегия заполнения ('mean', 'median', 'mode' или конкретное значение).
        - columns: Список столбцов для обработки (по умолчанию все числовые столбцы).
        """
        if columns is None:
            # Выбираем только числовые столбцы
            columns = df.select_dtypes(include='number').columns

        for col in columns:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = strategy  # Конкретное значение

            df.fillna({col:fill_value}, inplace=True)
