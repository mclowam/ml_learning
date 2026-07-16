import io
import os
import pandas as pd
import requests

DATA_URL = "https://raw.githubusercontent.com/Vijaysinh-Lendave/cardiac-arrest-prediction/main/cardio_train.csv"

try:
    response = requests.get(DATA_URL)

    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text), sep=';')

        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        train_df.to_csv('ssz_train.csv', index=False)
        test_df.to_csv('ssz_test.csv', index=False)

    else:
        print(f"Ошибка при скачивании. Статус ответа сервера: {response.status_code}")

except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")