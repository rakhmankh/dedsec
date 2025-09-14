# etl/clean_data.py
import pandas as pd
import os

def main():
    # Загружаем исходный датасет
    raw_path = "geo_locations_astana_hackathon/geo_locations_astana_hackathon"
    df = pd.read_csv(raw_path)

    df = df.rename(columns={"lat": "latitude", "lng": "longitude"})

    # Проверим наличие timestamp
    if "timestamp" not in df.columns:
        # Генерируем псевдо-время, равномерно распределённое
        df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="min")

    # Фильтрация данных
    df = df.dropna(subset=["latitude", "longitude"])        # убираем NaN
    df = df[(df["latitude"].between(50, 52)) &              # только Астана
            (df["longitude"].between(71, 72))]
    df = df[df["spd"] >= 0]                                 # скорость >= 0

    # Сохраняем очищенные данные
    os.makedirs("data", exist_ok=True)
    clean_path = "data/clean_trips.csv"
    df.to_csv(clean_path, index=False)

    print(f"✅ Clean data saved to {clean_path}, rows: {len(df)}")

if __name__ == "__main__":
    main()
