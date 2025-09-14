# etl/aggregate_trips.py
import pandas as pd
import os

def main():
    # Загружаем очищенные данные
    clean_path = "data/clean_trips.csv"
    df = pd.read_csv(clean_path)

    # Извлекаем признаки времени
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0 = Пн

    # Создаём грид (округляем координаты для агрегации)
    df["lat_grid"] = df["latitude"].round(3)
    df["lon_grid"] = df["longitude"].round(3)

    # Агрегация
    agg = (
        df.groupby(["lat_grid", "lon_grid", "hour", "day_of_week"])
        .size()
        .reset_index(name="count_trips")
    )

    # Сохраняем результат
    os.makedirs("data", exist_ok=True)
    agg_path = "data/agg_trips.csv"
    agg.to_csv(agg_path, index=False)

    print(f"✅ Aggregated data saved to {agg_path}, rows: {len(agg)}")

if __name__ == "__main__":
    main()
