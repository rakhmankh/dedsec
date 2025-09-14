# etl/anomalies.py
import pandas as pd
import numpy as np
import os

def trip_distance(trip_df):
    """Грубая длина маршрута (евклидово расстояние между точками)."""
    coords = trip_df[["latitude", "longitude"]].values
    if len(coords) < 2:
        return 0
    dists = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    return dists.sum()

def has_long_idle(trip_df, idle_points=5, speed_thresh=0.5):
    """Флаг: есть ли длительный простой (скорость ≈ 0 дольше N точек)."""
    idle_mask = trip_df["spd"] < speed_thresh
    max_idle = 0
    cur_idle = 0
    for idle in idle_mask:
        if idle:
            cur_idle += 1
            max_idle = max(max_idle, cur_idle)
        else:
            cur_idle = 0
    return max_idle >= idle_points

def main():
    # Загружаем очищенные данные
    df = pd.read_csv("data/clean_trips.csv")

    anomalies = []

    # Считаем длину и простои для каждой поездки
    for trip_id, group in df.groupby("randomized_id"):
        trip_len = trip_distance(group)
        idle_flag = has_long_idle(group)
        anomalies.append({
            "trip_id": trip_id,
            "length": trip_len,
            "idle": idle_flag
        })

    stats_df = pd.DataFrame(anomalies)

    # Порог для длинных маршрутов
    length_threshold = stats_df["length"].quantile(0.95)

    # Длинные маршруты
    stats_df["anomaly_long"] = stats_df["length"] > length_threshold
    # Простои уже отмечены
    stats_df["anomaly_idle"] = stats_df["idle"]

    # Сохраняем
    os.makedirs("data", exist_ok=True)
    out_path = "data/anomalies.csv"
    stats_df.to_csv(out_path, index=False)

    print(f"✅ Аномалии сохранены в {out_path}, строк: {len(stats_df)}")
    print(f"Порог длины маршрута: {length_threshold:.2f}")

if __name__ == "__main__":
    main()
