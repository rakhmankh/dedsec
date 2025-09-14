# etl/visualize.py
import pandas as pd
import numpy as np
import h3
from geopy.distance import geodesic
from scipy.optimize import linear_sum_assignment
import os

print("📂 Загружаем прогноз...")
df_pred = pd.read_csv("data/predictions.csv")

# Берём топ-10 зон с самым высоким спросом
hotspots = df_pred.sort_values("predicted", ascending=False).head(10).copy()
hotspots["lat"] = hotspots["h3"].apply(lambda h: h3.cell_to_latlng(h)[0])
hotspots["lng"] = hotspots["h3"].apply(lambda h: h3.cell_to_latlng(h)[1])

# Параметр: сколько водителей генерировать
N_DRIVERS = 100

print(f"🚖 Генерируем {N_DRIVERS} водителей...")
drivers = pd.DataFrame({
    "driver_id": [f"d{i}" for i in range(N_DRIVERS)],
    "lat": np.random.uniform(51.0, 51.25, N_DRIVERS),   # Астана bbox
    "lng": np.random.uniform(71.3, 71.6, N_DRIVERS)
})

# Строим матрицу стоимости (distance / demand)
cost_matrix = np.zeros((len(drivers), len(hotspots)))
for i, d in drivers.iterrows():
    for j, z in hotspots.iterrows():
        dist = geodesic((d.lat, d.lng), (z.lat, z.lng)).km
        cost_matrix[i, j] = dist / (z.predicted + 1)

# Решаем задачу оптимального распределения
print("⚡ Оптимизируем распределение водителей...")
row_ind, col_ind = linear_sum_assignment(cost_matrix)

assignments = []
for r, c in zip(row_ind, col_ind):
    driver = drivers.iloc[r]
    zone = hotspots.iloc[c]
    assignments.append({
        "driver_id": driver.driver_id,
        "driver_lat": driver.lat,
        "driver_lng": driver.lng,
        "zone_h3": zone.h3,
        "zone_lat": zone.lat,
        "zone_lng": zone.lng,
        "distance_km": geodesic((driver.lat, driver.lng), (zone.lat, zone.lng)).km,
        "zone_predicted": zone.predicted
    })

df_assign = pd.DataFrame(assignments)

# Сохраняем результат
os.makedirs("data", exist_ok=True)
df_assign.to_csv("data/assignments.csv", index=False)

print(f"✅ Оптимизация завершена. Результат сохранён: data/assignments.csv (строк: {len(df_assign)})")
print(df_assign.head())
