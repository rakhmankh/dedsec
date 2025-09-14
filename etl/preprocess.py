import pandas as pd
import h3
import os

print("📂 Загружаем датасет...")
df = pd.read_csv("data/clean_trips.csv")

# === 1. Добавляем псевдовремя (для лагов)
df["point_in_trip"] = df.groupby("randomized_id").cumcount()
df["hour"] = df.groupby("randomized_id").ngroup() % 24
df["dayofweek"] = df.groupby("randomized_id").ngroup() % 7

# === 2. Переводим координаты в H3 (resolution 7)
df["h3"] = df.apply(lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], 7), axis=1)

# === 3. Агрегация: количество поездок в ячейке за час ===
agg = df.groupby(["h3", "dayofweek", "hour"]).size().reset_index(name="count_trips")

# === 4. Лаги (предыдущие значения) ===
for lag in [1, 2, 3]:
    agg[f"lag{lag}"] = agg.groupby("h3")["count_trips"].shift(lag)

agg = agg.dropna()

# === 5. Сохраняем результат
os.makedirs("data", exist_ok=True)
agg.to_csv("data/aggregates.csv", index=False)

print("✅ Файл сохранён: data/aggregates.csv")
print("Размер:", agg.shape)
print(agg.head())
