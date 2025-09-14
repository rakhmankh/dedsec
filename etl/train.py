# etl/train.py
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import os

print("📂 Загружаем агрегаты...")
df = pd.read_csv("data/aggregates.csv")

# Проверяем, какие признаки реально есть
expected_features = ["hour", "dayofweek", "lag1", "lag2", "lag3"]
features = [f for f in expected_features if f in df.columns]

if not features:
    raise ValueError("❌ В агрегатах нет нужных признаков! Проверь preprocess.py")

X, y = df[features], df["count_trips"]

# train/test split (по времени, без перемешивания)
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"🚀 Обучаем модель (признаки: {features})...")
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Прогноз и метрика
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Обучение завершено. MAE = {mae:.2f}")

# Сохраняем модель
os.makedirs("data", exist_ok=True)
joblib.dump(model, "data/model.pkl")
print("💾 Модель сохранена: data/model.pkl")

# Делаем прогноз на "последний срез" (как будто это будущее)
latest = df.groupby("h3").tail(1).copy()
latest["predicted"] = model.predict(latest[features])

# Сохраняем прогноз
latest[["h3", "hour", "dayofweek", "count_trips", "predicted"]].to_csv(
    "data/predictions.csv", index=False
)
print("✅ Прогноз сохранён: data/predictions.csv")
print(latest.head())
