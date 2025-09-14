from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import random
import os
from geopy.distance import geodesic

app = FastAPI(title="inDrive Atlas API")

# ==== Static ====
app.mount("/static", StaticFiles(directory="inDriveAtlas/decentrathon"), name="static")

# ==== Paths ====
DATA_DIR = "data"
CLEAN_PATH = os.path.join(DATA_DIR, "clean_trips.csv")
AGG_PATH = os.path.join(DATA_DIR, "agg_trips.csv")
PRED_PATH = os.path.join(DATA_DIR, "predictions.csv")
ASSIGN_PATH = os.path.join(DATA_DIR, "assignments.csv")

# ==== Root ====
@app.get("/")
async def root():
    return FileResponse("inDriveAtlas/decentrathon/decentrathon.html")


# 🚕 Симуляция поездки
@app.get("/api/simulate")
async def simulate_trip():
    if not os.path.exists(CLEAN_PATH):
        return {"error": "clean_trips.csv не найден"}
    df = pd.read_csv(CLEAN_PATH)[["randomized_id", "latitude", "longitude", "timestamp"]]

    trip_id = random.choice(df["randomized_id"].unique())
    trip = df[df["randomized_id"] == trip_id].to_dict(orient="records")

    total_distance = 0
    for i in range(1, len(trip)):
        p1 = (trip[i - 1]["latitude"], trip[i - 1]["longitude"])
        p2 = (trip[i]["latitude"], trip[i]["longitude"])
        total_distance += geodesic(p1, p2).km

    return {
        "trip_id": str(trip_id),
        "points": trip,
        "total_distance_km": round(total_distance, 2),
    }


# 🔥 Горячие зоны
@app.get("/api/hotzones")
async def hotzones(limit: int = 2000):
    if not os.path.exists(AGG_PATH):
        return {"error": "agg_trips.csv не найден"}
    df = pd.read_csv(AGG_PATH)
    return df.head(limit).to_dict(orient="records")


# 📈 Аналитика (факт vs прогноз)
@app.get("/api/analytics")
async def analytics(hour: int = Query(12), day: int = Query(0)):
    if not (os.path.exists(AGG_PATH) and os.path.exists(PRED_PATH)):
        return {"error": "Нет данных"}
    fact = pd.read_csv(AGG_PATH)
    pred = pd.read_csv(PRED_PATH)

    fact = fact[(fact["hour"] == hour) & (fact["day_of_week"] == day)]
    pred = pred[(pred["hour"] == hour) & (pred["dayofweek"] == day)]

    return {
        "fact": fact.to_dict(orient="records"),
        "pred": pred.to_dict(orient="records"),
    }


# 📊 Будни vs Выходные
@app.get("/api/weekday_vs_weekend")
async def weekday_vs_weekend():
    if not os.path.exists(AGG_PATH):
        return {"error": "agg_trips.csv не найден"}
    df = pd.read_csv(AGG_PATH)
    df["day_type"] = df["day_of_week"].apply(lambda x: "Будни" if x < 5 else "Выходные")
    hourly = df.groupby(["day_type", "hour"])["count_trips"].mean().reset_index()
    return hourly.to_dict(orient="records")


# 🤔 What-if сценарий
@app.get("/api/whatif")
async def whatif(hour: int = Query(18), increase: int = Query(50)):
    if not os.path.exists(AGG_PATH):
        return {"error": "agg_trips.csv не найден"}
    df = pd.read_csv(AGG_PATH)
    base_data = df[df["hour"] == hour].copy()
    base_data["scenario_trips"] = base_data["count_trips"] * (1 + increase / 100)
    return base_data.to_dict(orient="records")


# 🚨 Аномалии
@app.get("/api/anomalies")
async def anomalies(limit: int = 2000):
    if not os.path.exists(CLEAN_PATH):
        return {"error": "clean_trips.csv не найден"}
    df = pd.read_csv(CLEAN_PATH).head(limit)

    trips = []
    for trip_id, trip in df.groupby("randomized_id"):
        trip = trip.sort_values("timestamp")
        total_dist = 0
        for i in range(1, len(trip)):
            p1 = (trip.iloc[i - 1]["latitude"], trip.iloc[i - 1]["longitude"])
            p2 = (trip.iloc[i]["latitude"], trip.iloc[i]["longitude"])
            total_dist += geodesic(p1, p2).km
        avg_speed = trip["spd"].mean() if "spd" in trip else 0
        trips.append({
            "trip_id": str(trip_id),
            "distance_km": round(total_dist, 2),
            "avg_speed": round(avg_speed, 2),
            "n_points": len(trip),
            "is_anomaly": total_dist > 30 or avg_speed > 120,
            "path": trip[["latitude", "longitude"]].to_dict(orient="records")
        })
    return trips


# 📲 Рекомендации водителям
@app.get("/api/recommendations")
async def recommendations():
    if not os.path.exists(ASSIGN_PATH):
        return {"error": "assignments.csv не найден"}
    df = pd.read_csv(ASSIGN_PATH)
    return df.to_dict(orient="records")
