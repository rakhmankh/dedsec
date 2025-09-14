# 🚕 inDrive Atlas — Аналитическая панель для поездок

## 📌 О проекте

**inDrive Atlas** — это аналитическая платформа для работы с геотреками поездок.
Мы собрали обезличенные данные, обработали их и построили прототип, который помогает:

* 🔥 Определять **горячие зоны спроса**
* 🚕 Симулировать **поездки в реальном времени**
* 📈 Делать **прогноз спроса (ML-модель LightGBM)**
* 📲 Выдавать **рекомендации для распределения водителей**
* 🚨 Находить **аномальные маршруты**
* 🤔 Анализировать сценарии **Факт vs Прогноз, Будни vs Выходные, What-if**

---

## 🖼️ Скриншоты

### 🔥 Горячие зоны

![photo_2025-09-14_23-22-28](https://github.com/user-attachments/assets/0a2b6a1b-e48b-40bc-a860-7a76264fe8c5)


### 📈 Прогноз спроса

![photo_2025-09-14_23-22-37](https://github.com/user-attachments/assets/1bf85424-24a3-4219-b6e4-69c509b93c95)
![photo_2025-09-14_23-22-34](https://github.com/user-attachments/assets/8055c44f-2952-4d66-a905-2dd4ee988d23)
![photo_2025-09-14_23-22-32](https://github.com/user-attachments/assets/a2450b20-1d2a-4de6-8f26-17f8ff71757b)


### 🚕 Симуляция поездки

![photo_2025-09-14_23-22-17](https://github.com/user-attachments/assets/eda2bc05-e490-4392-98d9-102954f4a67b)

### 🚨 Аномалии

![photo_2025-09-14_23-22-39](https://github.com/user-attachments/assets/f37cae85-a678-4011-a4f3-9cf30f2411d3)


### 📲 Рекомендации водителям

![photo_2025-09-14_23-22-30](https://github.com/user-attachments/assets/4df3c6f5-6e0f-4acf-98b6-4798844aa1ae)


---

## ⚙️ Технологии

* **Python 3.11**
* **Streamlit** — интерфейс и визуализация
* **Pydeck** — карты и 3D визуализация
* **H3** — геоагрегация
* **Scikit-learn + LightGBM** — прогнозирование спроса
* **Geopy + Scipy** — оптимизация распределения водителей

---

## 📂 Структура проекта

```
decentra/
│── data/                # Обработанные данные и результаты (csv, pkl)
│── etl/                 # ETL-скрипты (preprocess, train, visualize, anomalies)
│── WEB-SITE WITH DESIGN # HTML/CSS/JS — будущий сайт (версия для жюри)
│── main.py              # Основное Streamlit-приложение
│── requirements.txt     # Зависимости
│── README.md            # Документация
```

---

## 🚀 Запуск проекта

### 1. Клонировать репозиторий

```bash
git clone https://github.com/rakhmankh/dedsec.git
cd inDriveAtlas
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Запустить приложение

```bash
streamlit run main.py
```

---

## 🌍 Будущее развитие

* Перевод на **FastAPI** с интеграцией фронтенда (HTML/CSS/JS из `WEB-SITE WITH DESIGN`)
* Интеграция с **Kafka/Flink** для real-time обработки поездок
* Более сложные модели (TFT, Seq2Seq) для прогноза
* Масштабирование на другие города

---

💡 Проект создан в рамках **InDrive Hackathon 2025**. Команда: dedsec
Хамзанов Рахман
Нурбек Медина
Хайруллин Рамазан
Берік Дәурен
