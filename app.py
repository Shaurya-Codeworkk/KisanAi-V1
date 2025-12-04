# app.py (import-safe core functions only)
# Replace your current app.py with this full content.

import os
import json
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Optional AI libs (imported guarded)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# --- CONFIG ---
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CACHE_FILE = "ai_cache.json"

# --- Load or create cache ---
_ai_cache = {}
try:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            _ai_cache = json.load(f)
except Exception as e:
    print("app.py: cache load failed:", e)
    _ai_cache = {}

# --- ML models (guarded load) ---
model = None
scaler = None
encoder = None
try:
    model = joblib.load("model_gbc.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    print("app.py: ML models loaded successfully")
except Exception as e:
    print("app.py: ML model load failed (ok for now):", e)
    model = None
    scaler = None
    encoder = None

# --- small utilities ---
state_map = {"up": "Uttar Pradesh", "mh": "Maharashtra", "hr": "Haryana", "ka": "Karnataka", "tn": "Tamil Nadu"}

def format_city_for_weather(raw_city_state: str) -> str:
    try:
        parts = [p.strip() for p in str(raw_city_state).split(",")]
        city = parts[0].title() if parts and parts[0] else str(raw_city_state).title()
        state = parts[1].lower() if len(parts) > 1 else ""
        state_full = state_map.get(state, state.title() if state else "")
        if state_full:
            return f"{city},{state_full},IN"
        return f"{city},IN"
    except Exception:
        return str(raw_city_state).title()

def safe_json_parse(raw_content, fallback):
    try:
        if isinstance(raw_content, dict):
            return raw_content
        if not isinstance(raw_content, str):
            return fallback
        cleaned = raw_content.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except Exception as e:
        print("app.safe_json_parse failed:", e)
        return fallback

def get_live_weather(city_name: str):
    """Return (temperature_c, humidity_percent). Uses OpenWeatherMap if key present, otherwise defaults."""
    if not WEATHER_API_KEY:
        print("app.get_live_weather: WEATHER_API_KEY not set; using defaults")
        return 28.0, 60.0
    try:
        base_url = "https://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}appid={WEATHER_API_KEY}&q={city_name}&units=metric"
        resp = requests.get(complete_url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        main = data.get("main", {})
        temp = float(main.get("temp", 28.0))
        hum = float(main.get("humidity", 60.0))
        return temp, hum
    except Exception as e:
        print("app.get_live_weather error:", e)
        return 28.0, 60.0

# --- Minimal AI wrapper fallback (used only if mini_kisan_copilot not present) ---
def ask_ai_stub(prompt_key: str, system_prompt: str, user_prompt: str):
    key = prompt_key.lower().replace(" ", "_")
    if key in _ai_cache:
        return _ai_cache[key]
    _ai_cache[key] = {}
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_ai_cache, f, indent=2)
    except Exception:
        pass
    return {}

# --- Soil & Location Interpreter ---
def get_soil_and_location_details(farmer_prompt: str):
    try:
        if not farmer_prompt:
            return {"city_or_state": "India", "soil_type": "unknown"}
        text = str(farmer_prompt).strip()
        parts = [p.strip() for p in text.split(",")]
        city = parts[0].title() if parts and parts[0] else text.title()
        soil = "unknown"
        ltext = text.lower()
        if "red" in ltext: soil = "red"
        elif "black" in ltext or "regur" in ltext: soil = "black"
        elif "sandy" in ltext: soil = "sandy"
        return {"city_or_state": city, "soil_type": soil}
    except Exception as e:
        print("app.get_soil_and_location_details error:", e)
        return {"city_or_state": str(farmer_prompt), "soil_type": "unknown"}

# --- Fill Missing Values with safe defaults ---
def fill_missing_values_ai(details: dict):
    defaults = {'N': 50.0, 'P': 50.0, 'K': 50.0, 'pH': 6.5, 'rainfall': 400.0}
    for k, v in defaults.items():
        if k not in details or details.get(k) in [None, "", 0]:
            details[k] = float(v)
    return details

# --- Prediction ---
def make_prediction(input_data: dict, top_n=5):
    global model, scaler, encoder
    try:
        if model is not None and scaler is not None and encoder is not None:
            df = pd.DataFrame([input_data], columns=model.feature_names_in_)
            scaled = scaler.transform(df)
            probs = model.predict_proba(scaled)[0]
            smoothed = np.sqrt(np.maximum(probs, 0))
            total = smoothed.sum()
            normalized = (smoothed / total) if total > 0 else probs
            idxs = np.argsort(normalized)[::-1][:top_n]
            results = []
            for i in idxs:
                try:
                    crop = encoder.inverse_transform([i])[0].lower() if encoder is not None else f"crop{i}"
                except Exception:
                    crop = f"crop{i}"
                results.append((crop, round(float(normalized[i]) * 100, 2)))
            return results
    except Exception as e:
        print("app.make_prediction (model) error:", e)

    # fallback predictions
    fallback = [("rice", 45.0), ("wheat", 30.0), ("maize", 10.0), ("cotton", 8.0), ("potato", 7.0)]
    return fallback[:top_n]

# --- Live crop prices (fallback) ---
def get_live_crop_prices():
    defaults = {"rice": 40.0, "wheat": 35.0, "cotton": 80.0, "jute": 60.0, "coffee": 150.0, "mango": 120.0, "pigeonpeas": 90.0}
    return {k: float(v) for k, v in defaults.items()}

# --- Future price (simple multiplier fallback) ---
def get_future_price_ai(crop_name, location):
    prices = get_live_crop_prices()
    p = prices.get(crop_name.lower())
    if p:
        return round(p * 1.05, 2)
    return -1.0

# --- Rotation / ranking utils ---
def rank_top_3(crop_probs, live_prices, future_prices):
    transport_score = {"rice": 80, "wheat": 85, "cotton": 50, "jute": 60, "coffee": 40, "mango": 30, "pigeonpeas": 70}
    if not crop_probs:
        return ["N/A", "N/A", "N/A"]
    sorted_by_revenue = sorted(crop_probs, key=lambda x: future_prices.get(x[0], live_prices.get(x[0], 0)), reverse=True)
    sorted_by_transport = sorted(crop_probs, key=lambda x: transport_score.get(x[0], 0), reverse=True)
    balanced = []
    for crop, prob in crop_probs:
        rev = future_prices.get(crop, live_prices.get(crop, 0))
        trans = transport_score.get(crop, 0)
        score = (prob * 0.5) + (rev * 0.3) + (trans * 0.2)
        balanced.append((crop, score))
    sorted_balanced = sorted(balanced, key=lambda x: x[1], reverse=True)
    return [
        sorted_by_revenue[0][0] if sorted_by_revenue else "N/A",
        sorted_by_transport[0][0] if sorted_by_transport else "N/A",
        sorted_balanced[0][0] if sorted_balanced else "N/A"
    ]

def traffic_light_color(suitability):
    if suitability >= 70: return "Green"
    if suitability >= 40: return "Yellow"
    return "Red"

def save_results(input_data, predictions):
    try:
        row = {**input_data, "predictions": str(predictions), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        df = pd.DataFrame([row])
        file_exists = os.path.exists("prediction_logs.csv")
        df.to_csv("prediction_logs.csv", mode="a", header=not file_exists, index=False)
    except Exception as e:
        print("app.save_results error:", e)

# -----------------------------
# CLI / manual run (only when executed directly)
# -----------------------------
if __name__ == "__main__":
    try:
        print("app.py CLI mode — quick test of main flow")
        choice = input("1 → Agent prompt, 2 → Manual: ").strip()
        if choice == "1":
            farmer_input = input("Farmer prompt (city,state + soil): ")
            base = get_soil_and_location_details(farmer_input)
            base = fill_missing_values_ai(base)
            city = format_city_for_weather(base['city_or_state'])
            temp, hum = get_live_weather(city)
            final_data = {
                'N': base['N'], 'P': base['P'], 'K': base['K'],
                'temperature': temp, 'humidity': hum,
                'ph': base['pH'], 'rainfall': base['rainfall']
            }
            top = make_prediction(final_data, top_n=5)
            print("Top crops:", top)
        else:
            N = float(input("N: ") or 50)
            P = float(input("P: ") or 50)
            K = float(input("K: ") or 50)
            ph = float(input("pH: ") or 6.5)
            rainfall = float(input("rainfall: ") or 400)
            temp = float(input("temperature: ") or 28)
            hum = float(input("humidity: ") or 60)
            final_data = {'N': N, 'P': P, 'K': K, 'ph': ph, 'rainfall': rainfall, 'temperature': temp, 'humidity': hum}
            top = make_prediction(final_data, top_n=5)
            print("Top crops (manual):", top)
    except Exception as e:
        print("app.py main-run error:", e)
