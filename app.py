# app.py
import os
import json
import joblib
import pandas as pd
import numpy as np
import requests
import google.generativeai as genai
from datetime import datetime

# FastAPI app (this file provides core functions + the app instance)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CACHE_FILE = "gemini_cache.json"

# Load cache
try:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            ai_cache = json.load(f)
    else:
        ai_cache = {}
except Exception as e:
    print(f"⚠️ Cache load failed: {e}")
    ai_cache = {}

# Configure Gemini (if available)
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"⚠️ Failed to configure Gemini: {e}")
else:
    print("❌ CRITICAL: GOOGLE_API_KEY environment variable is not set!")

print("Libraries import ho gayi hain...")

# --- Load ML Models ---
model = None
scaler = None
encoder = None
try:
    model = joblib.load('model_gbc.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    print("ML Models successfully load ho gaye hain! ✅")
except Exception as e:
    print(f"❌ CRITICAL Error loading models: {e}. The app might not work correctly.")
    model = None
    scaler = None
    encoder = None

# --- State Abbreviations Mapping ---
state_map = {
    "up": "Uttar Pradesh", "mh": "Maharashtra", "hr": "Haryana",
    "ka": "Karnataka", "tn": "Tamil Nadu",
}

# --- Helper Functions ---
def format_city_for_weather(raw_city_state):
    try:
        parts = [p.strip() for p in raw_city_state.split(",")]
        city = parts[0].title()
        state = parts[1].lower() if len(parts) > 1 else ""
        state_full = state_map.get(state, state.title())
        return f"{city},{state_full},IN"
    except Exception:
        return raw_city_state.title()

def safe_json_parse(raw_content, fallback):
    try:
        cleaned = raw_content.strip()
        # Handle markdown code blocks
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except Exception as e:
        print(f"⚠️ JSON parse failed: {e}. Content was: {raw_content[:300] if raw_content else raw_content}")
        return fallback

# --- Live Weather ---
def get_live_weather(city_name):
    print(f"[Agent Research] Searching for live weather in {city_name}...")
    if not WEATHER_API_KEY:
        print("⚠️ WEATHER_API_KEY missing, using defaults.")
        return 28.0, 60.0

    base_url = "https://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={WEATHER_API_KEY}&q={city_name}&units=metric"
    try:
        response = requests.get(complete_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("cod") == 200:
            main = data["main"]
            return main["temp"], main["humidity"]
        return 28.0, 60.0
    except Exception as e:
        print(f"⚠️ Weather fetch error: {e}, using defaults.")
        return 28.0, 60.0

# --- Gemini Wrapper ---
def ask_gemini_ai(prompt_key: str, system_prompt: str, user_prompt: str):
    key_lower = prompt_key.lower().replace(" ", "_")
    if key_lower in ai_cache:
        return ai_cache[key_lower]

    if not GOOGLE_API_KEY:
        # return empty if no key so rest of app can run offline with defaults
        return {}

    try:
        full_prompt = f"{system_prompt}\n\nUser Query: {user_prompt}\n\nIMPORTANT: Return ONLY valid JSON."
        model_ai = genai.GenerativeModel('gemini-pro')
        response = model_ai.generate_content(full_prompt)
        content = getattr(response, "text", None) or getattr(response, "content", None) or ""
        data = safe_json_parse(content, {})

        if data:
            ai_cache[key_lower] = data
            try:
                with open(CACHE_FILE, "w") as f:
                    json.dump(ai_cache, f, indent=2)
            except Exception as e:
                print(f"⚠️ Failed to write cache: {e}")

        return data
    except Exception as e:
        print(f"!!!!!!!!!! GEMINI API CALL FAILED for key: {prompt_key} !!!!!!!!!!!")
        print(f"Error Details: {e}")
        return {}

# --- Soil & Location Interpreter ---
def get_soil_and_location_details(farmer_prompt):
    system_prompt = "Extract city/state and soil type. Return JSON: {city_or_state, soil_type}."
    user_prompt = f"Farmer query: {farmer_prompt}"
    data = ask_gemini_ai(f"soil_location_{farmer_prompt}", system_prompt, user_prompt)
    if not isinstance(data, dict):
        data = {}
    if not data.get("city_or_state"):
        data["city_or_state"] = farmer_prompt
    if not data.get("soil_type"):
        data["soil_type"] = "unknown"
    print("[Agent Reasoning] Extraction complete ✅", data)
    return data

# --- Fill Missing Values ---
def fill_missing_values_ai(details):
    location_soil = f"{details.get('city_or_state','')}, {details.get('soil_type','unknown')}"
    system_prompt = "Estimate N,P,K,pH,rainfall for Indian soils. Return JSON."
    user_prompt = f"Estimate for: {location_soil}"
    ai_values = ask_gemini_ai(f"npk_{location_soil}", system_prompt, user_prompt)
    defaults = {'N':50.0,'P':50.0,'K':50.0,'pH':6.5,'rainfall':400.0}
    for k in defaults:
        try:
            details[k] = float(ai_values.get(k, defaults[k]))
        except Exception:
            details[k] = float(defaults[k])
    print("[AI Estimate] Predicted NPK/pH/rainfall: ", {k: details[k] for k in ['N','P','K','pH','rainfall']})
    return details

# --- ML Prediction ---
def make_prediction(input_data, top_n=5):
    if model is None or scaler is None or encoder is None:
        print("❌ Model or preprocessing not loaded.")
        return []

    df = pd.DataFrame([input_data], columns=model.feature_names_in_)
    scaled = scaler.transform(df)
    original_probs = model.predict_proba(scaled)[0]
    # smoothing
    smoothed = np.sqrt(original_probs)
    total = smoothed.sum()
    if total == 0:
        normalized = original_probs
    else:
        normalized = smoothed / total
    top_idx = np.argsort(normalized)[::-1][:top_n]
    results = []
    for idx in top_idx:
        try:
            crop = encoder.inverse_transform([idx])[0].lower()
        except Exception:
            crop = str(idx)
        results.append((crop, round(normalized[idx]*100,2)))
    return results

# --- Gemini Validation ---
def validate_with_gemini(predictions, location):
    if not GOOGLE_API_KEY:
        return predictions
    crops_list = [p[0] for p in predictions]
    if not crops_list:
        return predictions

    system_prompt = "Strictly validate crop suitability. Return JSON {valid_crops}."
    user_prompt = f"Location: {location}\nCrops: {', '.join(crops_list)}"
    try:
        response = ask_gemini_ai(f"validate_{location}_{'_'.join(crops_list)}", system_prompt, user_prompt)
        valid = set([c.lower() for c in response.get("valid_crops", [])])
        if not valid:
            return predictions
        return [p for p in predictions if p[0].lower() in valid]
    except Exception as e:
        print(f"⚠️ Gemini validation failed: {e}")
        return predictions

# --- Live Prices ---
def get_live_crop_prices():
    system_prompt = "Return mandi prices JSON."
    user_prompt = "Prices for rice,wheat,cotton,jute,coffee,mango,pigeonpeas."
    prices = ask_gemini_ai("crop_prices", system_prompt, user_prompt)
    defaults = {"rice":40,"wheat":35,"cotton":80,"jute":60,"coffee":150,"mango":120,"pigeonpeas":90}
    for k,v in defaults.items():
        if k not in prices:
            prices[k]=v
    return {k:float(v) for k,v in prices.items()}

# --- Future Price AI ---
def get_future_price_ai(crop_name, location):
    if not location:
        location = "India"
    if location.lower() in ["manual location","manual data","unknown"]:
        location = "India"
    system_prompt = "Predict future crop price. Return JSON {'future_price':number}."
    user_prompt = f"Predict 6-month price for {crop_name} in {location}."
    prediction = ask_gemini_ai(f"future_{crop_name}_{location}", system_prompt, user_prompt)
    try:
        val = float(prediction.get("future_price",-1))
        return val/100 if val>0 else -1
    except Exception:
        return -1

# --- Crop Details ---
def get_crop_dynamic_details(crop_name):
    system_prompt="Return crop details JSON."
    user_prompt=f"Details for {crop_name}"
    data = ask_gemini_ai(f"details_{crop_name}",system_prompt,user_prompt)
    defaults={"soil":"loamy","irrigation":"flooding","fertilizer":"NPK 20-20-20","pesticides":"2kg/ha"}
    for k in defaults:
        if k not in data:
            data[k]=defaults[k]
    return data

# --- Crop Rotation ---
def get_crop_rotation_plan(current_crop, location):
    system_prompt="Suggest 2-season rotation. Return JSON."
    user_prompt=f"{current_crop} in {location}"
    return ask_gemini_ai(f"rotation_{current_crop}_{location}",system_prompt,user_prompt)

# --- Grow Guide ---
def get_grow_guide_details(crop_name):
    system_prompt="Give grow guide JSON."
    user_prompt=f"Guide for {crop_name}"
    return ask_gemini_ai(f"grow_{crop_name}",system_prompt,user_prompt)

# --- Ranking ---
def rank_top_3(crop_probs,live_prices,future_prices):
    transport={"rice":80,"wheat":85,"cotton":50,"jute":60,"coffee":40,"mango":30,"pigeonpeas":70}
    if not crop_probs:
        return ["N/A","N/A","N/A"]
    sorted_rev=sorted(crop_probs,key=lambda x:future_prices.get(x[0],0),reverse=True)
    sorted_trans=sorted(crop_probs,key=lambda x:transport.get(x[0],0),reverse=True)
    balanced=[]
    for crop,prob in crop_probs:
        sc=(prob*0.5)+(future_prices.get(crop,0)*0.3)+(transport.get(crop,0)*0.2)
        balanced.append((crop,sc))
    sorted_bal=sorted(balanced,key=lambda x:x[1],reverse=True)
    return [sorted_rev[0][0] if sorted_rev else "N/A", sorted_trans[0][0] if sorted_trans else "N/A", sorted_bal[0][0] if sorted_bal else "N/A"]

# --- Traffic Light Indicator ---
def traffic_light_color(suitability):
    if suitability >= 70: return "Green"
    elif suitability >= 40: return "Yellow"
    else: return "Red"

# --- Display Multi-Crop Comparison (CLI helper) ---
def display_crop_table(crop_probs, live_prices, future_prices):
    transport_score = {"rice": 80, "wheat": 85, "cotton": 50, "jute": 60, "coffee": 40, "mango": 30, "pigeonpeas": 70}
    print("\n📊 Multi-Crop Comparison Table:")
    print("Crop       | Suitability | Live Price | Future Price | Transport | Color")
    print("-----------------------------------------------------------------------------")
    for crop, prob in crop_probs:
        live = live_prices.get(crop, 'N/A')
        future_val = future_prices.get(crop, -1.0)
        future = future_val if future_val > 0 else 'N/A'
        transport = transport_score.get(crop, 0)
        color = traffic_light_color(prob)
        print(f"{crop.title():<10} | {prob:>10}% | Rs {live:<7} | Rs {future:<10} | {transport:>9} | {color}")

# --- Save Results ---
def save_results(input_data, predictions):
    try:
        row = {**input_data, "predictions": str(predictions), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        df = pd.DataFrame([row])
        file_exists = os.path.exists("prediction_logs.csv")
        df.to_csv("prediction_logs.csv", mode="a", header=not file_exists, index=False)
        print("📝 Results logged into prediction_logs.csv ✅")
    except Exception as e:
        print(f"⚠️ Logging failed: {e}")

# ---------------------------
# FASTAPI app initialization
# ---------------------------
app = FastAPI(title="KisanAI-V1 API (core)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve the static frontend from / (make sure static/index.html exists)
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
else:
    print("⚠️ static/ directory not found - ensure your index.html is in static/index.html")

# Minimal health endpoint (useful for Render and basic checks)
@app.get("/health")
def health():
    models_loaded = all([model is not None, scaler is not None, encoder is not None])
    return {"status": "ok", "models_loaded": models_loaded}

# Include API router from api.py (api.py must define `router = APIRouter(...)`)
try:
    from api import router as api_router  # api.py should be the file you already have
    app.include_router(api_router)
    print("✅ api.router included successfully.")
except Exception as e:
    print(f"⚠️ Couldn't include api.router: {e}. Ensure api.py exists and defines 'router'.")

# If you want to run CLI-style testing when executed directly:
if __name__ == "__main__":
    # Simple CLI mode for quick local tests (not used by uvicorn)
    choice = input("\nChoose input method:\n1 → AI Agent\n2 → Manual Data Entry\nEnter 1 or 2: ").strip()
    if choice == "1":
        farmer_input = input("Farmer prompt (village/state + soil type optional): ").strip()
        base = get_soil_and_location_details(farmer_input)
        base = fill_missing_values_ai(base)
        city = format_city_for_weather(base['city_or_state'])
        t, h = get_live_weather(city)
        final_data = {'N': base['N'], 'P': base['P'], 'K': base['K'], 'temperature': t, 'humidity': h, 'ph': base['pH'], 'rainfall': base['rainfall']}
        top = make_prediction(final_data, 5)
        live = get_live_crop_prices()
        future = {c: get_future_price_ai(c, base['city_or_state']) for c,_ in top}
        ranked = rank_top_3(top, live, future)
        display_crop_table(top, live, future)
    else:
        try:
            city_or_state = input("City/State: ").strip()
            N = float(input("N: "))
            P = float(input("P: "))
            K = float(input("K: "))
            ph = float(input("pH: "))
            rainfall = float(input("Rainfall (mm): "))
            temperature = float(input("Temperature (°C): "))
            humidity = float(input("Humidity (%): "))
            final_data = {'N': N, 'P': P, 'K': K, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
            top = make_prediction(final_data, 5)
            live = get_live_crop_prices()
            future = {c: get_future_price_ai(c, city_or_state) for c,_ in top}
            display_crop_table(top, live, future)
        except Exception as e:
            print(f"❌ Manual input failed: {e}")
