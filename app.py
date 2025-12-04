# app.py
# Merged final app.py: combines features from your api.py and app.py,
# replaces Groq usage with Google Gemini (google.generativeai).
# Keeps all features and endpoints, adds static serving for your UI.
# Do NOT change filenames when deploying: this file should be the one Render runs.

import joblib
import pandas as pd
import numpy as np
import requests
import json
import os
import google.generativeai as genai
from datetime import datetime

# FastAPI + static serving
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------
# CONFIG
# ---------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CACHE_FILE = "gemini_cache.json"   # Use Gemini cache filename

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

# Configure Gemini
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"⚠️ genai.configure failed: {e}")
else:
    print("❌ CRITICAL: GOOGLE_API_KEY environment variable is not set!")

print("Libraries import ho gayi hain...")

# ---------------------------
# Load ML Models
# ---------------------------
try:
    model = joblib.load('model_gbc.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    print("ML Models successfully load ho gaye hain! ✅")
except Exception as e:
    print(f"❌ CRITICAL Error loading models: {e}.")
    # keep going; endpoints should report if model missing

# ---------------------------
# State map & helpers
# ---------------------------
state_map = {
    "up": "Uttar Pradesh", "mh": "Maharashtra", "hr": "Haryana",
    "ka": "Karnataka", "tn": "Tamil Nadu",
}

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
        if raw_content is None:
            return fallback
        cleaned = raw_content.strip()
        # handle markdown codeblocks
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
        cleaned = cleaned.strip()
        # try to fix common problems: single quotes -> double quotes (only if it seems safe)
        try:
            return json.loads(cleaned)
        except Exception:
            # fallback: replace single quotes with double quotes only if looks like JSON object
            alt = cleaned.replace("'", '"')
            return json.loads(alt)
    except Exception as e:
        print(f"⚠️ JSON parse failed: {e}. Content was: {raw_content}")
        return fallback

# ---------------------------
# Live Weather
# ---------------------------
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
        print(f"⚠️ Weather fetch error: {e}")
        return 28.0, 60.0

# ---------------------------
# Gemini wrapper (replacement for Groq)
# ---------------------------
def ask_gemini_ai(prompt_key: str, system_prompt: str, user_prompt: str, model_name: str = "gemini-pro"):
    """
    Returns parsed JSON (dict) as produced by the Gemini response text.
    Results are cached in ai_cache by prompt_key.
    """
    key_lower = prompt_key.lower().replace(" ", "_")
    if key_lower in ai_cache:
        return ai_cache[key_lower]

    if not GOOGLE_API_KEY:
        print("❌ CRITICAL: GOOGLE_API_KEY is not set. Returning empty dict.")
        return {}

    try:
        full_prompt = f"{system_prompt}\n\nUser Query: {user_prompt}\n\nIMPORTANT: Return ONLY valid JSON."
        # Use the GenerativeModel wrapper
        model_obj = genai.GenerativeModel(model_name)
        response = model_obj.generate_content(full_prompt)
        # In some genai versions response may be .text or response.content; handle both
        content = None
        if hasattr(response, "text"):
            content = response.text
        elif hasattr(response, "content"):
            content = response.content
        else:
            # try string conversion
            content = str(response)

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
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        return {}

# ---------------------------
# Domain-specific AI helpers
# ---------------------------
def get_soil_and_location_details(farmer_prompt):
    print("[Agent Reasoning] Understanding farmer's query...")
    system_prompt = "You are an expert Indian agronomist. Your task is to extract the city/state and an optional soil type from a farmer's query. Return ONLY a valid JSON object with keys: 'city_or_state' and 'soil_type'."
    user_prompt = f"Farmer query: '{farmer_prompt}'"
    data = ask_gemini_ai(f"soil_location_{farmer_prompt}", system_prompt, user_prompt)
    if not data.get("city_or_state"): data["city_or_state"] = farmer_prompt
    if not data.get("soil_type"): data["soil_type"] = "unknown"
    print("[Agent Reasoning] Extraction complete ✅", data)
    return data

def fill_missing_values_ai(details):
    print("[AI Estimate] Generating estimated values...")
    location_soil = f"{details.get('city_or_state', '')}, {details.get('soil_type', 'unknown')}"
    system_prompt = "You are an expert agronomist. Your task is to estimate the typical N, P, K, pH, and annual rainfall (in mm) for a given soil type and location in India. Return ONLY a valid JSON object with numeric values for keys: 'N', 'P', 'K', 'pH', 'rainfall'."
    user_prompt = f"Estimate for: {location_soil}"
    ai_values = ask_gemini_ai(f"npk_{location_soil}", system_prompt, user_prompt)
    defaults = {'N': 50.0, 'P': 50.0, 'K': 50.0, 'pH': 6.5, 'rainfall': 400.0}
    for k in defaults:
        try:
            details[k] = float(ai_values.get(k, defaults[k]))
        except Exception:
            details[k] = defaults[k]
    print("[AI Estimate] Predicted NPK/pH/rainfall: ", {k: details[k] for k in ['N','P','K','pH','rainfall']})
    return details

def make_prediction(input_data, top_n=5):
    if 'model' not in globals() or model is None:
        raise RuntimeError("Model not loaded.")
    df = pd.DataFrame([input_data], columns=model.feature_names_in_)
    scaled_features = scaler.transform(df)
    original_probs = model.predict_proba(scaled_features)[0]
    smoothed_probs = np.sqrt(original_probs)
    total_smoothed = np.sum(smoothed_probs)
    if total_smoothed == 0:
        normalized_probs = original_probs
    else:
        normalized_probs = smoothed_probs / total_smoothed
    top_indices = np.argsort(normalized_probs)[::-1][:top_n]
    results = []
    for idx in top_indices:
        crop = encoder.inverse_transform([idx])[0].lower()
        results.append((crop, round(normalized_probs[idx] * 100, 2)))
    return results

def validate_with_gemini(predictions, location):
    if not GOOGLE_API_KEY:
        return predictions
    print(f"[Gemini Double-Check] Validating {len(predictions)} crops for {location}...")
    crops_list = [p[0] for p in predictions]
    if not crops_list: return predictions
    system_prompt = "You are an expert Indian Agronomist. Your task is to VALIDATE a list of proposed crops for a specific location. Strictly filter out any crops that are agronomically impossible or highly unsuitable for the region. Return JSON: { 'valid_crops': [ ... ] }"
    user_prompt = f"Location: {location}\nProposed Crops: {', '.join(crops_list)}"
    try:
        response = ask_gemini_ai(f"validate_{location}_{'_'.join(crops_list)}", system_prompt, user_prompt)
        valid_crops_names = response.get("valid_crops", [])
        if not valid_crops_names:
            print("[Gemini Double-Check] Returned empty list. Keeping original predictions as fallback.")
            return predictions
        valid_set = set(name.lower() for name in valid_crops_names)
        validated_predictions = [p for p in predictions if p[0].lower() in valid_set]
        if not validated_predictions:
            print("[Gemini Double-Check] All crops rejected! Falling back to original.")
            return predictions
        print(f"[Gemini Double-Check] Approved crops: {[p[0] for p in validated_predictions]}")
        return validated_predictions
    except Exception as e:
        print(f"⚠️ Gemini validation failed: {e}")
        return predictions

def get_live_crop_prices():
    print("[Market Research] Fetching live crop prices...")
    system_prompt = "You are a market data provider. Provide current average mandi prices in India (in Rs/kg) for major crops. Return ONLY a valid JSON object in the format: {'rice': 45, 'wheat': 36, ...}"
    user_prompt = "Provide prices for: rice, wheat, cotton, jute, coffee, mango, pigeonpeas."
    prices = ask_gemini_ai("crop_prices", system_prompt, user_prompt)
    defaults = {"rice": 40.0, "wheat": 35.0, "cotton": 80.0, "jute": 60.0, "coffee": 150.0, "mango": 120.0, "pigeonpeas": 90.0}
    for k, v in defaults.items():
        if k not in prices: prices[k] = v
    return {k.lower(): float(v) for k, v in prices.items()}

def get_future_price_ai(crop_name, location):
    if location is None:
        location = "India"
    if isinstance(location, str) and location.lower() in ["manual location", "manual data", "unknown", "मैनुअल डेटा", "मैनुअल स्थान"]:
        location = "India"
    print(f"[AI Price Predictor] Predicting future price for '{crop_name}' in {location}...")
    system_prompt = "You are an expert Indian market analyst. Predict the approximate price (in Rs/quintal) of a crop in a specific region after a typical 6-month harvest period. Return ONLY a valid JSON object with the key 'future_price' and a numeric value."
    user_prompt = f"Predict the 6-month future price (in Rs/quintal) of '{crop_name}' in the '{location}' region."
    prediction = ask_gemini_ai(f"future_price_quintal_{crop_name}_{location}", system_prompt, user_prompt)
    try:
        price_per_quintal = float(prediction.get("future_price", -1.0))
    except Exception:
        price_per_quintal = -1.0
    return price_per_quintal / 100 if price_per_quintal > 0 else -1.0

def get_crop_dynamic_details(crop_name):
    print(f"[AI Details] Fetching details for {crop_name}...")
    system_prompt = "You are an agronomist. Provide structured data for a crop. Return ONLY a valid JSON object with keys: 'soil', 'irrigation', 'fertilizer', 'pesticides'."
    user_prompt = f"Provide details for {crop_name}."
    details = ask_gemini_ai(f"crop_details_{crop_name}", system_prompt, user_prompt)
    defaults = {"soil": "loamy", "irrigation": "flooding", "fertilizer": "NPK 20-20-20", "pesticides": "2kg/ha"}
    for k in defaults:
        if k not in details: details[k] = defaults[k]
    return details

def get_crop_rotation_plan(current_crop, location):
    print(f"[AI Rotation Planner] Generating plan for '{current_crop}' in {location}...")
    system_prompt = "You are an expert Indian agronomist specializing in sustainable farming. Suggest a smart 2-season crop rotation plan after a harvest. Provide a brief reason for each choice. Return ONLY a valid JSON object with keys: 'season_1_crop', 'season_1_reason', 'season_2_crop', 'season_2_reason'."
    user_prompt = f"A farmer in {location}, India just harvested '{current_crop}'. Suggest the next two seasons of crops."
    plan = ask_gemini_ai(f"rotation_{current_crop}_{location}", system_prompt, user_prompt)
    if "season_1_crop" not in plan:
        return {"error": "Could not generate a plan."}
    print("[AI Rotation Planner] Plan ready! ✅")
    return plan

def get_grow_guide_details(crop_name: str):
    print(f"[AI Grow Guide] Generating guide for {crop_name}...")
    system_prompt = "You are an expert Indian agronomist providing a detailed grow guide. Return ONLY a valid JSON object."
    user_prompt = f"""
    Provide a detailed guide for growing '{crop_name}' in India.
    Return ONLY a valid JSON object with the following keys:
    - "description": A short, engaging summary of the crop.
    - "season": The primary growing seasons (e.g., Kharif, Rabi) and ideal planting months.
    - "growth_duration": Typical time from sowing to harvest in days.
    - "irrigation_plan": A practical, brief irrigation schedule.
    - "pesticide_usage": Key pests/diseases and recommended management practices.
    """
    guide_data = ask_gemini_ai(f"grow_guide_{crop_name}", system_prompt, user_prompt)
    # If the AI returned nested object keys directly, return them; otherwise ensure defaults
    defaults = {
        "description": f"No description available for {crop_name}.",
        "season": "Kharif/Rabi (varies)",
        "growth_duration": "Varies",
        "irrigation_plan": "Irrigate as per local needs",
        "pesticide_usage": "As per local extension advice"
    }
    for k in defaults:
        if k not in guide_data:
            guide_data[k] = defaults[k]
    return guide_data

# ---------------------------
# Ranking & utilities
# ---------------------------
def rank_top_3(crop_probs, live_prices, future_prices):
    transport_score = {"rice": 80, "wheat": 85, "cotton": 50, "jute": 60, "coffee": 40, "mango": 30, "pigeonpeas": 70}
    if not crop_probs:
        return {"best_revenue": "N/A", "transport_friendly": "N/A", "balanced_choice": "N/A"}
    sorted_by_revenue = sorted(crop_probs, key=lambda x: future_prices.get(x[0], live_prices.get(x[0], 0)), reverse=True)
    sorted_by_transport = sorted(crop_probs, key=lambda x: transport_score.get(x[0], 0), reverse=True)
    balanced_score = []
    for crop, prob in crop_probs:
        rev_score = future_prices.get(crop, live_prices.get(crop, 0))
        trans_score = transport_score.get(crop, 0)
        score = (prob * 0.5) + (rev_score * 0.3) + (trans_score * 0.2)
        balanced_score.append((crop, score))
    sorted_balanced = sorted(balanced_score, key=lambda x: x[1], reverse=True)
    return {
        "best_revenue": sorted_by_revenue[0][0] if sorted_by_revenue else "N/A",
        "transport_friendly": sorted_by_transport[0][0] if sorted_by_transport else "N/A",
        "balanced_choice": sorted_balanced[0][0] if sorted_balanced else "N/A"
    }

def traffic_light_color(suitability):
    if suitability >= 70: return "Green"
    elif suitability >= 40: return "Yellow"
    else: return "Red"

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
# FastAPI app & endpoints
# ---------------------------
app = FastAPI(title="KisanAI - merged app")

# Serve static UI:
# Prefer mounting static/ directory if you have it (recommended).
# Also provide a fallback root that returns index.html in repo root if present.
static_dir = "static"
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    # Mount static folder path to /static for assets and provide root FileResponse below
    if os.path.isdir("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# If index.html is in project root and static/ not used, serve it:
@app.get("/", include_in_schema=False)
def root_index():
    # If static directory mounted with html=True, it will already handle root,
    # but FileResponse here acts as a fallback.
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    # If static/index.html exists, try that
    if os.path.exists(os.path.join("static", "index.html")):
        return FileResponse(os.path.join("static", "index.html"))
    return {"status": "ok", "message": "UI not found - place index.html in project root or static/ directory."}

class PredictRequest(BaseModel):
    N: float | None = None
    P: float | None = None
    K: float | None = None
    ph: float | None = None
    rainfall: float | None = None
    temperature: float | None = None
    humidity: float | None = None
    farmer_prompt: str | None = None

class GrowGuideRequest(BaseModel):
    crop_name: str

@app.get("/health")
def health():
    models_loaded = 'model' in globals() and model is not None
    return {"status":"ok","models_loaded": models_loaded}

# For compatibility with frontend (manual prediction endpoint)
@app.post("/predict/manual")
def predict_manual(req: PredictRequest):
    try:
        data = {
            'N': req.N or 50,
            'P': req.P or 50,
            'K': req.K or 50,
            'ph': req.ph or 6.5,
            'rainfall': req.rainfall or 400,
            'temperature': req.temperature or 28,
            'humidity': req.humidity or 60
        }
        top = make_prediction(data, top_n=5)
        live = get_live_crop_prices()
        future = {c: get_future_price_ai(c, "India") for c,_ in top}
        validated = validate_with_gemini(top, "India")
        ranked = rank_top_3(top, live, future)
        # Build return in UI-friendly format similar to what UI expects
        comparison_table = []
        for crop, prob in top:
            comparison_table.append({
                "crop": crop,
                "suitabilit
