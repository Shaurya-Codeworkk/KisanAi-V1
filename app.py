# app.py
# Merged application: FastAPI server + Gemini wrapper + ML prediction pipeline
# Replace your current app.py with this file.

import os
import json
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Gemini
import google.generativeai as genai

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# -------------------------
# Configuration
# -------------------------
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

# Configure Gemini
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"⚠️ genai.configure failed: {e}")
else:
    print("❌ CRITICAL: GOOGLE_API_KEY environment variable is not set!")

print("Libraries import ho gayi hain...")

# -------------------------
# Load ML models
# -------------------------
model = None
scaler = None
encoder = None

try:
    model = joblib.load('model_gbc.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    print("ML Models successfully load ho gaye hain! ✅")
except Exception as e:
    print(f"❌ CRITICAL Error loading models: {e}.")

# -------------------------
# Helpers & domain logic
# -------------------------
state_map = {
    "up": "Uttar Pradesh", "mh": "Maharashtra", "hr": "Haryana",
    "ka": "Karnataka", "tn": "Tamil Nadu",
}

def format_city_for_weather(raw_city_state: str) -> str:
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
        cleaned = str(raw_content).strip()
        # if markdown codeblock present, extract inner JSON
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
        cleaned = cleaned.strip()
        # first try direct json.loads
        try:
            return json.loads(cleaned)
        except Exception:
            # fallback: replace single quotes with double quotes (best-effort)
            alt = cleaned.replace("'", '"')
            return json.loads(alt)
    except Exception as e:
        print(f"⚠️ JSON parse failed: {e}. Content was: {raw_content}")
        return fallback

def get_live_weather(city_name: str):
    print(f"[Agent Research] Searching for live weather in {city_name}...")
    if not WEATHER_API_KEY:
        print("⚠️ WEATHER_API_KEY missing, using defaults.")
        return 28.0, 60.0
    base_url = "https://api.openweathermap.org/data/2.5/weather?"
    url = f"{base_url}appid={WEATHER_API_KEY}&q={city_name}&units=metric"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("cod") == 200:
            main = data["main"]
            return main.get("temp", 28.0), main.get("humidity", 60.0)
        return 28.0, 60.0
    except Exception as e:
        print(f"⚠️ Weather fetch error: {e}")
        return 28.0, 60.0

def ask_gemini_ai(prompt_key: str, system_prompt: str, user_prompt: str, model_name: str = "gemini-pro"):
    key_lower = prompt_key.lower().replace(" ", "_")
    if key_lower in ai_cache:
        return ai_cache[key_lower]
    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not set; returning empty dict.")
        return {}
    try:
        full_prompt = f"{system_prompt}\n\nUser Query: {user_prompt}\n\nIMPORTANT: Return ONLY valid JSON."
        model_obj = genai.GenerativeModel(model_name)
        response = model_obj.generate_content(full_prompt)
        content = getattr(response, "text", str(response))
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
        print(f"⚠️ Gemini call failed for key={prompt_key}: {e}")
        return {}

def get_soil_and_location_details(farmer_prompt: str):
    system_prompt = "You are an expert Indian agronomist. Extract 'city_or_state' and optional 'soil_type'. Return ONLY JSON."
    user_prompt = f"Farmer query: '{farmer_prompt}'"
    data = ask_gemini_ai(f"soil_location_{farmer_prompt}", system_prompt, user_prompt)
    if not data.get("city_or_state"):
        data["city_or_state"] = farmer_prompt
    if not data.get("soil_type"):
        data["soil_type"] = "unknown"
    return data

def fill_missing_values_ai(details: dict):
    location_soil = f"{details.get('city_or_state','')}, {details.get('soil_type','unknown')}"
    system_prompt = "Estimate N,P,K,pH,rainfall for Indian soils. Return ONLY JSON with numeric keys."
    user_prompt = f"Estimate for: {location_soil}"
    ai_values = ask_gemini_ai(f"npk_{location_soil}", system_prompt, user_prompt)
    defaults = {'N':50.0,'P':50.0,'K':50.0,'pH':6.5,'rainfall':400.0}
    for k in defaults:
        try:
            details[k] = float(ai_values.get(k, defaults[k]))
        except Exception:
            details[k] = defaults[k]
    return details

def make_prediction(input_data: dict, top_n: int = 5):
    if model is None:
        raise RuntimeError("Model not loaded.")
    df = pd.DataFrame([input_data], columns=model.feature_names_in_)
    scaled = scaler.transform(df)
    probs = model.predict_proba(scaled)[0]
    smoothed = np.sqrt(probs)
    total = smoothed.sum()
    normalized = probs if total == 0 else (smoothed / total)
    top_idx = np.argsort(normalized)[::-1][:top_n]
    results = []
    for idx in top_idx:
        crop = encoder.inverse_transform([idx])[0].lower()
        results.append((crop, round(normalized[idx]*100, 2)))
    return results

def validate_with_gemini(predictions, location):
    if not GOOGLE_API_KEY:
        return predictions
    crops_list = [p[0] for p in predictions]
    if not crops_list:
        return predictions
    system_prompt = "Strictly validate crop suitability. Return JSON {'valid_crops': [...]}."
    user_prompt = f"Location: {location}\nCrops: {', '.join(crops_list)}"
    response = ask_gemini_ai(f"validate_{location}_{'_'.join(crops_list)}", system_prompt, user_prompt)
    valid = set([c.lower() for c in response.get("valid_crops", [])]) if isinstance(response, dict) else set()
    if not valid:
        return predictions
    validated = [p for p in predictions if p[0] in valid]
    return validated if validated else predictions

def get_live_crop_prices():
    system_prompt = "Return mandi prices JSON in Rs/kg for main crops."
    user_prompt = "Prices for rice,wheat,cotton,jute,coffee,mango,pigeonpeas."
    prices = ask_gemini_ai("crop_prices", system_prompt, user_prompt)
    defaults = {"rice":40.0,"wheat":35.0,"cotton":80.0,"jute":60.0,"coffee":150.0,"mango":120.0,"pigeonpeas":90.0}
    for k,v in defaults.items():
        if k not in prices:
            prices[k] = v
    return {k: float(v) for k,v in prices.items()}

def get_future_price_ai(crop_name, location):
    if not location or (isinstance(location, str) and location.lower() in ["manual location","manual data","unknown"]):
        location = "India"
    system_prompt = "Predict 6-month future price in Rs/quintal. Return JSON {'future_price': number}."
    user_prompt = f"Predict future price for {crop_name} in {location}"
    prediction = ask_gemini_ai(f"future_{crop_name}_{location}", system_prompt, user_prompt)
    try:
        val = float(prediction.get("future_price", -1))
    except Exception:
        val = -1
    return val/100 if val > 0 else -1

def get_crop_rotation_plan(current_crop, location):
    system_prompt = "Suggest 2-season rotation. Return JSON with season_1_crop, season_1_reason, season_2_crop, season_2_reason."
    user_prompt = f"{current_crop} in {location}"
    plan = ask_gemini_ai(f"rotation_{current_crop}_{location}", system_prompt, user_prompt)
    if "season_1_crop" not in plan:
        return {"error":"Could not generate a plan."}
    return plan

def get_grow_guide_details(crop_name: str):
    system_prompt = "Provide grow guide JSON with keys: description, season, growth_duration, irrigation_plan, pesticide_usage."
    user_prompt = f"Guide for {crop_name}"
    guide = ask_gemini_ai(f"grow_guide_{crop_name}", system_prompt, user_prompt)
    defaults = {
        "description": f"No description for {crop_name}.",
        "season": "Varies",
        "growth_duration": "Varies",
        "irrigation_plan": "Local recommendations apply",
        "pesticide_usage": "Local recommendations apply"
    }
    for k in defaults:
        if k not in guide:
            guide[k] = defaults[k]
    return guide

def rank_top_3(crop_probs, live_prices, future_prices):
    transport = {"rice":80,"wheat":85,"cotton":50,"jute":60,"coffee":40,"mango":30,"pigeonpeas":70}
    if not crop_probs:
        return {"best_revenue":"N/A","transport_friendly":"N/A","balanced_choice":"N/A"}
    sorted_rev = sorted(crop_probs, key=lambda x: future_prices.get(x[0], live_prices.get(x[0],0)), reverse=True)
    sorted_trans = sorted(crop_probs, key=lambda x: transport.get(x[0],0), reverse=True)
    balanced = []
    for crop, prob in crop_probs:
        rev = future_prices.get(crop, live_prices.get(crop,0))
        trans = transport.get(crop,0)
        score = (prob * 0.5) + (rev * 0.3) + (trans * 0.2)
        balanced.append((crop, score))
    sorted_bal = sorted(balanced, key=lambda x: x[1], reverse=True)
    return {
        "best_revenue": sorted_rev[0][0] if sorted_rev else "N/A",
        "transport_friendly": sorted_trans[0][0] if sorted_trans else "N/A",
        "balanced_choice": sorted_bal[0][0] if sorted_bal else "N/A"
    }

def traffic_light_color(suitability):
    if suitability >= 70: return "Green"
    elif suitability >= 40: return "Yellow"
    else: return "Red"

def save_results(input_data, predictions):
    try:
        row = {**input_data, "predictions": str(predictions), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        df = pd.DataFrame([row])
        df.to_csv("prediction_logs.csv", mode="a", header=not os.path.exists("prediction_logs.csv"), index=False)
    except Exception as e:
        print(f"⚠️ Logging failed: {e}")

# -------------------------
# FastAPI server
# -------------------------
app = FastAPI(title="KisanAI-Merged")

# Static serving: prefer static/ directory
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
else:
    # still allow assets under /static if present
    if os.path.isdir("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", include_in_schema=False)
def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    if os.path.exists(os.path.join("static","index.html")):
        return FileResponse(os.path.join("static","index.html"))
    return {"status":"ok","message":"UI not found. Put index.html in project root or static/."}

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
    return {"status":"ok","models_loaded": (model is not None)}

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
        top = make_prediction(data, 5)
        live = get_live_crop_prices()
        future = {c: get_future_price_ai(c, "India") for c,_ in top}
        validated = validate_with_gemini(top, "India")
        ranked = rank_top_3(top, live, future)
        comparison_table = []
        for crop, prob in top:
            comparison_table.append({
                "crop": crop,
                "suitability_percent": prob,
                "recommendation_color": traffic_light_color(prob),
                "live_price_rs_kg": live.get(crop, "N/A"),
                "predicted_future_price_rs_kg": future.get(crop, -1.0)
            })
        save_results(data, top)
        return {
            "input_parameters": data,
            "top_crops": top,
            "validated": validated,
            "live_prices": live,
            "future_prices": future,
            "ranked_top3": ranked,
            "comparison_table": comparison_table,
            "top_3_recommendations": {
                "best_revenue": ranked["best_revenue"],
                "transport_friendly": ranked["transport_friendly"],
                "balanced_choice": ranked["balanced_choice"]
            },
            "smart_rotation_plan": get_crop_rotation_plan(ranked["balanced_choice"], "India")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/agent")
def predict_agent(payload: dict):
    try:
        prompt = payload.get("prompt", "") if payload else ""
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing 'prompt'.")
        base_details = get_soil_and_location_details(prompt)
        base_details = fill_missing_values_ai(base_details)
        city = format_city_for_weather(base_details['city_or_state'])
        temp, hum = get_live_weather(city)
        final_data = {
            'N': base_details['N'], 'P': base_details['P'], 'K': base_details['K'],
            'temperature': temp, 'humidity': hum,
            'ph': base_details['pH'], 'rainfall': base_details['rainfall']
        }
        top_crops = make_prediction(final_data, 5)
        live = get_live_crop_prices()
        future = {c: get_future_price_ai(c, base_details['city_or_state']) for c,_ in top_crops}
        validated = validate_with_gemini(top_crops, base_details['city_or_state'])
        ranked = rank_top_3(top_crops, live, future)
        comparison_table = []
        for crop, prob in top_crops:
            comparison_table.append({
                "crop": crop,
                "suitability_percent": prob,
                "recommendation_color": traffic_light_color(prob),
                "live_price_rs_kg": live.get(crop, "N/A"),
                "predicted_future_price_rs_kg": future.get(crop, -1.0)
            })
        save_results(final_data, top_crops)
        return {
            "input_parameters": final_data,
            "location_analyzed": base_details['city_or_state'],
            "top_crops": top_crops,
            "validated": validated,
            "live_prices": live,
            "future_prices": future,
            "ranked_top3": ranked,
            "comparison_table": comparison_table,
            "top_3_recommendations": {
                "best_revenue": ranked["best_revenue"],
                "transport_friendly": ranked["transport_friendly"],
                "balanced_choice": ranked["balanced_choice"]
            },
            "smart_rotation_plan": get_crop_rotation_plan(ranked["balanced_choice"], base_details['city_or_state'])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/grow_guide")
def api_grow(req: GrowGuideRequest):
    try:
        guide = get_grow_guide_details(req.crop_name)
        return guide
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mini_chat")
def mini_chat(payload: dict):
    try:
        query = payload.get("query", "") if payload else ""
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query'.")
        system_prompt = "You are KisanAI Mini: short helpful farming answers. Return plain text."
        # Try structured call first
        resp = ask_gemini_ai(f"mini_{query}", system_prompt, query)
        if isinstance(resp, dict):
            for k in ("answer","response","text","reply"):
                if k in resp:
                    return {"response": str(resp[k])}
            vals = [str(v) for v in resp.values() if isinstance(v, (str,int,float))]
            if vals:
                return {"response": " ".join(vals)}
        # fallback to direct plain text generation
        try:
            model_obj = genai.GenerativeModel("gemini-pro")
            direct = model_obj.generate_content(f"{system_prompt}\n\nUser Query: {query}\n\nIMPORTANT: Return a short plain-text answer.")
            text = getattr(direct, "text", str(direct))
            return {"response": text}
        except Exception:
            return {"response": "Sorry, I couldn't fetch an answer right now."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# CLI mode (optional)
# -------------------------
if __name__ == "__main__":
    try:
        choice = input("\nChoose input method:\n1 → AI Agent\n2 → Manual Data Entry\nEnter 1 or 2: ").strip()
        if choice == "1":
            farmer_input = input("Farmer prompt: ")
            base = get_soil_and_location_details(farmer_input)
            base = fill_missing_values_ai(base)
            city = format_city_for_weather(base['city_or_state'])
            temp, hum = get_live_weather(city)
            final = {
                'N': base['N'], 'P': base['P'], 'K': base['K'],
                'temperature': temp, 'humidity': hum,
                'ph': base['pH'], 'rainfall': base['rainfall']
            }
            top = make_prediction(final, 5)
            live = get_live_crop_prices()
            future = {c: get_future_price_ai(c, base['city_or_state']) for c,_ in top}
            display = top
            display_crop_table(top, live, future)
        else:
            city_or_state = input("City/State: ").strip()
            N = float(input("Nitrogen (N): "))
            P = float(input("Phosphorus (P): "))
            K = float(input("Potassium (K): "))
            pH = float(input("pH value: "))
            rainfall = float(input("Rainfall (mm/year): "))
            temperature = float(input("Temperature (°C): "))
            humidity = float(input("Humidity (%): "))
            final = {'N':N,'P':P,'K':K,'ph':pH,'rainfall':rainfall,'temperature':temperature,'humidity':humidity}
            top = make_prediction(final, 5)
            live = get_live_crop_prices()
            future = {c: get_future_price_ai(c, city_or_state) for c,_ in top}
            display_crop_table(top, live, future)
    except Exception as e:
        print(f"❌ CLI error: {e}")
