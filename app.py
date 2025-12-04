# app.py  (FULL merged version)
import os
import json
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Gemini (Google) client
import google.generativeai as genai

# FastAPI and utils
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- CONFIG ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")       # for Gemini (optional)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")     # for OpenWeather (optional)
CACHE_FILE = "gemini_cache.json"

# --- Load / init cache for AI responses ---
try:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            ai_cache = json.load(f)
    else:
        ai_cache = {}
except Exception as e:
    print(f"⚠️ Cache load failed: {e}")
    ai_cache = {}

# Configure Gemini if API key present
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print("❌ Failed to configure Gemini:", e)
else:
    print("⚠️ GOOGLE_API_KEY not set — Gemini features will return defaults.")

print("Libraries import ho gayi hain...")

# --- Load ML models (safe) ---
model = None
scaler = None
encoder = None
try:
    model = joblib.load("model_gbc.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    print("ML Models successfully load ho gaye hain! ✅")
except Exception as e:
    print(f"❌ Error loading models: {e}. Predictions will use defaults or fail gracefully.")

# --- state map ---
state_map = {
    "up": "Uttar Pradesh", "mh": "Maharashtra", "hr": "Haryana",
    "ka": "Karnataka", "tn": "Tamil Nadu",
}

# --- Helpers ---
def format_city_for_weather(raw_city_state: str) -> str:
    try:
        parts = [p.strip() for p in raw_city_state.split(",")]
        city = parts[0].title()
        state = parts[1].lower() if len(parts) > 1 else ""
        state_full = state_map.get(state, state.title())
        return f"{city},{state_full},IN"
    except Exception:
        return raw_city_state.title()

def safe_json_parse(raw_content: str, fallback):
    try:
        cleaned = raw_content.strip()
        # handle ```json blocks
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except Exception as e:
        # best-effort: try to find a JSON substring
        try:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(cleaned[start:end+1])
        except Exception:
            pass
        print(f"⚠️ JSON parse failed: {e}. Raw: {str(raw_content)[:200]}")
        return fallback

# --- Live weather ---
def get_live_weather(city_name: str):
    print(f"[Agent Research] Searching for live weather in {city_name}...")
    if not WEATHER_API_KEY:
        print("⚠️ WEATHER_API_KEY missing, using defaults.")
        return 28.0, 60.0
    try:
        base_url = "https://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}appid={WEATHER_API_KEY}&q={city_name}&units=metric"
        resp = requests.get(complete_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("cod") == 200:
            main = data.get("main", {})
            return float(main.get("temp", 28.0)), float(main.get("humidity", 60.0))
        return 28.0, 60.0
    except Exception as e:
        print("⚠️ Weather fetch error:", e)
        return 28.0, 60.0

# --- Gemini wrapper (caching) ---
def ask_gemini_ai(prompt_key: str, system_prompt: str, user_prompt: str):
    key_lower = prompt_key.lower().replace(" ", "_")[:240]
    if key_lower in ai_cache:
        return ai_cache[key_lower]

    if not GOOGLE_API_KEY:
        # Don't fail hard — return empty dict, functions using it have defaults
        return {}

    try:
        full_prompt = f"{system_prompt}\n\nUser Query: {user_prompt}\n\nIMPORTANT: Return ONLY valid JSON."
        model_ai = genai.GenerativeModel("gemini-pro")
        response = model_ai.generate_content(full_prompt)
        content = getattr(response, "text", "") or str(response)
        data = safe_json_parse(content, {})
        if isinstance(data, dict) and data:
            ai_cache[key_lower] = data
            try:
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(ai_cache, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
        return data
    except Exception as e:
        print("⚠️ Gemini call failed:", e)
        return {}

# --- Soil interpreter ---
def get_soil_and_location_details(farmer_prompt: str):
    system_prompt = "Extract city_or_state and soil_type. Return JSON with keys: city_or_state, soil_type."
    user_prompt = f"Farmer query: {farmer_prompt}"
    data = ask_gemini_ai(f"soil_location_{farmer_prompt}", system_prompt, user_prompt)
    if not isinstance(data, dict):
        data = {}
    if not data.get("city_or_state"):
        data["city_or_state"] = farmer_prompt
    if not data.get("soil_type"):
        data["soil_type"] = "unknown"
    return data

# --- Fill missing values ---
def fill_missing_values_ai(details: dict):
    location_soil = f"{details.get('city_or_state','')}, {details.get('soil_type','unknown')}"
    system_prompt = "Estimate N,P,K,pH,rainfall for Indian soils. Return JSON with numeric keys N,P,K,pH,rainfall."
    user_prompt = f"Estimate for: {location_soil}"
    ai_vals = ask_gemini_ai(f"npk_{location_soil}", system_prompt, user_prompt)
    defaults = {'N':50.0,'P':50.0,'K':50.0,'pH':6.5,'rainfall':400.0}
    for k, dv in defaults.items():
        try:
            details[k] = float(ai_vals.get(k, dv)) if ai_vals and ai_vals.get(k) is not None else dv
        except Exception:
            details[k] = dv
    return details

# --- Prediction using loaded model ---
def make_prediction(input_data: dict, top_n=5):
    if model is None or scaler is None or encoder is None:
        # graceful fallback: return a few hard-coded crops with rough scores
        fallback = [("rice", 40.0), ("wheat", 30.0), ("maize", 15.0), ("cotton", 8.0), ("mango", 7.0)]
        return fallback[:top_n]
    try:
        df = pd.DataFrame([input_data], columns=model.feature_names_in_)
        scaled = scaler.transform(df)
        probs = model.predict_proba(scaled)[0]
        smoothed = np.sqrt(np.maximum(probs, 0))
        total = smoothed.sum()
        normalized = smoothed / total if total > 0 else probs
        top_idx = np.argsort(normalized)[::-1][:top_n]
        results = []
        for idx in top_idx:
            try:
                crop = encoder.inverse_transform([idx])[0].lower()
            except Exception:
                crop = str(idx)
            results.append((crop, round(float(normalized[idx]*100), 2)))
        return results
    except Exception as e:
        print("⚠️ make_prediction failed:", e)
        return []

# --- Validation with Gemini (optional) ---
def validate_with_gemini(predictions, location):
    if not GOOGLE_API_KEY:
        return predictions
    crops_list = [p[0] for p in predictions]
    if not crops_list:
        return predictions
    system_prompt = "Strictly validate which crops are suitable for a given location. Return JSON {'valid_crops': ['rice','wheat', ...]}."
    user_prompt = f"Location: {location}\nCrops: {', '.join(crops_list)}"
    resp = ask_gemini_ai(f"validate_{location}", system_prompt, user_prompt)
    valid = set([c.lower() for c in resp.get("valid_crops", [])]) if isinstance(resp, dict) else set()
    if not valid:
        return predictions
    return [p for p in predictions if p[0] in valid]

# --- Live crop prices (Gemini fallback) ---
def get_live_crop_prices():
    system_prompt = "Return mandi prices JSON (Rs/kg) for rice,wheat,cotton,jute,coffee,mango,pigeonpeas."
    user_prompt = "Prices for rice,wheat,cotton,jute,coffee,mango,pigeonpeas."
    prices = ask_gemini_ai("crop_prices", system_prompt, user_prompt)
    defaults = {"rice":40.0,"wheat":35.0,"cotton":80.0,"jute":60.0,"coffee":150.0,"mango":120.0,"pigeonpeas":90.0}
    if not isinstance(prices, dict): prices = {}
    for k,v in defaults.items():
        prices.setdefault(k, v)
    return {k: float(prices.get(k, defaults[k])) for k in defaults.keys()}

# --- Future price predictor (Gemini) ---
def get_future_price_ai(crop_name: str, location: str):
    if not location:
        location = "India"
    system_prompt = "Predict the approximate future price (Rs/quintal) for a crop in 6 months. Return JSON {'future_price': number}."
    user_prompt = f"Predict 6-month future price for {crop_name} in {location}."
    resp = ask_gemini_ai(f"future_{crop_name}_{location}", system_prompt, user_prompt)
    try:
        val = float(resp.get("future_price", -1))
        return val/100 if val > 0 else -1.0
    except Exception:
        return -1.0

# --- Crop dynamic details ---
def get_crop_dynamic_details(crop_name: str):
    system_prompt = "Return crop details JSON with keys soil, irrigation, fertilizer, pesticides."
    user_prompt = f"Details for {crop_name}"
    resp = ask_gemini_ai(f"details_{crop_name}", system_prompt, user_prompt)
    defaults = {"soil":"loamy","irrigation":"flooding","fertilizer":"NPK 20-20-20","pesticides":"2kg/ha"}
    if not isinstance(resp, dict):
        resp = {}
    for k,v in defaults.items():
        resp.setdefault(k, v)
    return resp

# --- Crop rotation ---
def get_crop_rotation_plan(current_crop, location):
    system_prompt = "Suggest 2-season rotation. Return JSON with season_1_crop, season_1_reason, season_2_crop, season_2_reason or an error key."
    user_prompt = f"{current_crop} in {location}"
    resp = ask_gemini_ai(f"rotation_{current_crop}_{location}", system_prompt, user_prompt)
    if not isinstance(resp, dict) or "season_1_crop" not in resp:
        return {"error": "Could not generate a plan."}
    return resp

# --- Grow guide details ---
def get_grow_guide_details(crop_name: str):
    system_prompt = "Provide a detailed grow guide JSON with description, season, growth_duration, irrigation_plan, pesticide_usage."
    user_prompt = f"Provide grow guide for {crop_name} in India."
    resp = ask_gemini_ai(f"grow_{crop_name}", system_prompt, user_prompt)
    if not isinstance(resp, dict):
        resp = {}
    defaults = {
        "description": f"Practical guide for {crop_name}.",
        "season": "Kharif/Rabi (varies)",
        "growth_duration": "Varies",
        "irrigation_plan": "Irrigate as per crop needs",
        "pesticide_usage": "Use recommended pesticides"
    }
    for k,v in defaults.items():
        resp.setdefault(k, v)
    return resp

# --- Ranking & utilities ---
def traffic_light_color(suitability):
    try:
        s = float(suitability)
    except Exception:
        return "Red"
    if s >= 70: return "Green"
    elif s >= 40: return "Yellow"
    else: return "Red"

def rank_top_3(crop_probs, live_prices, future_prices):
    transport = {"rice":80,"wheat":85,"cotton":50,"jute":60,"coffee":40,"mango":30,"pigeonpeas":70}
    if not crop_probs:
        return ["N/A","N/A","N/A"]
    sorted_rev = sorted(crop_probs, key=lambda x: future_prices.get(x[0], live_prices.get(x[0], 0)), reverse=True)
    sorted_trans = sorted(crop_probs, key=lambda x: transport.get(x[0],0), reverse=True)
    balanced=[]
    for crop, prob in crop_probs:
        rev = future_prices.get(crop, live_prices.get(crop,0))
        sc = (prob * 0.5) + (rev * 0.3) + (transport.get(crop,0) * 0.2)
        balanced.append((crop, sc))
    sorted_bal = sorted(balanced, key=lambda x: x[1], reverse=True)
    return [
        sorted_rev[0][0] if sorted_rev else "N/A",
        sorted_trans[0][0] if sorted_trans else "N/A",
        sorted_bal[0][0] if sorted_bal else "N/A"
    ]

def save_results(input_data, predictions):
    try:
        row = {**input_data, "predictions": str(predictions), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        df = pd.DataFrame([row])
        df.to_csv("prediction_logs.csv", mode="a", header=not os.path.exists("prediction_logs.csv"), index=False)
    except Exception as e:
        print("⚠️ Logging failed:", e)

# --- FastAPI app & Static mount ---
app = FastAPI(title="KisanAI-V1 API")
# Serve the UI from ./static (ensure your index.html is at static/index.html)
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
else:
    print("⚠️ static/ folder not found — static files won't be served by FastAPI. Ensure you have static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Helpers for endpoints ---
def build_comparison_table(predictions, live_prices, future_prices):
    table = []
    for crop, prob in predictions:
        try:
            suit_percent = float(round(prob, 2))
        except Exception:
            suit_percent = 0.0
        color = traffic_light_color(suit_percent)
        live = float(live_prices.get(crop, 0))
        future_val = future_prices.get(crop, -1.0)
        future_display = float(future_val) if (isinstance(future_val, (int,float)) and future_val > 0) else "N/A"
        table.append({
            "crop": crop,
            "suitability_percent": suit_percent,
            "recommendation_color": color,
            "live_price_rs_kg": live,
            "predicted_future_price_rs_kg": future_display
        })
    return table

# --- Health ---
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": model is not None}

# --- Manual predict endpoint (used by UI /predict/manual) ---
@app.post("/predict/manual")
def predict_manual(payload: dict = Body(...)):
    try:
        city = payload.get("city_or_state") or payload.get("city") or "Manual Location"
        input_params = {
            "N": float(payload.get("N", 50)),
            "P": float(payload.get("P", 50)),
            "K": float(payload.get("K", 50)),
            "ph": float(payload.get("ph", 6.5)),
            "temperature": float(payload.get("temperature", 28)),
            "humidity": float(payload.get("humidity", 60)),
            "rainfall": float(payload.get("rainfall", 400))
        }

        top_crops = make_prediction(input_params, top_n=7)
        live_prices = get_live_crop_prices()
        future_prices = {c: get_future_price_ai(c, city) for c,_ in top_crops}
        # Optionally validate with Gemini:
        validated = validate_with_gemini(top_crops, city)
        comparison_table = build_comparison_table(validated if validated else top_crops, live_prices, future_prices)
        top3 = rank_top_3(top_crops, live_prices, future_prices)

        top3_obj = {
            "best_revenue": top3[0],
            "transport_friendly": top3[1],
            "balanced_choice": top3[2]
        }

        # rotation plan
        try:
            rotation_plan = get_crop_rotation_plan(top3_obj["balanced_choice"], city)
        except Exception:
            rotation_plan = {"error": "Rotation plan generation failed."}

        response = {
            "input_parameters": input_params,
            "location_analyzed": city,
            "top_3_recommendations": top3_obj,
            "comparison_table": comparison_table,
            "smart_rotation_plan": rotation_plan
        }

        save_results(input_params, top_crops)
        return response
    except Exception as e:
        print("❌ predict_manual error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# --- Agent endpoint (used by UI /predict/agent) ---
@app.post("/predict/agent")
def predict_agent(body: dict = Body(...)):
    try:
        prompt = body.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing prompt in request")
        base = get_soil_and_location_details(prompt)
        base = fill_missing_values_ai(base)
        city_for_weather = format_city_for_weather(base.get("city_or_state", "Manual Location"))
        temp, hum = get_live_weather(city_for_weather)
        input_params = {
            "N": float(base.get("N", 50)),
            "P": float(base.get("P", 50)),
            "K": float(base.get("K", 50)),
            "ph": float(base.get("pH", 6.5)),
            "temperature": float(temp),
            "humidity": float(hum),
            "rainfall": float(base.get("rainfall", 400))
        }

        top_crops = make_prediction(input_params, top_n=7)
        live_prices = get_live_crop_prices()
        future_prices = {c: get_future_price_ai(c, base.get("city_or_state", "India")) for c,_ in top_crops}
        validated = validate_with_gemini(top_crops, base.get("city_or_state", "India"))
        comparison_table = build_comparison_table(validated if validated else top_crops, live_prices, future_prices)
        top3 = rank_top_3(top_crops, live_prices, future_prices)
        top3_obj = {"best_revenue": top3[0], "transport_friendly": top3[1], "balanced_choice": top3[2]}

        try:
            rotation_plan = get_crop_rotation_plan(top3_obj["balanced_choice"], base.get("city_or_state", "India"))
        except Exception:
            rotation_plan = {"error": "Rotation plan generation failed."}

        response = {
            "input_parameters": input_params,
            "location_analyzed": base.get("city_or_state", prompt),
            "top_3_recommendations": top3_obj,
            "comparison_table": comparison_table,
            "smart_rotation_plan": rotation_plan
        }

        save_results(input_params, top_crops)
        return response

    except HTTPException:
        raise
    except Exception as e:
        print("❌ predict_agent error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# --- mini_chat endpoint (used by UI /mini_chat) ---
@app.post("/mini_chat")
def mini_chat(body: dict = Body(...)):
    try:
        query = body.get("query", "")
        if not query:
            return {"response": "Please send a query."}
        qlower = query.strip().lower()

        # try to detect crop name and return quick facts
        details = {}
        try:
            details = get_crop_dynamic_details(qlower)
        except Exception:
            details = {}

        if details and any(k in details for k in ["soil","irrigation","fertilizer"]):
            text = f"Quick facts: Soil: {details.get('soil')}, Irrigation: {details.get('irrigation')}, Fertilizer: {details.get('fertilizer')}."
            return {"response": text}

        # fallback small answer via Gemini (short)
        ai_short = ask_gemini_ai(f"mini_{qlower}", "Answer briefly about the query. Return JSON {'answer': '...'}", query)
        if isinstance(ai_short, dict) and ai_short.get("answer"):
            return {"response": ai_short.get("answer")}
        return {"response": "Sorry, I couldn't find an answer. Try 'tell me about wheat'."}

    except Exception as e:
        print("❌ mini_chat error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# --- Grow guide endpoints ---
@app.post("/grow_guide")
def grow_guide_post(body: dict = Body(...)):
    try:
        crop = body.get("crop_name") or body.get("crop") or ""
        if not crop:
            raise HTTPException(status_code=400, detail="Missing crop_name")
        guide = get_grow_guide_details(crop)
        # ensure returned keys exist
        for k in ["description","season","growth_duration","irrigation_plan","pesticide_usage"]:
            guide.setdefault(k, "Not available")
        return guide
    except HTTPException:
        raise
    except Exception as e:
        print("❌ grow_guide_post error:", e)
        raise HTTPException(st
