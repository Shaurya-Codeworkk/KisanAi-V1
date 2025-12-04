import joblib
import pandas as pd
import numpy as np
import requests
import json
import os
import google.generativeai as genai
from datetime import datetime   # Needed for logs

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

# Configure Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("❌ CRITICAL: GOOGLE_API_KEY environment variable is not set!")

print("Libraries import ho gayi hain...")

# --- Load ML Models ---
try:
    model = joblib.load('model_gbc.pkl') 
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    print("ML Models successfully load ho gaye hain! ✅")
except Exception as e:
    print(f"❌ CRITICAL Error loading models: {e}.")

# --- State Abbreviations Mapping ---
state_map = {
    "up": "Uttar Pradesh", "mh": "Maharashtra", "hr": "Haryana",
    "ka": "Karnataka", "tn": "Tamil Nadu",
}

# --- Helper ---
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
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except Exception as e:
        print(f"⚠️ JSON parse failed: {e}. Content was: {raw_content}")
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
    except Exception:
        return 28.0, 60.0

# --- Gemini Wrapper ---
def ask_gemini_ai(prompt_key: str, system_prompt: str, user_prompt: str):
    key_lower = prompt_key.lower().replace(" ", "_")
    if key_lower in ai_cache:
        return ai_cache[key_lower]

    if not GOOGLE_API_KEY:
        return {}
        
    try:
        full_prompt = f"{system_prompt}\n\nUser Query: {user_prompt}\n\nIMPORTANT: Return ONLY valid JSON."
        model_ai = genai.GenerativeModel('gemini-pro')
        response = model_ai.generate_content(full_prompt)
        content = response.text
        data = safe_json_parse(content, {})

        if data:
            ai_cache[key_lower] = data
            with open(CACHE_FILE, "w") as f:
                json.dump(ai_cache, f, indent=2)

        return data
        
    except Exception:
        return {}

# --- Soil Interpreter ---
def get_soil_and_location_details(farmer_prompt):
    system_prompt = "Extract city/state and soil type. Return JSON: {city_or_state, soil_type}."
    user_prompt = f"Farmer query: {farmer_prompt}"
    data = ask_gemini_ai(f"soil_location_{farmer_prompt}", system_prompt, user_prompt)
    if not data.get("city_or_state"): data["city_or_state"] = farmer_prompt
    if not data.get("soil_type"): data["soil_type"] = "unknown"
    return data

# --- Fill Missing Values ---
def fill_missing_values_ai(details):
    location_soil = f"{details.get('city_or_state','')}, {details.get('soil_type','unknown')}"
    system_prompt = "Estimate N,P,K,pH,rainfall for Indian soils. Return JSON."
    user_prompt = f"Estimate for: {location_soil}"
    ai_values = ask_gemini_ai(f"npk_{location_soil}", system_prompt, user_prompt)
    defaults = {'N':50,'P':50,'K':50,'pH':6.5,'rainfall':400}
    for k in defaults:
        details[k] = float(ai_values.get(k, defaults[k]))
    return details

# --- ML Prediction ---
def make_prediction(input_data, top_n=5):
    df = pd.DataFrame([input_data], columns=model.feature_names_in_)
    scaled = scaler.transform(df)
    original_probs = model.predict_proba(scaled)[0]
    smoothed = np.sqrt(original_probs)
    total = smoothed.sum()
    if total == 0:
        normalized = original_probs
    else:
        normalized = smoothed / total
    top_idx = np.argsort(normalized)[::-1][:top_n]
    results = []
    for idx in top_idx:
        crop = encoder.inverse_transform([idx])[0].lower()
        results.append((crop, round(normalized[idx]*100,2)))
    return results

# --- Gemini Validation ---
def validate_with_gemini(predictions, location):
    if not GOOGLE_API_KEY: return predictions
    crops_list = [p[0] for p in predictions]
    if not crops_list: return []
    system_prompt = "Strictly validate crop suitability. Return JSON {valid_crops}."
    user_prompt = f"Location: {location}\nCrops: {', '.join(crops_list)}"
    response = ask_gemini_ai(f"validate_{location}", system_prompt, user_prompt)
    valid = set([c.lower() for c in response.get("valid_crops", [])])
    if not valid: return predictions
    return [p for p in predictions if p[0] in valid]

# --- Live Prices ---
def get_live_crop_prices():
    system_prompt = "Return mandi prices JSON."
    user_prompt = "Prices for rice,wheat,cotton,jute,coffee,mango,pigeonpeas."
    prices = ask_gemini_ai("crop_prices", system_prompt, user_prompt)
    defaults = {"rice":40,"wheat":35,"cotton":80,"jute":60,"coffee":150,"mango":120,"pigeonpeas":90}
    for k,v in defaults.items():
        if k not in prices: prices[k]=v
    return {k:float(v) for k,v in prices.items()}

# --- Future Price AI ---
def get_future_price_ai(crop_name, location):
    if location.lower() in ["manual location","manual data","unknown"]:
        location="India"
    system_prompt="Predict future crop price. Return JSON {'future_price':number}."
    user_prompt=f"Predict 6-month price for {crop_name} in {location}."
    prediction = ask_gemini_ai(f"future_{crop_name}_{location}", system_prompt, user_prompt)
    val = float(prediction.get("future_price",-1))
    return val/100 if val>0 else -1

# --- Crop Details ---
def get_crop_dynamic_details(crop_name):
    system_prompt="Return crop details JSON."
    user_prompt=f"Details for {crop_name}"
    data=ask_gemini_ai(f"details_{crop_name}",system_prompt,user_prompt)
    defaults={"soil":"loamy","irrigation":"flooding","fertilizer":"NPK 20-20-20","pesticides":"2kg/ha"}
    for k in defaults:
        if k not in data: data[k]=defaults[k]
    return data

# --- Crop Rotation ---
def get_crop_rotation_plan(current_crop,location):
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
    sorted_rev=sorted(crop_probs,key=lambda x:future_prices.get(x[0],0),reverse=True)
    sorted_trans=sorted(crop_probs,key=lambda x:transport.get(x[0],0),reverse=True)
    balanced=[]
    for crop,prob in crop_probs:
        sc=(prob*0.5)+(future_prices.get(crop,0)*0.3)+(transport.get(crop,0)*0.2)
        balanced.append((crop,sc))
    sorted_bal=sorted(balanced,key=lambda x:x[1],reverse=True)
    return [sorted_rev[0][0],sorted_trans[0][0],sorted_bal[0][0]]

# --- SAVE LOG ---
def save_results(input_data,predictions):
    try:
        row={**input_data,"predictions":str(predictions),"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        df=pd.DataFrame([row])
        df.to_csv("prediction_logs.csv",mode="a",header=not os.path.exists("prediction_logs.csv"),index=False)
    except Exception:
        pass


# ----------------------------------------------------
# 🚀🚀🚀 FASTAPI WRAPPER (REQUIRED FOR RENDER) 🚀🚀🚀
# ----------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="KisanAI-V1 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

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
    return {"status":"ok","models_loaded":True}

@app.post("/predict")
def api_predict(req: PredictRequest):
    try:
        if req.farmer_prompt:
            base = get_soil_and_location_details(req.farmer_prompt)
            base = fill_missing_values_ai(base)
            city = format_city_for_weather(base['city_or_state'])
            temp,hum = get_live_weather(city)
            data = {
                'N':base['N'],'P':base['P'],'K':base['K'],
                'temperature':temp,'humidity':hum,
                'ph':base['pH'],'rainfall':base['rainfall']
            }
        else:
            data = {
                'N': req.N or 50,
                'P': req.P or 50,
                'K': req.K or 50,
                'ph': req.ph or 6.5,
                'rainfall': req.rainfall or 400,
                'temperature': req.temperature or 28,
                'humidity': req.humidity or 60
            }

        top = make_prediction(data,5)
        live = get_live_crop_prices()
        future = {c:get_future_price_ai(c,"India") for c,_ in top}
        validated = validate_with_gemini(top,"India")
        ranked = rank_top_3(top,live,future)

        return {
            "final_input": data,
            "top_crops": top,
            "validated": validated,
            "live_prices": live,
            "future_prices": future,
            "ranked_top3": ranked
        }
    except Exception as e:
        raise HTTPException(500,str(e))

@app.post("/grow_guide")
def api_grow(req: GrowGuideRequest):
    try:
        return {"crop":req.crop_name, "guide":get_grow_guide_details(req.crop_name)}
    except Exception as e:
        raise HTTPException(500,str(e))

