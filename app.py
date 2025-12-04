import joblib
import pandas as pd
import numpy as np
import requests
import json
import os
import google.generativeai as genai
from datetime import datetime

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
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
except:
    ai_cache = {}

# Configure Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("❌ GOOGLE_API_KEY missing")

print("Libraries import ho gayi hain...")

# ----------------------------------------------------
# MODEL LOAD
# ----------------------------------------------------
try:
    model = joblib.load('model_gbc.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    print("ML Models loaded successfully ✔")
except Exception as e:
    print("❌ Model Load Error:", e)

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------

state_map = {
    "up": "Uttar Pradesh", "mh": "Maharashtra", "hr": "Haryana",
    "ka": "Karnataka", "tn": "Tamil Nadu",
}

def format_city_for_weather(raw):
    try:
        parts = [p.strip() for p in raw.split(",")]
        city = parts[0].title()
        state = parts[1].lower() if len(parts) > 1 else ""
        state_full = state_map.get(state, state.title())
        return f"{city},{state_full},IN"
    except:
        return raw.title()

def safe_json_parse(raw, fallback):
    try:
        c = raw.strip()
        if "```json" in c:
            c = c.split("```json")[1].split("```")[0]
        elif "```" in c:
            c = c.split("```")[1].split("```")[0]
        return json.loads(c.strip())
    except:
        return fallback

def get_live_weather(city):
    if not WEATHER_API_KEY:
        return 28, 60
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?appid={WEATHER_API_KEY}&q={city}&units=metric"
        r = requests.get(url, timeout=10).json()
        if r.get("cod") == 200:
            main = r["main"]
            return main["temp"], main["humidity"]
        return 28, 60
    except:
        return 28, 60

def ask_gemini_ai(key, sys, usr):
    key = key.lower().replace(" ", "_")
    if key in ai_cache:
        return ai_cache[key]

    if not GOOGLE_API_KEY:
        return {}

    try:
        prompt = f"{sys}\n\nUser Query: {usr}\n\nReturn ONLY JSON."
        m = genai.GenerativeModel("gemini-pro")
        resp = m.generate_content(prompt)
        data = safe_json_parse(resp.text, {})
        if data:
            ai_cache[key] = data
            with open(CACHE_FILE, "w") as f:
                json.dump(ai_cache, f, indent=2)
        return data
    except:
        return {}

# ----------------------------------------------------
# CORE LOGIC
# ----------------------------------------------------
def get_soil_and_location_details(q):
    sys = "Extract city/state and soil_type. Return JSON."
    usr = f"Query: {q}"
    d = ask_gemini_ai(f"soil_{q}", sys, usr)
    d.setdefault("city_or_state", q)
    d.setdefault("soil_type", "unknown")
    return d

def fill_missing_values_ai(d):
    loc = f"{d['city_or_state']}, {d['soil_type']}"
    sys = "Estimate N,P,K,pH,rainfall. Return JSON."
    usr = f"Estimate for: {loc}"
    ai = ask_gemini_ai(f"npk_{loc}", sys, usr)
    defaults = {"N":50,"P":50,"K":50,"pH":6.5,"rainfall":400}
    for k in defaults:
        d[k] = float(ai.get(k, defaults[k]))
    return d

def make_prediction(data, top_n=5):
    df = pd.DataFrame([data], columns=model.feature_names_in_)
    scaled = scaler.transform(df)
    probs = model.predict_proba(scaled)[0]
    sm = np.sqrt(probs)
    sm = sm / sm.sum()
    idx = np.argsort(sm)[::-1][:top_n]
    return [(encoder.inverse_transform([i])[0].lower(), round(sm[i]*100,2)) for i in idx]

def validate_with_gemini(pred, loc):
    crops = [p[0] for p in pred]
    if not crops: return pred
    sys = "Strictly validate crop suitability. Return JSON {valid_crops}."
    usr = f"Location: {loc} | Crops: {', '.join(crops)}"
    r = ask_gemini_ai(f"validate_{loc}", sys, usr)
    valid = set([c.lower() for c in r.get("valid_crops", [])])
    return [p for p in pred if p[0] in valid] if valid else pred

def get_live_crop_prices():
    sys = "Return mandi prices JSON."
    usr = "Prices for rice,wheat,cotton,jute,coffee,mango,pigeonpeas."
    raw = ask_gemini_ai("prices", sys, usr)
    defaults = {"rice":40,"wheat":35,"cotton":80,"jute":60,"coffee":150,"mango":120,"pigeonpeas":90}
    for k,v in defaults.items():
        raw.setdefault(k, v)
    return {k: float(v) for k,v in raw.items()}

def get_future_price_ai(crop, loc):
    if loc.lower() in ["manual location","manual data","unknown"]:
        loc="India"
    sys = "Predict future price (per quintal). Return JSON."
    usr = f"Price for {crop} in {loc}"
    r = ask_gemini_ai(f"future_{crop}_{loc}", sys, usr)
    v = float(r.get("future_price",-1))
    return v/100 if v>0 else -1

def get_crop_dynamic_details(crop):
    sys="Return crop details JSON."
    usr=f"Details for {crop}"
    d = ask_gemini_ai(f"details_{crop}", sys, usr)
    defaults={"soil":"loamy","irrigation":"flooding","fertilizer":"NPK 20-20-20","pesticides":"2kg/ha"}
    for k,v in defaults.items():
        d.setdefault(k,v)
    return d

def get_crop_rotation_plan(crop,loc):
    sys="Suggest 2-season rotation JSON."
    usr=f"{crop} in {loc}"
    return ask_gemini_ai(f"rot_{crop}_{loc}",sys,usr)

def get_grow_guide_details(crop):
    sys="Give grow guide JSON."
    usr=f"Guide for {crop}"
    return ask_gemini_ai(f"grow_{crop}",sys,usr)

def rank_top_3(crops,live,future):
    trans={"rice":80,"wheat":85,"cotton":50,"jute":60,"coffee":40,"mango":30,"pigeonpeas":70}
    rev=sorted(crops,key=lambda x:future.get(x[0],0),reverse=True)
    tr =sorted(crops,key=lambda x:trans.get(x[0],0),reverse=True)
    bal=[]
    for c,p in crops:
        sc=(p*0.5)+(future.get(c,0)*0.3)+(trans.get(c,0)*0.2)
        bal.append((c,sc))
    bal=sorted(bal,key=lambda x:x[1],reverse=True)
    return [rev[0][0], tr[0][0], bal[0][0]]

# ----------------------------------------------------
# FASTAPI + STATIC HTML
# ----------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

# Serve UI
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

# CORS
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
def predict_api(req: PredictRequest):
    try:
        if req.farmer_prompt:
            base = get_soil_and_location_details(req.farmer_prompt)
            base = fill_missing_values_ai(base)
            city = format_city_for_weather(base["city_or_state"])
            t,h = get_live_weather(city)
            data = {
                "N":base["N"], "P":base["P"], "K":base["K"],
                "temperature":t, "humidity":h,
                "ph":base["pH"], "rainfall":base["rainfall"]
            }
        else:
            data = {
                "N":req.N or 50,
                "P":req.P or 50,
                "K":req.K or 50,
                "ph":req.ph or 6.5,
                "rainfall":req.rainfall or 400,
                "temperature":req.temperature or 28,
                "humidity":req.humidity or 60
            }

        top = make_prediction(data,5)
        live = get_live_crop_prices()
        future = {c:get_future_price_ai(c,"India") for c,_ in top}
        validated = validate_with_gemini(top,"India")
        ranked = rank_top_3(top,live,future)

        return {
            "final_input":data,
            "top_crops":top,
            "validated":validated,
            "live_prices":live,
            "future_prices":future,
            "ranked_top3":ranked
        }

    except Exception as e:
        raise HTTPException(500,str(e))

@app.post("/grow_guide")
def grow_guide_api(req: GrowGuideRequest):
    try:
        return {"crop":req.crop_name, "guide":get_grow_guide_details(req.crop_name)}
    except Exception as e:
        raise HTTPException(500,str(e))
