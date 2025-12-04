# api.py  — safe, lazy-importing FastAPI wrapper (paste entire file, overwrite existing api.py)

import importlib
import traceback
import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Immediately expose an ASGI app so Uvicorn won't fail before imports.
app = FastAPI(
    title="Kisan Sathi API (Safe Loader)",
    description="Frontend + backend wrapper that lazily loads heavy modules.",
    version="2.0.0"
)

# Templates (make sure your index.html is in templates/index.html)
templates = Jinja2Templates(directory="templates")

# CORS — keep permissive for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# --- Pydantic request models ---
class AgentInput(BaseModel):
    prompt: str = Field(..., example="Suggest crops for black soil in Pune, MH")

class ManualInput(BaseModel):
    city_or_state: str = Field(..., example="Pune, MH")
    N: float = Field(..., example=90.0)
    P: float = Field(..., example=42.0)
    K: float = Field(..., example=43.0)
    temperature: float = Field(..., example=24.5)
    humidity: float = Field(..., example=82.1)
    ph: float = Field(..., example=6.5)
    rainfall: float = Field(..., example=202.9)

class MiniChatInput(BaseModel):
    query: str = Field(..., example="tell me about wheat")

# --- Lazy import helpers ---
_cached_core = None
_cached_mini = None
_cached_core_err = None
_cached_mini_err = None

def import_core_module():
    """
    Lazily import app.py as 'core'. Return tuple (module or None, error_trace or None)
    """
    global _cached_core, _cached_core_err
    if _cached_core is not None or _cached_core_err is not None:
        return _cached_core, _cached_core_err
    try:
        _cached_core = importlib.import_module("app")  # your app.py
        _cached_core_err = None
    except Exception as e:
        _cached_core = None
        _cached_core_err = traceback.format_exc()
        print("api.import_core_module failed:", _cached_core_err)
    return _cached_core, _cached_core_err

def import_mini_module():
    """
    Lazily import mini_kisan_copilot.
    """
    global _cached_mini, _cached_mini_err
    if _cached_mini is not None or _cached_mini_err is not None:
        return _cached_mini, _cached_mini_err
    try:
        _cached_mini = importlib.import_module("mini_kisan_copilot")
        _cached_mini_err = None
    except Exception:
        _cached_mini = None
        _cached_mini_err = traceback.format_exc()
        print("api.import_mini_module failed:", _cached_mini_err)
    return _cached_mini, _cached_mini_err

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_homepage(request: Request):
    """
    Serves templates/index.html. Ensure templates/index.html exists.
    """
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        # Template missing — return helpful message
        return HTMLResponse(
            content=f"<h2>Index not found</h2><pre>{str(e)}</pre>",
            status_code=500
        )

@app.get("/status", tags=["Health"])
def get_status():
    """
    Returns whether core and mini modules loaded plus any import error traces so you can debug.
    """
    core, core_err = import_core_module()
    mini, mini_err = import_mini_module()

    models_loaded = False
    try:
        if core and all(getattr(core, name, None) for name in ("model", "scaler", "encoder")):
            models_loaded = True
    except Exception:
        models_loaded = False

    return {
        "status": "ok",
        "service": "Kisan Sathi API",
        "core_loaded": core is not None,
        "mini_loaded": mini is not None,
        "models_loaded": models_loaded,
        "core_import_error": None if core_err is None else core_err.splitlines()[-10:],  # last 10 lines
        "mini_import_error": None if mini_err is None else mini_err.splitlines()[-10:]
    }

@app.post("/predict/agent", tags=["KisanAI Pro"])
def predict_with_agent(data: AgentInput):
    """
    Agent endpoint. Will lazily load core and run agent flow (if core provides process_recommendations).
    """
    core, core_err = import_core_module()
    if core is None:
        raise HTTPException(status_code=500, detail=f"Core module failed to import. See /status for error. Trace excerpt:\n{core_err}")

    # Prefer a single helper if present
    if hasattr(core, "process_recommendations"):
        try:
            # run agent flow (core handles AI fill etc)
            final = core.process_recommendations  # function ref
            # call as in your api: process_recommendations expects final_data, location, use_ai_validation True
            # core.get_soil_and_location_details + fill + get_live_weather used inside core.process_recommendations in original design
            base_details = core.get_soil_and_location_details(data.prompt) if hasattr(core, "get_soil_and_location_details") else {"city_or_state": data.prompt, "soil_type": "unknown"}
            if hasattr(core, "fill_missing_values_ai"):
                base_details = core.fill_missing_values_ai(base_details)
            city_for_weather = core.format_city_for_weather(base_details["city_or_state"]) if hasattr(core, "format_city_for_weather") else base_details.get("city_or_state", "India")
            live_temp, live_hum = core.get_live_weather(city_for_weather) if hasattr(core, "get_live_weather") else (28.0, 60.0)
            final_data = {
                "N": base_details.get("N", 50),
                "P": base_details.get("P", 50),
                "K": base_details.get("K", 50),
                "temperature": live_temp,
                "humidity": live_hum,
                "ph": base_details.get("pH", 6.5),
                "rainfall": base_details.get("rainfall", 400)
            }
            # Call the core helper; if signature different, it should raise so we catch below
            result = core.process_recommendations(final_data, base_details.get("city_or_state", "India"), use_ai_validation=True)
            return JSONResponse(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent prediction failed: {traceback.format_exc()}")
    else:
        # Fallback: use basic make_prediction if available
        if not hasattr(core, "make_prediction"):
            raise HTTPException(status_code=500, detail="Core does not provide process_recommendations or make_prediction. See /status.")
        try:
            base = core.get_soil_and_location_details(data.prompt) if hasattr(core, "get_soil_and_location_details") else {"city_or_state": data.prompt, "soil_type": "unknown"}
            base = core.fill_missing_values_ai(base) if hasattr(core, "fill_missing_values_ai") else base
            city = core.format_city_for_weather(base["city_or_state"]) if hasattr(core, "format_city_for_weather") else base["city_or_state"]
            temp, hum = core.get_live_weather(city) if hasattr(core, "get_live_weather") else (28.0, 60.0)
            final_input = {"N": base.get("N", 50), "P": base.get("P", 50), "K": base.get("K", 50), "temperature": temp, "humidity": hum, "ph": base.get("pH", 6.5), "rainfall": base.get("rainfall", 400)}
            top = core.make_prediction(final_input, top_n=5)
            return JSONResponse({
                "input_parameters": final_input,
                "top_crops": top
            })
        except Exception:
            raise HTTPException(status_code=500, detail=f"Agent fallback failed: {traceback.format_exc()}")

@app.post("/predict/manual", tags=["KisanAI Pro"])
def predict_with_manual_input(data: ManualInput):
    core, core_err = import_core_module()
    if core is None:
        raise HTTPException(status_code=500, detail=f"Core module failed to import. See /status for details.")
    try:
        final_data = data.dict()
        location = final_data.pop("city_or_state", "Unknown")
        # If process_recommendations exists, use it (same format as previous api.py)
        if hasattr(core, "process_recommendations"):
            result = core.process_recommendations(final_data, location, use_ai_validation=False)
            return JSONResponse(result)
        # else simple fallback
        top = core.make_prediction(final_data, top_n=5) if hasattr(core, "make_prediction") else []
        live_prices = core.get_live_crop_prices() if hasattr(core, "get_live_crop_prices") else {}
        future_prices = {c: core.get_future_price_ai(c, location) for c,_ in top} if hasattr(core, "get_future_price_ai") else {}
        ranked_top3 = core.rank_top_3(top, live_prices, future_prices) if hasattr(core, "rank_top_3") else []
        return JSONResponse({
            "input_parameters": final_data,
            "top_crops": top,
            "live_prices": live_prices,
            "future_prices": future_prices,
            "ranked_top3": ranked_top3
        })
    except Exception:
        raise HTTPException(status_code=500, detail=f"Manual prediction failed: {traceback.format_exc()}")

@app.post("/mini_chat", tags=["KisanAI Mini"])
def mini_chat_endpoint(data: MiniChatInput):
    mini, mini_err = import_mini_module()
    if mini is None:
        # helpful error, not silent — user needs to add dependency or env var
        raise HTTPException(status_code=500, detail=f"mini_kisan_copilot failed to import. See /status for trace excerpt.\n{mini_err}")
    if not hasattr(mini, "mini_copilot_response"):
        raise HTTPException(status_code=500, detail="mini_kisan_copilot module does not expose mini_copilot_response()")
    try:
        text = mini.mini_copilot_response(data.query)
        return {"response": text}
    except Exception:
        raise HTTPException(status_code=500, detail=f"Mini chat failed: {traceback.format_exc()}")

@app.get("/grow_guide/{crop_name}", tags=["Grow Guide"])
def get_grow_guide_endpoint(crop_name: str):
    core, core_err = import_core_module()
    if core is None:
        raise HTTPException(status_code=500, detail=f"Core import failed. See /status.")
    if not hasattr(core, "get_grow_guide_details"):
        raise HTTPException(status_code=500, detail="Core does not implement get_grow_guide_details()")
    try:
        guide = core.get_grow_guide_details(crop_name)
        if not guide:
            raise HTTPException(status_code=404, detail="Could not generate guide for this crop.")
        return guide
    except Exception:
        raise HTTPException(status_code=500, detail=f"Grow guide failed: {traceback.format_exc()}")
