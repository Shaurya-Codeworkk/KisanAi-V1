# api.py
import os
import json
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Keep mini copilot import (you already had this)
from mini_kisan_copilot import mini_copilot_response

router = APIRouter(prefix="")

# --- Pydantic Models ---
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


# --- Helper: Safe check if ML models are available (deferred import to avoid circular import) ---
def _models_loaded() -> bool:
    try:
        # Deferred import from app.py where models are loaded
        from app import model, scaler, encoder  # noqa: F401
        return all([model is not None, scaler is not None, encoder is not None])
    except Exception:
        return False


# --- Helper: deferred imports of functions from app.py (prevents circular import at module-load time) ---
def _get_core_funcs():
    """
    Returns a dict of functions imported from app.py.
    Import is done when endpoint is executed.
    """
    try:
        from app import (
            get_soil_and_location_details,
            fill_missing_values_ai,
            format_city_for_weather,
            get_live_weather,
            make_prediction,
            validate_with_gemini,
            get_live_crop_prices,
            get_future_price_ai,
            rank_top_3,
            get_crop_rotation_plan,
            get_grow_guide_details,
            save_results,
            traffic_light_color,
        )
        return {
            "get_soil_and_location_details": get_soil_and_location_details,
            "fill_missing_values_ai": fill_missing_values_ai,
            "format_city_for_weather": format_city_for_weather,
            "get_live_weather": get_live_weather,
            "make_prediction": make_prediction,
            "validate_with_gemini": validate_with_gemini,
            "get_live_crop_prices": get_live_crop_prices,
            "get_future_price_ai": get_future_price_ai,
            "rank_top_3": rank_top_3,
            "get_crop_rotation_plan": get_crop_rotation_plan,
            "get_grow_guide_details": get_grow_guide_details,
            "save_results": save_results,
            "traffic_light_color": traffic_light_color,
        }
    except Exception as e:
        # If import fails, return empty dict — endpoints will handle failure
        print(f"Deferred core import failed: {e}")
        return {}


# --- Endpoints ---

@router.get("/status", tags=["Health"])
def get_status():
    """
    API health check - reports if ML models are loaded.
    """
    return {
        "status": "ok",
        "service": "Kisan Sathi API",
        "models_loaded": _models_loaded(),
        "message": "Service is running."
    }


@router.post("/predict/agent", tags=["KisanAI Pro"])
def predict_with_agent(data: AgentInput):
    """
    Agent-driven prediction endpoint (uses AI to parse farmer prompt and then full workflow).
    """
    core = _get_core_funcs()
    if not core:
        raise HTTPException(status_code=500, detail="Core functions unavailable. Check app.py importability.")

    if not _models_loaded():
        raise HTTPException(status_code=503, detail="ML models are not loaded. Cannot make predictions.")

    try:
        base_details = core["get_soil_and_location_details"](data.prompt)
        base_details = core["fill_missing_values_ai"](base_details)
        city_for_weather = core["format_city_for_weather"](base_details["city_or_state"])
        live_temp, live_humidity = core["get_live_weather"](city_for_weather)

        final_data = {
            "N": base_details["N"],
            "P": base_details["P"],
            "K": base_details["K"],
            "temperature": live_temp,
            "humidity": live_humidity,
            "ph": base_details["pH"],
            "rainfall": base_details["rainfall"],
        }

        # Run full workflow (with AI validation)
        top_crops = core["make_prediction"](final_data, top_n=5)
        # validate
        try:
            top_crops_validated = core["validate_with_gemini"](top_crops, base_details["city_or_state"])
            if top_crops_validated:
                top_crops = top_crops_validated
        except Exception as e:
            print(f"Validation step failed (continuing with ML results): {e}")

        # prices
        live_prices = {}
        try:
            live_prices = core["get_live_crop_prices"]()
        except Exception as e:
            print(f"Live price fetch failed: {e}")

        future_prices = {crop: core["get_future_price_ai"](crop, base_details["city_or_state"]) for crop, _ in top_crops}
        ranked_top3 = core["rank_top_3"](top_crops, live_prices, future_prices)

        rotation_plan = {}
        if ranked_top3 and ranked_top3[2] != "N/A":
            rotation_plan = core["get_crop_rotation_plan"](ranked_top3[2], base_details["city_or_state"])

        comparison_table = [
            {
                "crop": c.title(),
                "suitability_percent": p,
                "live_price_rs_kg": f'₹{live_prices.get(c, 0):.2f}',
                "predicted_future_price_rs_kg": f'₹{future_prices.get(c, -1.0):.2f}' if future_prices.get(c, -1.0) > 0 else "N/A",
                "recommendation_color": core["traffic_light_color"](p),
            }
            for c, p in top_crops
        ]

        # Save results asynchronously-ish (function handles exceptions)
        try:
            core["save_results"](final_data, top_crops)
        except Exception as e:
            print(f"Save results warning: {e}")

        return {
            "input_parameters": final_data,
            "location_analyzed": base_details["city_or_state"],
            "comparison_table": comparison_table,
            "top_3_recommendations": {
                "best_revenue": ranked_top3[0] if ranked_top3 else "N/A",
                "transport_friendly": ranked_top3[1] if ranked_top3 else "N/A",
                "balanced_choice": ranked_top3[2] if ranked_top3 else "N/A",
            },
            "smart_rotation_plan": rotation_plan,
        }

    except Exception as e:
        print(f"Error in /predict/agent: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred in the AI Agent.")


@router.post("/predict/manual", tags=["KisanAI Pro"])
def predict_with_manual_input(data: ManualInput):
    """
    Manual numeric input -> ML-only predictions (no AI validation).
    """
    core = _get_core_funcs()
    if not core:
        raise HTTPException(status_code=500, detail="Core functions unavailable. Check app.py importability.")

    if not _models_loaded():
        raise HTTPException(status_code=503, detail="ML models are not loaded. Cannot make predictions.")

    try:
        final_data = data.dict()
        location = final_data.pop("city_or_state", "Unknown")

        top_crops = core["make_prediction"](final_data, top_n=5)

        live_prices = {}
        try:
            live_prices = core["get_live_crop_prices"]()
        except Exception as e:
            print(f"Live price fetch failed: {e}")

        future_prices = {crop: core["get_future_price_ai"](crop, location) for crop, _ in top_crops}
        ranked_top3 = core["rank_top_3"](top_crops, live_prices, future_prices)

        rotation_plan = {}
        if ranked_top3 and ranked_top3[2] != "N/A":
            rotation_plan = core["get_crop_rotation_plan"](ranked_top3[2], location)

        comparison_table = [
            {
                "crop": c.title(),
                "suitability_percent": p,
                "live_price_rs_kg": f'₹{live_prices.get(c, 0):.2f}',
                "predicted_future_price_rs_kg": f'₹{future_prices.get(c, -1.0):.2f}' if future_prices.get(c, -1.0) > 0 else "N/A",
                "recommendation_color": core["traffic_light_color"](p),
            }
            for c, p in top_crops
        ]

        try:
            core["save_results"](final_data, top_crops)
        except Exception as e:
            print(f"Save results warning: {e}")

        return {
            "input_parameters": final_data,
            "location_analyzed": location,
            "comparison_table": comparison_table,
            "top_3_recommendations": {
                "best_revenue": ranked_top3[0] if ranked_top3 else "N/A",
                "transport_friendly": ranked_top3[1] if ranked_top3 else "N/A",
                "balanced_choice": ranked_top3[2] if ranked_top3 else "N/A",
            },
            "smart_rotation_plan": rotation_plan,
        }

    except Exception as e:
        print(f"Error in /predict/manual: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during manual prediction.")


@router.post("/mini_chat", tags=["KisanAI Mini"])
def mini_chat_endpoint(data: MiniChatInput):
    """
    Mini chat endpoint using mini_kisan_copilot.py
    """
    try:
        response_text = mini_copilot_response(data.query)
        return {"response": response_text}
    except Exception as e:
        print("!!!!!!!!!! ERROR IN /mini_chat ENDPOINT !!!!!!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise HTTPException(status_code=500, detail="An error occurred in the Mini Chat assistant.")


@router.get("/grow_guide/{crop_name}", tags=["Grow Guide"])
def get_grow_guide_endpoint(crop_name: str):
    """
    Grow guide endpoint — defers to app.py's function.
    """
    core = _get_core_funcs()
    if not core:
        raise HTTPException(status_code=500, detail="Core functions unavailable. Check app.py importability.")

    try:
        guide_data = core["get_grow_guide_details"](crop_name)
        if not guide_data:
            raise HTTPException(status_code=404, detail="Could not generate a guide for this crop.")
        return guide_data
    except Exception as e:
        print(f"An error occurred in /grow_guide: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the grow guide.")


# Optional: a lightweight route to serve static index (if you want to keep it here).
# NOTE: Recommended: let app.py mount StaticFiles. If you still want this, uncomment:
#
# @router.get("/", response_class=FileResponse)
# def serve_index():
#     if os.path.exists("static/index.html"):
#         return FileResponse("static/index.html")
#     raise HTTPException(status_code=404, detail="Index not found.")
#
