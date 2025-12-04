import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

# --- IMPORTANT: Import all logic from your other files ---
# This makes all the functions from your upgraded files available here.
from app import *
from mini_kisan_copilot import mini_copilot_response

# --- FastAPI App Initialization & Template Config ---
app = FastAPI(
    title="Kisan Sathi API",
    description="An intelligent API for crop recommendations using ML and AI agents.",
    version="2.0.0" # Upgraded version
)
# This is needed to serve your index.html file
templates = Jinja2Templates(directory="templates")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API validation ---
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


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
def serve_homepage(request: Request):
    """
    Serves the main index.html file as the homepage.
    This allows your single Web Service to act as both frontend and backend.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status", tags=["Health"])
def get_status():
    """
    A new feature to check the health of the API and see if ML models are loaded.
    """
    models_loaded = all([model, scaler, encoder])
    return {
        "status": "ok",
        "service": "Kisan Sathi API",
        "models_loaded": models_loaded,
        "message": "Service is running." if models_loaded else "Service is running, but ML models failed to load."
    }

@app.post("/predict/agent", tags=["KisanAI Pro"])
def predict_with_agent(data: AgentInput):
    """
    Endpoint for the KisanAI Pro agent. Takes a natural language prompt.
    """
    if not all([model, scaler, encoder]):
        raise HTTPException(status_code=503, detail="ML models are not loaded. Cannot make predictions.")
    try:
        # Step 1: Use AI to understand the user's prompt (from app.py)
        base_details = get_soil_and_location_details(data.prompt)
        
        # Step 2: Use AI to fill in missing soil/weather data (from app.py)
        base_details = fill_missing_values_ai(base_details)
        
        # Step 3: Get live weather for the location (from app.py)
        city_for_weather = format_city_for_weather(base_details['city_or_state'])
        live_temp, live_humidity = get_live_weather(city_for_weather)
        
        # Step 4: Prepare data for the ML model
        final_data = {
            'N': base_details['N'], 'P': base_details['P'], 'K': base_details['K'],
            'temperature': live_temp, 'humidity': live_humidity,
            'ph': base_details['pH'], 'rainfall': base_details['rainfall']
        }
        
        # Step 5: Run the full recommendation workflow (prediction, prices, rotation)
        # Enable AI validation for the Agent flow
        recommendations = process_recommendations(final_data, base_details['city_or_state'], use_ai_validation=True)
        return recommendations
        
    except Exception as e:
        print(f"Error in /predict/agent: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred in the AI Agent.")

@app.post("/predict/manual", tags=["KisanAI Pro"])
def predict_with_manual_input(data: ManualInput):
    """
    Endpoint for manual data input.
    """
    if not all([model, scaler, encoder]):
        raise HTTPException(status_code=503, detail="ML models are not loaded. Cannot make predictions.")
    try:
        final_data = data.dict()
        location = final_data.pop('city_or_state', 'Unknown')
        
        # Run the full recommendation workflow with the manual data
        # Disable AI validation for Manual flow (Pure ML)
        recommendations = process_recommendations(final_data, location, use_ai_validation=False)
        return recommendations
        
    except Exception as e:
        print(f"Error in /predict/manual: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during manual prediction.")

@app.post("/mini_chat", tags=["KisanAI Mini"])
def mini_chat_endpoint(data: MiniChatInput):
    """
    Upgraded endpoint for the KisanAI Mini chatbot.
    It now uses the powerful logic from your mini_kisan_copilot.py file.
    """
    try:
        # This now calls the advanced function from your other file
        response_text = mini_copilot_response(data.query)
        return {"response": response_text}
    except Exception as e:
        print(f"!!!!!!!!!! ERROR IN /mini_chat ENDPOINT !!!!!!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise HTTPException(status_code=500, detail="An error occurred in the Mini Chat assistant.")

@app.get("/grow_guide/{crop_name}", tags=["Grow Guide"])
def get_grow_guide_endpoint(crop_name: str):
    """
    Provides a detailed growing guide for a specific crop using an AI.
    """
    try:
        # This line now calls the correct new function from app.py
        guide_data = get_grow_guide_details(crop_name)

        if not guide_data:
            raise HTTPException(status_code=404, detail="Could not generate a guide for this crop.")
        return guide_data
    except Exception as e:
        print(f"An error occurred in /grow_guide: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the grow guide.")

# --- New Helper Function for Cleaner Code ---
def process_recommendations(final_data: Dict, location: str, use_ai_validation: bool = False) -> Dict:
    """
    A helper function to run the full prediction and analysis workflow.
    Args:
        use_ai_validation: If True, uses Gemini to double-check predictions (for Agent flow).
                           If False, uses raw ML predictions (for Manual flow).
    """
    top_crops = make_prediction(final_data, top_n=5)
    if not top_crops:
        raise ValueError("ML model failed to return predictions.")
        
    # --- Gemini Double-Check (Only if requested) ---
    if use_ai_validation:
        try:
            top_crops = validate_with_gemini(top_crops, location)
        except Exception as e:
            print(f"⚠️ Validation step failed: {e}")
        
    live_prices = {}
    try:
        live_prices = get_live_crop_prices()
    except Exception as e:
        print(f"⚠️ Live prices fetch failed: {e}")
    future_prices = {crop: get_future_price_ai(crop, location) for crop, _ in top_crops}
    ranked_top3 = rank_top_3(top_crops, live_prices, future_prices)
    
    rotation_plan = {}
    if ranked_top3 and ranked_top3[2] != "N/A": # Use the balanced choice for rotation
        rotation_plan = get_crop_rotation_plan(ranked_top3[2], location)
        
    # Formatting results into a nice JSON structure for the frontend
    comparison_table = [
        {
            "crop": c.title(),
            "suitability_percent": p,
            "live_price_rs_kg": f'₹{live_prices.get(c, 0):.2f}',
            "predicted_future_price_rs_kg": f'₹{future_prices.get(c, -1.0):.2f}' if future_prices.get(c, -1.0) > 0 else 'N/A',
            "recommendation_color": traffic_light_color(p)
        } for c, p in top_crops
    ]
    
    # Save the results in the background
    save_results(final_data, top_crops)

    return {
        "input_parameters": final_data,
        "location_analyzed": location,
        "comparison_table": comparison_table,
        "top_3_recommendations": {
            "best_revenue": ranked_top3[0],
            "transport_friendly": ranked_top3[1],
            "balanced_choice": ranked_top3[2]
        },
        "smart_rotation_plan": rotation_plan
    }


