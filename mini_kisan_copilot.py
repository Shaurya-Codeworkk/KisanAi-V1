# ------------------- mini_kisan_copilot.py (Upgraded by Gemini) -------------------

import os
import json
import requests
from groq import Groq
import re

# --- Config ---
# Securely gets the API key from environment variables.
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
CACHE_FILE = "groq_cache.json"

# --- Load cache ---
# Note: On Render's free tier, this cache file will be deleted when the server sleeps.
try:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            groq_cache = json.load(f)
    else:
        groq_cache = {}
except Exception as e:
    print(f"⚠️ Cache load failed: {e}")
    groq_cache = {}

# --- Improved Groq API Wrapper ---
def ask_groq_ai(prompt_key: str, system_prompt: str, user_prompt: str, is_json: bool = True):
    """
    Handles calls to the Groq API, with improved error handling and caching.
    """
    key_lower = prompt_key.lower().replace(" ", "_")
    if key_lower in groq_cache:
        return groq_cache[key_lower]
        
    if not GROQ_API_KEY:
        print("❌ CRITICAL: GROQ_API_KEY environment variable is not set!")
        return {} if is_json else "Error: API key is not configured on the server."
        
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Set response format only if JSON is requested
        response_format = {"type": "json_object"} if is_json else None
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",  # <-- UPGRADED MODEL NAME
            temperature=0.5,
            response_format=response_format
        )
        content = chat_completion.choices[0].message.content
        
        # Parse the content if JSON is expected
        data = json.loads(content) if is_json else content
        
        # Update cache
        groq_cache[key_lower] = data
        with open(CACHE_FILE, "w") as f:
            json.dump(groq_cache, f, indent=2)
            
        return data
    except Exception as e:
        print(f"!!!!!!!!!! GROQ API CALL FAILED for key: {prompt_key} !!!!!!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {} if is_json else f"Sorry, an error occurred while talking to the AI."

# In mini_kisan_copilot.py

def get_live_crop_prices(crop_list: list):
    crops_string = ", ".join(crop_list)
    # Ask for the price per QUINTAL (100 kg)
    prompt = f"""
    Provide current average mandi prices in India (in Rs/quintal) for the following crops: {crops_string}.
    Return ONLY a valid JSON object in the format: {{"crop1": price1, "crop2": price2, ...}}
    For example: {{"rice": 4500, "wheat": 2350}}
    """
    system_prompt = "You are a data provider. Return only a valid JSON object."
    prices_per_quintal = ask_groq_ai(f"live_prices_{'_'.join(crop_list)}", system_prompt, prompt, is_json=True)

    prices_per_kg = {}
    if prices_per_quintal:
        for crop, price in prices_per_quintal.items():
            # Divide by 100 to convert to per/kg
            prices_per_kg[crop.lower()] = float(price) / 100

    # Add defaults for any crops the AI missed
    defaults = {"rice": 45.0, "wheat": 25.0, "cotton": 80.0}
    for crop in crop_list:
        if crop.lower() not in prices_per_kg:
            prices_per_kg[crop.lower()] = defaults.get(crop.lower(), 'N/A')

    return prices_per_kg

# --- Crop Dynamic Details ---
# In mini_kisan_copilot.py

def get_crop_dynamic_details(crop_name: str):
    prompt = f"""
    You are an expert Indian agronomist.
    Provide a structured JSON object ONLY with the following keys for the crop "{crop_name}":
    "definition": A clear, friendly explanation of the crop, its uses, main growing regions in India, and importance.
    "soil": Ideal soil type.
    "irrigation": General irrigation needs.
    "fertilizer": Common fertilizer recommendations (NPK).
    "pesticides": Common pest management advice.
    "quick_tip": One single, actionable tip for the farmer.
    """
    system_prompt = "You are an expert agronomist. Return only a valid JSON object."
    data = ask_groq_ai(f"crop_details_{crop_name}", system_prompt, prompt, is_json=True)

    # --- THIS IS THE NEW PART ---
    # Add default values to prevent "N/A"
    defaults = {
        "definition": f"{crop_name.title()} is a major crop in India with various agricultural uses.",
        "soil": "Varies by region, but generally well-drained soil is preferred.",
        "irrigation": "Requires regular watering, especially during dry seasons.",
        "fertilizer": "A balanced NPK fertilizer is recommended.",
        "pesticides": "Monitor for common pests and use appropriate treatments.",
        "quick_tip": f"Ensure proper soil preparation before planting {crop_name} for best results."
    }

    # Fill in any missing keys with default values
    for key in defaults:
        if key not in data or not data[key]:
            data[key] = defaults[key]

    return data

# --- NEW FEATURE: Smart Crop Extractor ---
def extract_crops_from_query(user_query: str) -> list:
    """Uses AI to dynamically find crop names mentioned in the user's query."""
    prompt = f"""
    From the following text, extract a list of all Indian crop names mentioned.
    Return ONLY a valid JSON object with a single key "crops" which is a list of strings.
    For example, if the user says "tell me about rice and potato", you should return: {{"crops": ["rice", "potato"]}}
    Text: "{user_query}"
    """
    system_prompt = "You are a text analysis tool. Return only a valid JSON object."
    result = ask_groq_ai(f"extract_{user_query}", system_prompt, prompt, is_json=True)
    
    # Fallback to simple regex if AI fails
    if not result or "crops" not in result:
        known_crops = ["rice", "wheat", "cotton", "jute", "coffee", "mango", "potato", "maize", "pigeonpeas"]
        found_crops = [crop for crop in known_crops if re.search(r'\b' + crop + r'\b', user_query.lower())]
        return found_crops
        
    return result.get("crops", [])

# --- IMPROVED Mini Co-pilot Response ---
def mini_copilot_response(user_query: str):
    print(f"🤖 Analyzing query: '{user_query}'...")
    mentioned_crops = extract_crops_from_query(user_query)

    if not mentioned_crops:
        return "🤖 Mini Kisan Co-pilot: Sorry bhai, I couldn't identify any specific crops in your question. Please ask me something like 'tell me about wheat and rice'."

    print(f"🌱 Crops identified: {', '.join(mentioned_crops)}")
    
    prices = get_live_crop_prices(mentioned_crops)
    response = "🤖 **Mini Kisan Co-pilot Report:**\n\n"

    for crop in mentioned_crops:
        crop_info = get_crop_dynamic_details(crop)
        response += f"--- **🌾 {crop.title()}** ---\n"
        response += f"**📖 About:** {crop_info.get('definition', 'N/A')}\n"
        response += f"**💰 Live Price:** Rs {prices.get(crop, 'N/A')}/kg\n"
        response += f"**🌱 Soil:** {crop_info.get('soil', 'N/A')}\n"
        response += f"**💧 Irrigation:** {crop_info.get('irrigation', 'N/A')}\n"
        response += f"**💊 Fertilizer:** {crop_info.get('fertilizer', 'N/A')}\n"
        response += f"**🐞 Pests:** {crop_info.get('pesticides', 'N/A')}\n"
        response += f"**💡 Quick Tip:** {crop_info.get('quick_tip', 'N/A')}\n\n"

    if len(mentioned_crops) > 1 and prices:
        response += "--- **📊 Quick Price Comparison** ---\n"
        # Sort by price, handling potential non-numeric values
        sorted_by_price = sorted(prices.items(), key=lambda item: float(item[1]) if isinstance(item[1], (int, float)) else -1, reverse=True)
        for rank, (crop, price) in enumerate(sorted_by_price, start=1):
            response += f"{rank}. {crop.title()} → Rs {price}/kg\n"

    return response

# --- CLI for Testing ---
if __name__ == "__main__":
    print("=== Mini Kisan Co-pilot CLI (Upgraded) ===")
    print("Ask me about any crop, like 'what is potato?' or 'compare rice and wheat'")
    while True:
        query = input("\nAsk me about crops (or type 'exit'): ")
        if query.lower() == "exit":
            break
        print(mini_copilot_response(query))


