# ------------------- mini_kisan_copilot.py (Upgraded to use Google Gemini) -------------------

import os
import json
import requests
import re
import google.generativeai as genai
from datetime import datetime

# --- Config ---
# Use the GOOGLE_API_KEY environment variable for Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CACHE_FILE = "gemini_cache.json"

# --- Load cache ---
try:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            gemini_cache = json.load(f)
    else:
        gemini_cache = {}
except Exception as e:
    print(f"⚠️ Cache load failed: {e}")
    gemini_cache = {}

# --- Configure Gemini client if key present ---
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"⚠️ Failed to configure google.generativeai: {e}")
else:
    print("⚠️ GOOGLE_API_KEY not set. Gemini calls will be disabled and fallbacks used.")

# --- Helper: safe JSON parse for AI text outputs ---
def _safe_parse_json_from_text(raw_text, fallback=None):
    if fallback is None:
        fallback = {}
    try:
        text = raw_text.strip()
        # if there's a ```json block, extract it
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        # otherwise if there's a ``` block, extract it
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        # try straight json load
        return json.loads(text)
    except Exception:
        # last attempt: find first { ... } block
        try:
            match = re.search(r"(\{[\s\S]*\})", raw_text)
            if match:
                return json.loads(match.group(1))
        except Exception:
            pass
    return fallback

# --- Gemini wrapper (replacement for Groq wrapper) ---
def ask_gemini_ai(prompt_key: str, system_prompt: str, user_prompt: str, is_json: bool = True, temperature: float = 0.3):
    """
    Ask Gemini (google.generativeai). Caches by prompt_key.
    If GOOGLE_API_KEY is not set, returns {} for JSON calls or an error string for text calls.
    """
    key_lower = prompt_key.lower().replace(" ", "_")
    if key_lower in gemini_cache:
        return gemini_cache[key_lower]

    if not GOOGLE_API_KEY:
        print("❌ CRITICAL: GOOGLE_API_KEY environment variable is not set!")
        return {} if is_json else "Error: API key is not configured on the server."

    try:
        full_prompt = f"{system_prompt}\n\nUser Query: {user_prompt}\n\nIMPORTANT: Return ONLY valid JSON." if is_json else f"{system_prompt}\n\nUser Query: {user_prompt}"

        model_ai = genai.GenerativeModel("gemini-pro")
        # generate_content returns an object with .text in current SDK usage
        response = model_ai.generate_content(full_prompt, temperature=temperature)
        content = getattr(response, "text", "") or str(response)

        if is_json:
            data = _safe_parse_json_from_text(content, {})
            if data:
                gemini_cache[key_lower] = data
                try:
                    with open(CACHE_FILE, "w") as f:
                        json.dump(gemini_cache, f, indent=2)
                except Exception as e:
                    print(f"⚠️ Failed to write cache file: {e}")
            return data
        else:
            # store plain text responses as well (if desired)
            gemini_cache[key_lower] = content
            try:
                with open(CACHE_FILE, "w") as f:
                    json.dump(gemini_cache, f, indent=2)
            except Exception:
                pass
            return content

    except Exception as e:
        print(f"!!!!!!!!!! GEMINI API CALL FAILED for key: {prompt_key} !!!!!!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {} if is_json else f"Sorry, an error occurred while talking to the AI."

# --- Public functions (port of previous groq logic) ---

def get_live_crop_prices(crop_list: list):
    """
    Returns prices per kg for crops using Gemini (prices in Rs/quintal from AI, converted to Rs/kg).
    Provides defaults for crops the AI misses.
    """
    if not crop_list:
        return {}

    crops_string = ", ".join(crop_list)
    prompt = f"""
    Provide current average mandi prices in India (in Rs/quintal) for the following crops: {crops_string}.
    Return ONLY a valid JSON object in the format: {{"crop1": price1, "crop2": price2, ...}}
    For example: {{"rice": 4500, "wheat": 2350}}
    """
    system_prompt = "You are a data provider. Return only a valid JSON object."

    prices_per_quintal = ask_gemini_ai(f"live_prices_{'_'.join(crop_list)}", system_prompt, prompt, is_json=True)

    prices_per_kg = {}
    if prices_per_quintal:
        try:
            for crop, price in prices_per_quintal.items():
                prices_per_kg[crop.lower()] = float(price) / 100.0
        except Exception:
            # If parsing failed, ignore and fill with defaults later
            prices_per_kg = {}

    # Add defaults for any crops the AI missed
    defaults = {"rice": 45.0, "wheat": 25.0, "cotton": 80.0, "jute": 60.0, "coffee": 150.0/100.0, "mango": 120.0/100.0}
    for crop in crop_list:
        if crop.lower() not in prices_per_kg:
            prices_per_kg[crop.lower()] = defaults.get(crop.lower(), 'N/A')

    return prices_per_kg

def get_crop_dynamic_details(crop_name: str):
    """
    Returns a structured JSON with crop details (definition, soil, irrigation, fertilizer, pesticides, quick_tip).
    """
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
    data = ask_gemini_ai(f"crop_details_{crop_name}", system_prompt, prompt, is_json=True)

    # defaults to avoid missing fields
    defaults = {
        "definition": f"{crop_name.title()} is a major crop in India with various agricultural uses.",
        "soil": "Varies by region, but generally well-drained soil is preferred.",
        "irrigation": "Requires regular watering, especially during dry seasons.",
        "fertilizer": "A balanced NPK fertilizer is recommended.",
        "pesticides": "Monitor for common pests and use appropriate treatments.",
        "quick_tip": f"Ensure proper soil preparation before planting {crop_name} for best results."
    }

    if not isinstance(data, dict):
        data = {}

    for key in defaults:
        if key not in data or not data.get(key):
            data[key] = defaults[key]

    return data

def extract_crops_from_query(user_query: str) -> list:
    """
    Uses AI to pull crop names from free text. Falls back to a regex list if AI fails.
    """
    prompt = f"""
    From the following text, extract a list of all Indian crop names mentioned.
    Return ONLY a valid JSON object with a single key "crops" which is a list of strings.
    Example: {{ "crops": ["rice", "potato"] }}
    Text: "{user_query}"
    """
    system_prompt = "You are a text analysis tool. Return only a valid JSON object."
    result = ask_gemini_ai(f"extract_{user_query}", system_prompt, prompt, is_json=True)

    # Fallback to simple regex if AI fails
    if not result or "crops" not in result:
        known_crops = ["rice", "wheat", "cotton", "jute", "coffee", "mango", "potato", "maize", "pigeonpeas"]
        found_crops = [crop for crop in known_crops if re.search(r'\b' + re.escape(crop) + r'\b', user_query.lower())]
        return found_crops

    return result.get("crops", [])

def mini_copilot_response(user_query: str):
    """
    Builds a friendly multi-crop mini-report using the helper functions above.
    """
    print(f"🤖 Analyzing query: '{user_query}'...")
    mentioned_crops = extract_crops_from_query(user_query)

    if not mentioned_crops:
        return "🤖 Mini Kisan Co-pilot: Sorry bhai, I couldn't identify any specific crops in your question. Please ask me something like 'tell me about wheat and rice'."

    print(f"🌱 Crops identified: {', '.join(mentioned_crops)}")

    prices = get_live_crop_prices(mentioned_crops)
    response_lines = ["🤖 Mini Kisan Co-pilot Report:\n"]

    for crop in mentioned_crops:
        crop_info = get_crop_dynamic_details(crop)
        live_price = prices.get(crop.lower(), 'N/A')
        response_lines.append(f"--- 🌾 {crop.title()} ---")
        response_lines.append(f"📖 About: {crop_info.get('definition', 'N/A')}")
        response_lines.append(f"💰 Live Price: Rs {live_price}/kg")
        response_lines.append(f"🌱 Soil: {crop_info.get('soil', 'N/A')}")
        response_lines.append(f"💧 Irrigation: {crop_info.get('irrigation', 'N/A')}")
        response_lines.append(f"💊 Fertilizer: {crop_info.get('fertilizer', 'N/A')}")
        response_lines.append(f"🐞 Pests: {crop_info.get('pesticides', 'N/A')}")
        response_lines.append(f"💡 Quick Tip: {crop_info.get('quick_tip', 'N/A')}\n")

    if len(mentioned_crops) > 1 and prices:
        response_lines.append("--- 📊 Quick Price Comparison ---")
        # sort by numeric price where possible (fallback 'N/A' handled)
        def price_key(item):
            try:
                return float(item[1])
            except Exception:
                return -1.0
        sorted_by_price = sorted(prices.items(), key=price_key, reverse=True)
        for rank, (crop, price) in enumerate(sorted_by_price, start=1):
            response_lines.append(f"{rank}. {crop.title()} → Rs {price}/kg")

    return "\n".join(response_lines)

# --- CLI for Testing ---
if __name__ == "__main__":
    print("=== Mini Kisan Co-pilot CLI (Gemini) ===")
    print("Ask me about any crop, like 'what is potato?' or 'compare rice and wheat'")
    while True:
        query = input("\nAsk me about crops (or type 'exit'): ")
        if query.lower() == "exit":
            break
        print(mini_copilot_response(query))
