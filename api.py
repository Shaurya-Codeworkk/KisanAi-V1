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

# --- Static/templates setup (add/replace this block in api.py after imports) ---
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Prefer templates/index.html if you intentionally used Jinja templates
TEMPLATE_PATH = os.path.join("templates", "index.html")
STATIC_PATH = os.path.join("static", "index.html")
ROOT_INDEX = "index.html"

# Mount /static so your CSS/JS/images load if you store them in static/
if os.path.isdir("static"):
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
        print("Mounted /static -> ./static")
    except Exception as e:
        print("Failed to mount /static:", e)

# Keep Jinja templates for other endpoints, but fall back to static files
templates = Jinja2Templates(directory="templates")

def _find_index_file():
    """Return (path_type, path) where path_type is 'templates'|'static'|'root' or None."""
    if os.path.exists(TEMPLATE_PATH):
        return "templates", TEMPLATE_PATH
    if os.path.exists(STATIC_PATH):
        return "static", STATIC_PATH
    if os.path.exists(ROOT_INDEX):
        return "root", ROOT_INDEX
    return None, None

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_homepage(request: Request):
    """
    Serve the frontend index.html. Tries templates/index.html, static/index.html, then ./index.html.
    Returns helpful error if none are present.
    """
    location, path = _find_index_file()
    if location == "templates":
        try:
            return templates.TemplateResponse("index.html", {"request": request})
        except Exception as e:
            return HTMLResponse(f"<h2>Template render error</h2><pre>{e}</pre>", status_code=500)
    elif location in ("static", "root"):
        # FileResponse will serve the raw HTML file
        try:
            return FileResponse(path, media_type="text/html")
        except Exception as e:
            return HTMLResponse(f"<h2>File serve error</h2><pre>{e}</pre>", status_code=500)

    # not found — give a clear message so you can fix files
    debug = {
        "templates_exists": os.path.exists("templates"),
        "static_exists": os.path.exists("static"),
        "root_index_exists": os.path.exists("index.html"),
        "cwd": os.getcwd(),
        "list_dir": os.listdir(".")[:40]  # first 40 entries
    }
    return HTMLResponse(
        content=f"<h2>index.html not found</h2>"
                f"<p>Checked: templates/index.html, static/index.html, ./index.html</p>"
                f"<pre>{debug}</pre>",
        status_code=500
    )
