import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from . import db, ml, risk

app = FastAPI(title="FarmGuard AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # tighten in production
MAX_BYTES = 8 * 1024 * 1024

@app.get("/api/health")
def health():
    return {"ok": True, "model_loaded": ml.available()}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...), crop: str = Form(...), language: str = Form("en")):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 8 MB.")
    if not ml.available():
        db.save(crop, language, "demo", None, None)
        return {"mode": "demo", "prediction": None, "confidence": None,
                "message": "No trained model found. Add backend/models/model.keras and labels.json (see README)."}
    try:
        res = ml.predict(raw)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
    db.save(crop, language, "model", res["prediction"], res["confidence"])
    return {"mode": "model", **res}

@app.post("/api/gradcam")
async def gradcam(file: UploadFile = File(...)):
    if not ml.available():
        raise HTTPException(503, "No model available.")
    return {"heatmap": ml.gradcam(await file.read())}

@app.get("/api/history")
def history(): return db.history()

@app.get("/api/risk")
def get_risk(temp_c: float, humidity: float, rain_days_30: int = 0, growth_stage: str = "vegetative", prior_disease: bool = False):
    # Weather values must come from the caller or a real weather API; nothing is invented here.
    return risk.estimate_risk(temp_c, humidity, rain_days_30, growth_stage, prior_disease)

@app.get("/api/simulate")
def simulate(spray_interval: int = 14, scouting_per_week: int = 1, rainy_days: int = 8):
    return risk.simulate(spray_interval, scouting_per_week, rainy_days)

FRONT = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
app.mount("/", StaticFiles(directory=FRONT, html=True), name="frontend")
