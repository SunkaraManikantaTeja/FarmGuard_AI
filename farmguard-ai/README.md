# FarmGuard AI

Frontend (plain HTML/CSS/JS) + FastAPI backend + SQLite.

## Run
```bash
cd backend
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000 (the backend serves the frontend). API docs: http://127.0.0.1:8000/docs

## Real predictions (optional)
1. `pip install tensorflow`
2. Download PlantVillage (one folder per class) and run `python train.py path/to/PlantVillage`
3. This writes `models/model.keras` and `models/labels.json`; restart the server.
Without a model the API returns `mode: "demo"` and the UI shows the Demonstration label. Nothing is faked.

## Endpoints
GET /api/health · POST /api/predict · POST /api/gradcam · GET /api/history · GET /api/risk · GET /api/simulate

## Honesty notes
Risk and simulator values are transparent estimates with assumed coefficients, not validated forecasts. Report only accuracy you measure yourself. Weather must come from a real API or user input.

## Next steps
Move frontend to React + Vite, add Voice Assistant (Web Speech API), Dashboard and Community pages, PostgreSQL via SQLAlchemy.
