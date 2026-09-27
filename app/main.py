from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.joblib"

app = FastAPI(title="F1 Winner Prediction API", version="1.0.0")

_model = None
_feature_columns = []

class PredictionRequest(BaseModel):
    grid: int = Field(ge=1, le=24)
    driver_form: float = Field(ge=1.0, le=30.0)
    constructor_strength: float = Field(ge=0.0, le= 30.0)
    constructor: str
    circuit: str

class PredictionResponse(BaseModel):
    win_probability: float
    model: str

@app.on_event("startup")
def load_model():
    global _model, _feature_columns
    bundle = joblib.load(MODEL_PATH)
    _model = bundle["model"]
    _feature_columns = bundle["feature_columns"]

def build_vector(req: PredictionRequest) -> np.ndarray:
    """ To rebuild the exact 61-column training vector from a 
    friendly prediction request."""
    row = {c: 0.0 for c in _feature_columns}
    row["grid"] = float(req.grid)
    row["driver_form"] = float(req.driver_form)
    row["constructor_strength"] = float(req.constructor_strength)
    for key in (f"constructor_{req.constructor}", f"circuit_{req.circuit}"):
        if key in row:
            row[key] = 1.0
    return np.array([[row[c] for c in _feature_columns]], dtype=float)

@app.get("/health")
def health()->dict:
    return {"status":"ok", "model_loaded": _model is not None}

@app.post("/predict", response_model= PredictionResponse)
def predict(req: PredictionRequest)-> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    proba = float(_model.predict_proba(build_vector(req))[0,1])
    return PredictionResponse(win_probability=round(proba,2), model="gradient_boosting")
            
