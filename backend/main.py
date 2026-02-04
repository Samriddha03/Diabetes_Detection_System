from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint, confloat, validator
import joblib
import numpy as np
import os

# ----------------------
# Helper function to convert numpy types to Python float
# ----------------------
def to_float(value):
    if isinstance(value, np.ndarray):
        return float(value.item())
    elif isinstance(value, (np.floating, np.integer)):
        return float(value)
    else:
        return float(value)

# ----------------------
# Pydantic Models
# ----------------------
class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float

class PatientData(BaseModel):
    Pregnancies: conint(ge=0)                          # ≥0
    Glucose: confloat(gt=0, le=300)                    # realistic range
    BloodPressure: confloat(gt=0, le=200)
    SkinThickness: confloat(ge=0, le=100)
    Insulin: confloat(ge=0, le=900)
    BMI: confloat(gt=0, le=70)
    DiabetesPedigreeFunction: confloat(gt=0, le=2)
    Age: conint(gt=0, le=120)

    # Optional custom validator for Glucose
    @validator("Glucose")
    def check_glucose(cls, v):
        if v < 70:
            raise ValueError("Glucose too low, might be hypoglycemia")
        return v

# ----------------------
# FastAPI App
# ----------------------
app = FastAPI(
    title="Diabetes Disease Detection API",
    version="1.2"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Load Model and Metrics
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pipeline = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
metrics_data = joblib.load(os.path.join(BASE_DIR, "metrics.pkl"))

# ----------------------
# Routes
# ----------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "metrics_available": True,
        "version": "1.2"
    }

@app.post("/predict")
def predict(data: PatientData):
    try:
        input_data = np.array([[
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.SkinThickness,
            data.Insulin,
            data.BMI,
            data.DiabetesPedigreeFunction,
            data.Age
        ]])

        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        return {
            "prediction": int(prediction),
            "probability": round(to_float(probability), 6),
            "result": "Diabetes Detected" if int(prediction) == 1 else "No Diabetes"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return {
        "accuracy": round(to_float(metrics_data["accuracy"]), 3),
        "precision": round(to_float(metrics_data["precision"]), 3),
        "recall": round(to_float(metrics_data["recall"]), 3),
        "f1_score": round(to_float(metrics_data["f1_score"]), 3)
    }
