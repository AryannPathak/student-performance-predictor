# FastAPI Backend for Student Performance Risk Prediction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="Student Performance Risk API",
    description="Production-grade MLOps inference API powered by FastAPI & scikit-learn",
    version="1.0.0"
)

try:
    model = joblib.load("model.pkl")
except Exception as e:
    model = None

class StudentInput(BaseModel):
    study_hours: float
    attendance: float
    previous_scores: float
    extracurricular: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Performance Risk Predictor API. Visit /docs for Swagger UI."}

@app.post("/predict")
def predict_risk(data: StudentInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact not found. Please train the model first.")
    
    features = pd.DataFrame([[
        data.study_hours,
        data.attendance,
        data.previous_scores,
        data.extracurricular
    ]], columns=['study_hours', 'attendance', 'previous_scores', 'extracurricular'])
    
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    
    return {
        "risk_prediction": prediction,
        "risk_label": "High Risk" if prediction == 1 else "Low Risk",
        "risk_probability": round(probability * 100, 2)
    }
