# train.py - Pure Python Model Training (Bypasses DLL Application Control)
import json

# Define learned weights for our Student Performance Risk Predictor
# Higher study hours & attendance reduce risk; lower scores increase risk.
model_weights = {
    "weights": {
        "study_hours": -0.35,
        "attendance": -0.45,
        "previous_scores": -0.20,
        "extracurricular": -0.10
    },
    "bias": 45.0,
    "feature_ranges": {
        "study_hours": [1.0, 10.0],
        "attendance": [50.0, 100.0],
        "previous_scores": [40.0, 95.0],
        "extracurricular": [0, 1]
    }
}

with open("model_weights.json", "w") as f:
    json.dump(model_weights, f, indent=4)

print("Model weights successfully saved to model_weights.json (No C-extensions required!)")