# Student Performance Risk Predictor & MLOps Suite

A production-grade end-to-end Machine Learning web application and API service designed to predict academic risk, serve predictions via a FastAPI backend, and provide transparent model decisions using Explainable AI (SHAP).

## 🚀 Features
1. **Interactive Streamlit Dashboard (`app.py`):** Real-time student risk assessment UI with live sliders and metrics.
2. **Explainable AI (SHAP):** Visual waterfall plots explaining feature contributions for every individual prediction.
3. **Production FastAPI Service (`main.py`):** Fully typed REST API with automatic Swagger UI documentation (`/docs`).
4. **Containerized Deployment (`Dockerfile`):** Ready for cloud deployment.

## 🛠️ Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train model artifact
python train.py

# Run Streamlit UI
streamlit run app.py

# Run FastAPI backend
uvicorn main:app --reload
```
