# 🎓 Student Performance Risk Predictor & Explainable AI (XAI) Suite

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, end-to-end Machine Learning web application and analytics suite designed to predict academic risk, deliver real-time interactive scores, and provide transparent, explainable AI (XAI) insights for educational institutions.

---

## 🚀 Key Features

* **Interactive Streamlit Dashboard (`app.py`):** Real-time student risk assessment UI with responsive sliders for weekly study hours, attendance percentages, and previous scores.
* **Explainable AI (XAI) Engine:** Transparent feature impact breakdown showing *why* a student is flagged as high or low risk (compliant with strict institutional security and WDAC policies).
* **Pure Python Scoring Architecture:** Robust inference engine avoiding restricted binary dependencies, ensuring seamless deployment across local and enterprise environments.
* **Modular MLOps Design:** Clean separation of model training (`train.py`), weight configuration (`model_weights.json`), and front-end visualization.

---

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit, Matplotlib
* **Core Logic:** Python, JSON Weight Serialization
* **Version Control:** Git & GitHub

---

## 📂 Project Structure

```text
student-performance-predictor/
│
├── train.py                # Model training & weight generation script
├── model_weights.json      # Serialized model parameters and bias
├── app.py                  # Main Streamlit web dashboard & XAI engine
├── requirements.txt        # Pinned project dependencies
└── README.md               # Project documentation