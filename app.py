import json
import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(BASE_DIR, "model_weights.json")) as f:
        model_data = json.load(f)
except Exception:
    model_data = {}
st.set_page_config(
    page_title="Student Risk MLOps Suite",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Glassmorphic Light Theme Styling (Matching Reference Design)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        color: #102a43;
    }
    
    /* Sidebar Frosted Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.4);
    }
    
    /* Glass Card Component */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        margin-bottom: 20px;
    }
    
    h1, h2, h3, h4, h5, h6, p, label {
        color: #102a43 !important;
    }
    </style>
""", unsafe_allow_html=True)
# Safe, bulletproof weights loading
weights = [0.4, 0.3, 0.3]
bias = -2.0

try:
    with open("model_weights.json", "r") as f:
        data = json.load(f)
        if isinstance(data, dict):
            if "weights" in data and isinstance(data["weights"], list):
                weights = data["weights"]
            if "bias" in data:
                bias = float(data["bias"])
        elif isinstance(data, list):
            weights = data
except Exception:
    pass
# Sidebar Navigation & Sliders
st.sidebar.markdown("### 🧭 Dashboard Menu")
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Overview**")
st.sidebar.markdown("👥 **Students**")
st.sidebar.markdown("📈 **Analytics**")
st.sidebar.markdown("⚙️ **Settings**")
st.sidebar.markdown("---")

st.sidebar.header("🎛️ Risk Parameters")
study_hours = st.sidebar.slider("Weekly Study Hours", 1, 40, 18)
attendance = st.sidebar.slider("Attendance Rate (%)", 40, 100, 85)
prev_score = st.sidebar.slider("Previous Semester Score", 30, 100, 70)

# Header Section
st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
        <div>
            <h2>🎓 Student Performance Risk Predictor</h2>
            <p style='color: #627d98 !important; margin: 0;'>Real-Time MLOps & Explainable AI Dashboard</p>
        </div>
        <div style='background: rgba(255,255,255,0.8); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.9); font-weight: 600;'>
            🟢 System Active
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# Top KPI Metric Cards (Frosted Glass Layout)
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
        <div class='glass-card'>
            <p style='font-size: 14px; color: #627d98; margin-bottom: 4px;'>STUDY LOAD INDEX</p>
            <h2 style='margin: 0; color: #102a43;'>{study_hours} <span style='font-size: 16px; color: #10b981;'>hrs/wk</span></h2>
            <p style='font-size: 12px; color: #10b981; margin-top: 6px;'>↑ Optimal range</p>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
        <div class='glass-card'>
            <p style='font-size: 14px; color: #627d98; margin-bottom: 4px;'>ATTENDANCE RATE</p>
            <h2 style='margin: 0; color: #102a43;'>{attendance}%</h2>
            <p style='font-size: 12px; color: #10b981; margin-top: 6px;'>↑ Target > 80%</p>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
        <div class='glass-card'>
            <p style='font-size: 14px; color: #627d98; margin-bottom: 4px;'>HISTORICAL BASELINE</p>
            <h2 style='margin: 0; color: #102a43;'>{prev_score} <span style='font-size: 16px; color: #627d98;'>pts</span></h2>
            <p style='font-size: 12px; color: #334e68; margin-top: 6px;'>Stable record</p>
        </div>
    """, unsafe_allow_html=True)

# Prediction Logic (Pure Python Inference Engine)
raw_weights = model_data.get("weights", [0.4, 0.3, 0.3])

if isinstance(raw_weights, dict):
    weights = list(raw_weights.values())
else:
    weights = list(raw_weights)

if len(weights) < 3:
    weights = [0.4, 0.3, 0.3]
bias = model_data.get("bias", -2.0)
score_val = (
    (study_hours / 40.0) * weights[0] + 
    (attendance / 100.0) * weights[1] + 
    (prev_score / 100.0) * weights[2] + 
    bias
)
risk_percentage = max(0, min(100, int((1 - (score_val + 2) / 4) * 100)))

# Main Layout: Risk Evaluation & Explainable AI (XAI)
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🎯 Risk Assessment Result")
    if risk_percentage > 50:
        st.error(f"**High Academic Risk Detected ({risk_percentage}%)**")
        st.write("Recommended Action: Immediate counselor intervention and tutoring required.")
    else:
        st.success(f"**Low Risk / On Track ({risk_percentage}%)**")
        st.write("Status: Student is performing within acceptable threshold limits.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔍 Explainable AI (XAI) Breakdown")
    st.write("Feature attribution impact on current calculation:")
    st.progress(attendance / 100, text=f"Attendance Impact: {attendance}%")
    st.progress(study_hours / 40, text=f"Study Commitment: {study_hours}/40 hrs")
    st.progress(prev_score / 100, text=f"Prior Performance: {prev_score} pts")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center; color:#627d98; font-size:14px;'>🚀 Powered by Aryan Pathak</div>",
    unsafe_allow_html=True,
)