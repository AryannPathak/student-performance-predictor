import json
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Student Risk MLOps Suite",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Glassmorphic Light Theme Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        color: #102a43;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.4);
    }
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

# Bulletproof weights loading
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

# Sidebar Navigation Menu
st.sidebar.markdown("### 🧭 Dashboard Menu")
app_mode = st.sidebar.radio("Navigation", ["Overview", "Students", "Analytics", "Settings"])
st.sidebar.markdown("---")

# 1. OVERVIEW PAGE
if app_mode == "Overview":
    st.sidebar.header("🎛️ Risk Parameters")
    study_hours = st.sidebar.slider("Weekly Study Hours", 1, 40, 18)
    attendance = st.sidebar.slider("Attendance Rate (%)", 40, 100, 85)
    prev_score = st.sidebar.slider("Previous Semester Score", 30, 100, 70)

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

    score_val = (
        (study_hours / 40.0) * weights[0] + 
        (attendance / 100.0) * weights[1] + 
        (prev_score / 100.0) * weights[2] + 
        bias
    )
    risk_percentage = max(0, min(100, int((1 - (score_val + 2) / 4) * 100)))

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

# 2. STUDENTS PAGE
elif app_mode == "Students":
    st.title("👥 Student Records Directory")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Active Batch Risk Monitoring")
    
    df = pd.DataFrame({
        "Student ID": ["STU-101", "STU-102", "STU-103", "STU-104", "STU-105"],
        "Name": ["Aarav Sharma", "Priya Nair", "Rahul Verma", "Sneha Patil", "Karan Mehta"],
        "Study Hours": [18, 12, 30, 22, 14],
        "Attendance (%)": [85, 60, 92, 78, 65],
        "Risk Status": ["Low Risk", "High Risk", "Low Risk", "Moderate Risk", "Moderate Risk"]
    })
    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 3. ANALYTICS PAGE
elif app_mode == "Analytics":
    st.title("📈 Performance Analytics & Trends")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Study Hours vs. Academic Performance Correlation")
    
    chart_data = pd.DataFrame({
        "Study Hours": [5, 10, 15, 20, 25, 30, 35, 40],
        "Predicted Score": [50, 60, 72, 80, 88, 92, 95, 98]
    })
    st.line_chart(chart_data.set_index("Study Hours"))
    st.markdown("</div>", unsafe_allow_html=True)

# 4. SETTINGS PAGE
elif app_mode == "Settings":
    st.title("⚙️ MLOps Configuration & Settings")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Model Weights & Parameters")
    st.json({"weights": weights, "bias": bias, "framework": "Scikit-Learn / Streamlit"})
    st.write("Environment: Production Inference Container")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("🚀 Powered by Aryan Pathak")