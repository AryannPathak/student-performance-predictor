# app.py - Pure Python Streamlit Dashboard & Explainability
import streamlit as st
import json
import math

st.set_page_config(page_title="Student Performance Risk Predictor", page_icon="🎓", layout="wide")

@st.cache_data
def load_model():
    try:
        with open("model_weights.json", "r") as f:
            return json.load(f)
    except:
        return None

model_data = load_model()

st.title("🎓 Student Performance Risk Predictor & Explainable AI")
st.markdown("Production-grade academic risk dashboard optimized for secure institutional environments.")

if model_data is None:
    st.error("Model weights not found! Please run `train.py` first.")
else:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input Student Parameters")
        study_hours = st.slider("Weekly Study Hours", 1.0, 10.0, 5.0, 0.5)
        attendance = st.slider("Attendance Percentage (%)", 50.0, 100.0, 80.0, 1.0)
        previous_scores = st.slider("Previous Semester Score", 40.0, 95.0, 70.0, 1.0)
        extracurricular = st.selectbox("Extracurricular Engagement", [0, 1], format_func=lambda x: "Active" if x == 1 else "Inactive")
        
        if st.button("Predict Academic Risk", type="primary"):
            w = model_data["weights"]
            bias = model_data["bias"]
            
            # Linear combination
            score = (w["study_hours"] * study_hours + 
                     w["attendance"] * attendance + 
                     w["previous_scores"] * previous_scores + 
                     w["extracurricular"] * extracurricular + bias)
            
            # Sigmoid probability
            prob = 1 / (1 + math.exp(-score / 15.0))
            pred = 1 if prob >= 0.5 else 0
            
            # Feature attributions (Contribution to risk)
            attributions = {
                "Weekly Study Hours": w["study_hours"] * study_hours,
                "Attendance (%)": w["attendance"] * attendance,
                "Previous Scores": w["previous_scores"] * previous_scores,
                "Extracurricular": w["extracurricular"] * extracurricular
            }
            
            st.session_state['pred'] = pred
            st.session_state['prob'] = prob
            st.session_state['attributions'] = attributions

    with col2:
        st.subheader("📊 Risk Assessment & Explainability")
        if 'pred' in st.session_state:
            pred = st.session_state['pred']
            prob = st.session_state['prob']
            
            if pred == 1:
                st.error(f"⚠️ **High Academic Risk Detected** (Risk Probability: `{prob*100:.1f}%`)")
            else:
                st.success(f"✅ **Low Academic Risk / On Track** (Risk Probability: `{prob*100:.1f}%`)")
            
            st.markdown("---")
            st.markdown("### 🔍 Feature Impact Breakdown (Why this prediction?)")
            
            # Display clean feature contributions
            for feat, val in st.session_state['attributions'].items():
                st.metric(label=feat, value=f"{val:.2f} impact score")
        else:
            st.info("Adjust the student metrics on the left and click **Predict Academic Risk** to view real-time inference and explainability.")

    st.markdown("---")
    st.markdown("### 🚀 Project Architecture")
    st.markdown("- **Frontend:** Streamlit Web UI")
    st.markdown("- **Inference Engine:** Pure Python Logistic Scoring (WDAC Policy Compliant)")