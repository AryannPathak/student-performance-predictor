import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('model.pkl')

st.title('🎓 Student Performance Risk Predictor')
st.write('Enter student metrics below to evaluate academic standing.')

# User Inputs via Sliders and Number Inputs
hours = st.slider('Hours Studied per Day', 1.0, 12.0, 4.0)
attendance = st.slider('Attendance Percentage (%)', 10.0, 100.0, 75.0)
prev_score = st.number_input('Previous Exam Score (%)', 0.0, 100.0, 65.0)

# Prediction Button
if st.button('Predict Status'):
    input_data = np.array([[hours, attendance, prev_score]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success('🟢 **Low Risk:** Student is on track to perform well!')
    else:
        st.error('🔴 **At Risk:** Student may need additional academic support.')
