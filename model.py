import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Sample dataset for student performance prediction
data = {
    'Hours_Studied': [2, 5, 8, 1, 6, 9, 3, 7, 4, 10],
    'Attendance_Percentage': [50, 75, 90, 40, 80, 95, 60, 85, 70, 98],
    'Previous_Score': [55, 70, 88, 45, 78, 92, 62, 85, 68, 95],
    'Passed': [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]  # 1 = Pass, 0 = At Risk
}
df = pd.DataFrame(data)

X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
y = df['Passed']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model artifact
joblib.dump(model, 'model.pkl')
print("Model trained and saved successfully!")