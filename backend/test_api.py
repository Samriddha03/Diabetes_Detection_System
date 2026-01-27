import requests

# Test /metrics endpoint
metrics = requests.get("http://127.0.0.1:8000/metrics").json()
print("Metrics:", metrics)

# Test /predict endpoint
data = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 79,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 33
}
prediction = requests.post("http://127.0.0.1:8000/predict", json=data).json()
print("Prediction:", prediction)
