# Diabetes Disease Detection System

Diabetes Disease Detection System is a full-stack project that predicts the likelihood of diabetes in patients using a machine learning model. The system includes a FastAPI backend, a trained ML model, and a frontend interface for easy user interaction.

**Diabetes Disease Detection System** is a full-stack project that predicts the likelihood of diabetes in patients using a machine learning model. The system includes a **FastAPI backend**, a **trained ML model**, and a **frontend interface** for easy user interaction.

---

## 🔹 Features

- **Machine Learning Model**
  - Trained using Logistic Regression, Random Forest, and Gradient Boosting.
  - Automatically selects the **best model** based on **F1-score**.
  - Stores trained model (`diabetes_model.pkl`) and metrics (`metrics.pkl`) for deployment.

- **FastAPI Backend**
  - REST API Endpoints:
    - `/predict` – Predict diabetes risk from patient data.
    - `/metrics` – Retrieve model performance metrics.
    - `/health` – Check API health.
  - Input validation using **Pydantic**.
  - **CORS enabled** for frontend integration.
  - Swagger documentation at `/docs`.

- **Frontend Interface**
  - Built with **HTML, CSS, JavaScript**.
  - Enter patient data and display **prediction & probability**.
  - Works seamlessly with backend API.

---

## 🔹 Project Structure
diabetes-detection/
│
├── backend/
│ ├── main.py # FastAPI application
│ ├── diabetes_model.pkl # Trained ML model
│ ├── metrics.pkl # Model metrics
│
├── model_training/
│ ├── train_model.py # Script to train ML models
│ └── diabetes.csv # Dataset for training
│
├── frontend/
│ ├── index.html # Frontend UI
│ ├── style.css
│ └── script.js
│
├── requirements.txt # Python dependencies
└── README.md

---

## 🔹 API Endpoints

### 1. Health Check
GET /health

Response:
```
{
  "status": "healthy",
  "model_loaded": true,
  "metrics_available": true,
  "version": "1.2"
}
```


2. Predict Diabetes
POST /predict
Request:
```
{
  "Pregnancies": 1,
  "Glucose": 85,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 80,
  "BMI": 22.5,
  "DiabetesPedigreeFunction": 0.25,
  "Age": 29
}
```
Response:
```
{
  "prediction": 0,
  "probability": 0.028852,
  "result": "No Diabetes"
}
```

3. Model Metrics
GET /metrics
Response:
```
{
  "accuracy": 0.79,
  "precision": 0.75,
  "recall": 0.70,
  "f1_score": 0.72
}
```

🔹 Installation & Setup
1. Clone Repository

```
git clone https://github.com/yourusername/diabetes-detection.git
cd diabetes-detection
```

2. Backend Setup

```
cd backend
pip install -r requirements.txt
```

3. Run API
```
uvicorn main:app --reload

```

* Swagger UI: http://127.0.0.1:8000/docs

* ReDoc UI: http://127.0.0.1:8000/redoc


4. Frontend Setup

* Open frontend/index.html in a browser.

* Enter patient data and click Predict.

* Prediction and probability will display on the page.

🔹 Technologies Used

* Backend: Python, FastAPI, Pydantic, NumPy, Joblib
* Machine Learning: Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
* Frontend: HTML, CSS, JavaScript
* Deployment Ready: Docker compatible, CORS enabled

🔹 Author

Samriddha Chakraborty
Full-stack ML Developer | Python | FastAPI | Machine Learning

GitHub: https://github.com/Samriddha03