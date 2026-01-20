# Career Satisfaction & Salary Prediction

A machine learning project that predicts career success and salary outcomes based on student academic performance, experience, and soft skills.

## Overview

This project uses a **RandomForestRegressor** model to predict career outcomes (salary/success metric) based on 19 features including:
- Academic metrics (GPA, SAT scores)
- Experience (internships, projects, certifications)
- Soft skills and networking scores
- Career information (job offers, satisfaction, work-life balance)

**Model Performance:**
- R² Score: 0.992 (99.2% variance explained)
- MAE: $1,520.18
- RMSE: $2,690.95
- Minimal overfitting (Train R²: 0.998, Val R²: 0.992)

## Quick Start (macOS bash)

### 1. Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Model

```bash
PYTHONPATH=. python src/train.py
```

This generates:
- `models/model.joblib` - trained model
- `models/metrics.json` - evaluation metrics
- `models/predictions.png` - visualization of predictions vs actuals
- `models/metadata.json` - feature names for API

### 3. Evaluate for Overfitting

```bash
PYTHONPATH=. python src/evaluate_overfit.py
```

Generates:
- `models/learning_curve.png` - learning curves
- `models/overfit_summary.json` - cross-validation results

### 4. Start API Server

```bash
PYTHONPATH=. python -m uvicorn app.main:app --reload --port 8001
```

Server runs on: `http://127.0.0.1:8001`
- Interactive docs: `http://127.0.0.1:8001/docs`
- ReDoc: `http://127.0.0.1:8001/redoc`

### 5. Make Predictions

```bash
curl -X POST "http://127.0.0.1:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "S001",
    "age": 22,
    "gender": "Male",
    "high_school_gpa": 3.8,
    "sat_score": 1450,
    "university_gpa": 3.6,
    "field_of_study": "Computer Science",
    "internships_completed": 3,
    "projects_completed": 7,
    "certifications": 2,
    "soft_skills_score": 8,
    "networking_score": 7,
    "job_offers": 3,
    "career_satisfaction": 8,
    "years_to_promotion": 2,
    "current_job_level": "Mid",
    "work_life_balance": 7,
    "entrepreneurship": "No"
  }'
```

Response: `{"prediction": 85000.0}`

## Project Structure

```
├── src/
│   ├── train.py              # Model training script
│   ├── evaluate_overfit.py   # Overfitting evaluation
│   └── data_loader.py        # Data loading utilities
├── app/
│   └── main.py               # FastAPI application
├── data/
│   └── education_career_success_satisfaction.csv  # Training dataset
├── models/                   # Generated after training
│   ├── model.joblib
│   ├── metrics.json
│   ├── predictions.png
│   └── learning_curve.png
├── requirements.txt
└── README.md
```

## Dataset Features

The model uses 19 input features from the education_career_success_satisfaction dataset:
- `student_id`, `age`, `gender`
- `high_school_gpa`, `sat_score`, `university_gpa`, `field_of_study`
- `internships_completed`, `projects_completed`, `certifications`
- `soft_skills_score`, `networking_score`
- `job_offers`, `career_satisfaction`, `years_to_promotion`, `current_job_level`
- `work_life_balance`, `entrepreneurship`
- **Target**: Success metric (continuous value)

## Technologies

- **Python 3.14**
- **scikit-learn** - Machine learning
- **FastAPI** - REST API
- **Pandas, NumPy** - Data processing
- **Matplotlib** - Visualization
- **Joblib** - Model serialization
