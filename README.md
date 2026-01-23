# Salary Prediction

A machine learning project that predicts career success and salary outcomes based on student academic performance, experience, and soft skills.

## Overview

This project uses a **RandomForestRegressor** model to predict career outcomes (salary/success metric) based on 19 features including:
- Academic metrics (GPA, SAT scores)
- Experience (internships, projects, certifications)
- Soft skills and networking scores
- Career information (job offers, satisfaction, work-life balance)

### Model Performance

**Random Forest (Selected Model):**
- **R² Score**: 0.9923 (99.23% variance explained)
- **MAE**: $1,520.18
- **RMSE**: $2,690.95
- **MAPE**: 2.30% (average error)
- **Test Set R²**: 0.9923

**Cross-Validation Performance (5-Fold):**
- **Train R² Mean**: 0.9983 ± 0.0005 (99.83%)
- **Validation R² Mean**: 0.9915 ± 0.0141 (99.15%)
- **Overfitting Gap**: 0.69% ✅ Minimal (< 1%)
- **Conclusion**: Excellent generalization, model will perform well on new data

**Residual Analysis:**
- **Mean Residual**: -$777.64 (slight underprediction bias)
- **Std Residual**: $2,592.39 (prediction variability)

**Comparison with Linear Regression:**
| Metric | Linear Regression | Random Forest | Improvement |
|--------|------------------|---------------|-------------|
| R² Score | 0.9760 | 0.9923 | +1.63% |
| MAE | $3,806.86 | $1,520.18 | 60% better |
| RMSE | $4,753.69 | $2,690.95 | 43% better |
| MAPE | 5.10% | 2.30% | 55% better |
| Overfitting Gap | 0.38% | 0.69% | Comparable |

**Why Random Forest wins:**
- Captures non-linear relationships in student success factors
- Better handles feature interactions (GPA + internships, skills + networking, etc.)
- Consistent performance across different data samples
- Minimal overfitting despite higher complexity

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

### 3. Compare Models (Optional)

```bash
PYTHONPATH=. python src/compare_models.py
```

Compares Linear Regression vs Random Forest and generates:
- `models/comparison_results.json` - detailed comparison metrics
- `models/model_comparison.png` - visualization comparing both models

### 4. Evaluate for Overfitting

```bash
PYTHONPATH=. python src/evaluate_overfit.py
```

Generates:
- `models/learning_curve.png` - learning curves
- `models/overfit_summary.json` - cross-validation results

### 5. Start API Server

```bash
PYTHONPATH=. python -m uvicorn app.main:app --reload --port 8001
```

Server runs on: `http://127.0.0.1:8001`
- Interactive docs: `http://127.0.0.1:8001/docs`
- ReDoc: `http://127.0.0.1:8001/redoc`

### 6. Make Predictions

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
│   ├── train.py              # Random Forest model training
│   ├── compare_models.py     # Linear Regression vs Random Forest comparison
│   ├── evaluate_overfit.py   # Overfitting evaluation with cross-validation
│   └── data_loader.py        # Data loading utilities
├── app/
│   └── main.py               # FastAPI application for predictions
├── data/
│   └── education_career_success_satisfaction.csv  # Training dataset (400+ records)
├── models/                   # Generated after training
│   ├── model.joblib          # Trained Random Forest model
│   ├── metrics.json          # Training metrics
│   ├── comparison_results.json  # Model comparison results
│   ├── predictions.png       # Prediction visualization
│   ├── model_comparison.png  # Side-by-side model comparison
│   ├── learning_curve.png    # Learning curves for overfitting analysis
│   └── metadata.json         # Feature names for API
├── requirements.txt
└── README.md
```
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
- **scikit-learn** - Machine learning (RandomForest, LinearRegression)
- **FastAPI** - REST API framework
- **Pandas, NumPy** - Data processing and analysis
- **Matplotlib** - Data visualization
- **Joblib** - Model serialization

## Model Insights

### Overfitting Analysis Results

**5-Fold Cross-Validation Summary:**

| Model | Train R² | Val R² | Gap | Status |
|-------|----------|--------|-----|--------|
| **Random Forest** | 0.9983 | 0.9915 | 0.69% | ✅ Excellent |
| **Linear Regression** | 0.9749 | 0.9711 | 0.38% | ✅ Good |

**Key Findings:**
- **Overfitting Gap < 1%**: Model generalizes excellently to unseen data
- **Stable across folds**: Standard deviation in validation scores (1.41%) shows consistent performance
- **No signs of overfitting**: High validation R² indicates the model learns genuine patterns, not noise
- **Production-ready**: Safe to deploy with high confidence in predictions

**Learning Curve Interpretation:**
- Training error decreases as data increases
- Validation error converges with training error
- No divergence = no overfitting
- Plateau indicates sufficient data available

**Residual Patterns:**
- Residuals centered near zero (slight -$777 bias)
- Residuals normally distributed
- No systematic patterns (good for predictions)
- Some heteroscedasticity but acceptable

### Random Forest Advantages
- **Non-linear modeling**: Captures complex relationships between student attributes
- **Feature interactions**: Understands how GPA + internships + skills work together
- **Robustness**: Handles outliers and unusual patterns better
- **No scaling required**: Tree-based models are scale-invariant
- **Feature importance**: Can identify which factors matter most for career success

### Why Linear Regression Underperforms
- **Assumes linear relationships**: Student success has complex non-linear patterns
- **Cannot capture interactions**: Treats features independently
- **Sensitive to scaling**: Requires standardization (though included in pipeline)
- **Limited by data structure**: Assumes simple additive effects

### Cross-Validation Results
- **Random Forest**: Train R² 0.9983, Val R² 0.9915 (0.69% gap - minimal overfitting)
- **Linear Regression**: Train R² 0.9749, Val R² 0.9711 (0.38% gap - more stable but lower accuracy)
