# Career Prediction (scaffold)

Minimal scaffold for a career-prediction project.

Quick start (macOS bash):

1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train model

```bash
python -m src.train
```

3. Run API

```bash
uvicorn app.main:app --reload
```

4. Predict (example)

```bash
curl -X POST "http://127.0.0.1:8000/predict"  "   -d '{ent-Type: application/json"
    "student_id": "S649",
    "age": 21,
    "gender": "Male",
    "high_school_gpa": 3.9,
    "sat_score": 1580,
    "university_gpa": 3.9,
    "field_of_study": "Computer Science",
    "internships_completed": 4,
    "projects_completed": 9,
    "certifications": 5,
    "soft_skills_score": 8,
    "networking_score": 9,
    "job_offers": 0,
    "starting_salary": 14800000,
    "years_to_promotion": 1,
    "current_job_level": "Senior",
    "work_life_balance": 5,
    "entrepreneurship": "No"
  }'
```

Files created by scaffold:
- `src/` - code
- `data/sample.csv` - example dataset
- `models/model.joblib` - produced after training
- `app/main.py` - FastAPI app
