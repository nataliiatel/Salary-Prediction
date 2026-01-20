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
curl -sS -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"age":30, "education":"Bachelors", "years_experience":5, "skills":"python;sql"}'
```

Files created by scaffold:
- `src/` - code
- `data/sample.csv` - example dataset
- `models/model.joblib` - produced after training
- `app/main.py` - FastAPI app
