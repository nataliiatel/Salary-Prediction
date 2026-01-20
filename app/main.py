from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from fastapi import HTTPException
import joblib
from pathlib import Path
import numpy as np

app = FastAPI(title='Career Prediction')


class CareerInput(BaseModel):
    student_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    high_school_gpa: Optional[float] = None
    sat_score: Optional[int] = None
    university_gpa: Optional[float] = None
    field_of_study: Optional[str] = None
    internships_completed: Optional[int] = None
    projects_completed: Optional[int] = None
    certifications: Optional[int] = None
    soft_skills_score: Optional[int] = None
    networking_score: Optional[int] = None
    job_offers: Optional[int] = None
    starting_salary: Optional[float] = None
    years_to_promotion: Optional[int] = None
    current_job_level: Optional[str] = None
    work_life_balance: Optional[int] = None
    entrepreneurship: Optional[str] = None
    target: Optional[int] = None


BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / 'models' / 'model.joblib'
METADATA_PATH = BASE / 'models' / 'metadata.json'


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


model = load_model()
# load metadata if available (feature names)
metadata = {}
if METADATA_PATH.exists():
    try:
        import json
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    except Exception:
        metadata = {}


@app.post('/predict')
def predict(payload: CareerInput):
    global model
    if model is None:
        return {'error': 'Model not found. Train model first.'}
    # prepare input dict from payload
    in_dict = payload.dict()

    try:
        # If metadata.feature_names is available, build a full row aligned to training features
        if metadata and 'feature_names' in metadata:
            feature_names = metadata['feature_names']
            # mapping lowercase -> actual feature name
            lower_map = {fn.lower(): fn for fn in feature_names}
            # start row with NaNs
            row = {fn: np.nan for fn in feature_names}
            # map incoming keys case-insensitively to feature names
            for k, v in in_dict.items():
                ak = k.lower()
                if ak in lower_map:
                    row[lower_map[ak]] = v
                else:
                    # try direct match
                    if k in row:
                        row[k] = v
            df = pd.DataFrame([row])
        else:
            # no metadata available — use the full payload keys as provided
            # Pydantic will have validated/converted types where possible
            df = pd.DataFrame([in_dict])

        pred = model.predict(df)[0]
        prob = None
        try:
            prob = float(model.predict_proba(df)[0].max())
        except Exception:
            prob = None

        return {'prediction': int(pred), 'probability': prob}
    except Exception as e:
        # return HTTP 500 with error details
        raise HTTPException(status_code=500, detail=str(e))
