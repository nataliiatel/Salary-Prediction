import os
import shutil
from pathlib import Path

def test_train_creates_model(tmp_path):
    # Run train.py from project
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / 'models'
    model_file = models_dir / 'model.joblib'
    # Remove if exists
    if model_file.exists():
        model_file.unlink()

    # Call train
    import importlib
    train = importlib.import_module('src.train')
    train.main()

    assert model_file.exists()
