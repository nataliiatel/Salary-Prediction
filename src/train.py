import os
from pathlib import Path
import joblib
import pandas as pd
import argparse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import json
from src.data_loader import load_data
import matplotlib.pyplot as plt
from src.data_loader import load_data


def build_pipeline(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ]
    )

    reg = Pipeline(steps=[('preprocessor', preprocessor),
                          ('reg', RandomForestRegressor(n_estimators=100, random_state=42))])
    return reg


def main():
    parser = argparse.ArgumentParser(description='Train career prediction model')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV data file')
    parser.add_argument('--target', type=str, default=None, help='Name of the target column in the CSV')
    parser.add_argument('--debug', action='store_true', help='Dump extra debug output')
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    # prefer the provided dataset if present
    preferred = base / 'data' / 'education_career_success_satisfaction.csv'

    models_dir = base / 'models'
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'model.joblib'

    data_path = args.data
    if data_path is None:
        data_path = str(preferred) if preferred.exists() else None

    if data_path is not None and Path(data_path).exists():
        print('Using dataset:', data_path)
        X, y = load_data(str(data_path), target_cols=args.target)
    else:
        # synthetic data
        df = pd.DataFrame({
            'age': [25,30,22,40,28,35,45,32,26,29],
            'education': ['Bachelors','Masters','HS','PhD','Bachelors','Masters','HS','Bachelors','HS','Masters'],
            'years_experience': [2,5,1,15,4,8,20,6,3,7],
            'skills': ['python;sql','java;sql','excel','research;python','python;excel','java','management','python;java','excel','sql;python'],
            'target': [1,1,0,1,1,0,0,1,0,1]
        })
        X = df.drop(columns=['target'])
        y = df['target']

    # split into train / test (no stratification for regression)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # build pipeline based on X (no target column present)
    reg = build_pipeline(X)
    reg.fit(X_train, y_train)

    # evaluate
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate residuals for additional insights
    residuals = y_test - y_pred
    mean_residual = float(residuals.mean())
    std_residual = float(residuals.std())
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2_score': float(r2),
        'mean_residual': mean_residual,
        'std_residual': std_residual
    }

    # Plot actual vs predicted
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.3f})')
        
        # Residuals vs Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        
        # Residuals Distribution
        axes[1, 0].hist(residuals, bins=20, edgecolor='k', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        
        # Error Distribution
        errors = abs(residuals)
        axes[1, 1].hist(errors, bins=20, edgecolor='k', alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Absolute Errors')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Absolute Error Distribution (MAE = {mae:.2f})')
        
        plt.tight_layout()
        pred_path = models_dir / 'predictions.png'
        plt.savefig(pred_path, bbox_inches='tight', dpi=100)
        plt.close()
        metrics['prediction_plot'] = str(pred_path)
    except Exception:
        # plotting failed; continue without breaking training
        pass

    # save model and metrics
    joblib.dump(reg, model_path)
    # save metadata (feature names) so the API can construct inputs correctly
    metadata = {
        'feature_names': list(X.columns)
    }
    with open(models_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(models_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'Model saved to {model_path}')
    print('Evaluation metrics:')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
