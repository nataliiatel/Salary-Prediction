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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import json
from src.data_loader import load_data
import matplotlib.pyplot as plt
import numpy as np


def build_linear_pipeline(df: pd.DataFrame):
    """Build a pipeline with LinearRegression"""
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
                          ('reg', LinearRegression())])
    return reg


def build_rf_pipeline(df: pd.DataFrame):
    """Build a pipeline with RandomForestRegressor"""
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


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and return metrics"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    residuals = y_test - y_pred
    mean_residual = float(residuals.mean())
    std_residual = float(residuals.std())
    
    metrics = {
        'model': model_name,
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2_score': float(r2),
        'mean_residual': mean_residual,
        'std_residual': std_residual
    }
    
    return metrics, y_pred, residuals


def main():
    parser = argparse.ArgumentParser(description='Compare Linear Regression and Random Forest models')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV data file')
    parser.add_argument('--target', type=str, default=None, help='Name of the target column in the CSV')
    parser.add_argument('--cv', type=int, default=5, help='Number of CV folds')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    preferred = base / 'data' / 'education_career_success_satisfaction.csv'

    models_dir = base / 'models'
    models_dir.mkdir(exist_ok=True)

    data_path = args.data
    if data_path is None:
        data_path = str(preferred) if preferred.exists() else None

    if data_path is not None and Path(data_path).exists():
        print('Using dataset:', data_path)
        X, y = load_data(str(data_path), target_cols=args.target)
    else:
        raise SystemExit('No data file found')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('=' * 70)
    print('MODEL COMPARISON: LINEAR REGRESSION vs RANDOM FOREST')
    print('=' * 70)

    # ===== LINEAR REGRESSION =====
    print('\n1. Training Linear Regression Model...')
    lr_model = build_linear_pipeline(X)
    lr_model.fit(X_train, y_train)
    lr_metrics, lr_pred, lr_residuals = evaluate_model(lr_model, X_test, y_test, 'Linear Regression')

    # Cross-validation for Linear Regression
    print('   Running cross-validation...')
    lr_cv = cross_validate(lr_model, X, y, cv=args.cv, return_train_score=True, scoring='r2', n_jobs=-1)
    lr_train_mean = np.mean(lr_cv['train_score'])
    lr_val_mean = np.mean(lr_cv['test_score'])

    # ===== RANDOM FOREST =====
    print('\n2. Training Random Forest Model...')
    rf_model = build_rf_pipeline(X)
    rf_model.fit(X_train, y_train)
    rf_metrics, rf_pred, rf_residuals = evaluate_model(rf_model, X_test, y_test, 'Random Forest')

    # Cross-validation for Random Forest
    print('   Running cross-validation...')
    rf_cv = cross_validate(rf_model, X, y, cv=args.cv, return_train_score=True, scoring='r2', n_jobs=-1)
    rf_train_mean = np.mean(rf_cv['train_score'])
    rf_val_mean = np.mean(rf_cv['test_score'])

    # ===== RESULTS =====
    print('\n' + '=' * 70)
    print('TEST SET PERFORMANCE COMPARISON')
    print('=' * 70)
    
    comparison_data = {
        'Metric': ['R² Score', 'MAE', 'RMSE', 'MAPE', 'Mean Residual', 'Std Residual'],
        'Linear Regression': [
            f"{lr_metrics['r2_score']:.4f}",
            f"${lr_metrics['mae']:.2f}",
            f"${lr_metrics['rmse']:.2f}",
            f"{lr_metrics['mape']*100:.2f}%",
            f"${lr_metrics['mean_residual']:.2f}",
            f"${lr_metrics['std_residual']:.2f}"
        ],
        'Random Forest': [
            f"{rf_metrics['r2_score']:.4f}",
            f"${rf_metrics['mae']:.2f}",
            f"${rf_metrics['rmse']:.2f}",
            f"{rf_metrics['mape']*100:.2f}%",
            f"${rf_metrics['mean_residual']:.2f}",
            f"${rf_metrics['std_residual']:.2f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    print('\n' + '=' * 70)
    print('CROSS-VALIDATION RESULTS (5-Fold)')
    print('=' * 70)
    
    cv_data = {
        'Model': ['Linear Regression', 'Random Forest'],
        'Train R² Mean': [f"{lr_train_mean:.4f}", f"{rf_train_mean:.4f}"],
        'Validation R² Mean': [f"{lr_val_mean:.4f}", f"{rf_val_mean:.4f}"],
        'Overfitting Gap': [f"{(lr_train_mean - lr_val_mean):.4f}", f"{(rf_train_mean - rf_val_mean):.4f}"]
    }
    
    cv_df = pd.DataFrame(cv_data)
    print(cv_df.to_string(index=False))

    # ===== VISUALIZATION =====
    print('\n3. Generating comparison visualizations...')
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Linear Regression - Actual vs Predicted
    axes[0, 0].scatter(y_test, lr_pred, alpha=0.6, edgecolors='k', color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'Linear Regression: Actual vs Predicted\n(R² = {lr_metrics["r2_score"]:.4f})')
    axes[0, 0].grid(alpha=0.3)
    
    # Random Forest - Actual vs Predicted
    axes[0, 1].scatter(y_test, rf_pred, alpha=0.6, edgecolors='k', color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title(f'Random Forest: Actual vs Predicted\n(R² = {rf_metrics["r2_score"]:.4f})')
    axes[0, 1].grid(alpha=0.3)
    
    # Residuals Distribution Comparison
    axes[0, 2].hist(lr_residuals, bins=15, alpha=0.6, label='Linear Regression', color='blue', edgecolor='k')
    axes[0, 2].hist(rf_residuals, bins=15, alpha=0.6, label='Random Forest', color='green', edgecolor='k')
    axes[0, 2].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Residuals Distribution Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Linear Regression - Residuals vs Predicted
    axes[1, 0].scatter(lr_pred, lr_residuals, alpha=0.6, edgecolors='k', color='blue')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Linear Regression: Residuals vs Predicted')
    axes[1, 0].grid(alpha=0.3)
    
    # Random Forest - Residuals vs Predicted
    axes[1, 1].scatter(rf_pred, rf_residuals, alpha=0.6, edgecolors='k', color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Random Forest: Residuals vs Predicted')
    axes[1, 1].grid(alpha=0.3)
    
    # R² Comparison Bar Chart
    models = ['Linear\nRegression', 'Random\nForest']
    r2_scores = [lr_metrics['r2_score'], rf_metrics['r2_score']]
    colors = ['blue', 'green']
    bars = axes[1, 2].bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='k')
    axes[1, 2].set_ylabel('R² Score')
    axes[1, 2].set_title('R² Score Comparison')
    axes[1, 2].set_ylim([0.98, 1.0])
    axes[1, 2].grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.4f}',
                       ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    comparison_path = models_dir / 'model_comparison.png'
    plt.savefig(comparison_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f'   Saved to {comparison_path}')

    # ===== SAVE COMPARISON RESULTS =====
    comparison_results = {
        'linear_regression': lr_metrics,
        'random_forest': rf_metrics,
        'cross_validation': {
            'linear_regression': {
                'train_r2_mean': float(lr_train_mean),
                'val_r2_mean': float(lr_val_mean),
                'overfitting_gap': float(lr_train_mean - lr_val_mean)
            },
            'random_forest': {
                'train_r2_mean': float(rf_train_mean),
                'val_r2_mean': float(rf_val_mean),
                'overfitting_gap': float(rf_train_mean - rf_val_mean)
            }
        },
        'winner': 'Random Forest' if rf_metrics['r2_score'] > lr_metrics['r2_score'] else 'Linear Regression',
        'winner_r2': max(rf_metrics['r2_score'], lr_metrics['r2_score'])
    }

    with open(models_dir / 'comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)

    print('\n' + '=' * 70)
    print(f'WINNER: {comparison_results["winner"]} (R² = {comparison_results["winner_r2"]:.4f})')
    print('=' * 70)
    print(f'\nResults saved to {models_dir / "comparison_results.json"}')
    print(f'Visualization saved to {comparison_path}')


if __name__ == '__main__':
    main()
