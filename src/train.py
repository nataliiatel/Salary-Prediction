import os
from pathlib import Path
import joblib
import pandas as pd
import argparse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
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

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
    return clf


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

    # split into train / test
    strat = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    # build pipeline based on X (no target column present)
    clf = build_pipeline(X)
    clf.fit(X_train, y_train)

    # evaluate
    y_pred = clf.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    # if binary and predict_proba available, compute ROC AUC
    try:
        if hasattr(clf, 'predict_proba') and y.nunique() == 2:
            probs = clf.predict_proba(X_test)
            # take probability for positive class (assume class order)
            pos_proba = probs[:, 1]
            metrics['roc_auc'] = float(roc_auc_score(y_test, pos_proba))

            # Plot ROC curve
            try:
                fpr, tpr, _ = roc_curve(y_test, pos_proba)
                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
                roc_path = models_dir / 'roc_auc.png'
                plt.savefig(roc_path, bbox_inches='tight')
                plt.close()
                metrics['roc_plot'] = str(roc_path)
            except Exception:
                # plotting failed; continue without breaking training
                pass
    except Exception:
        pass

    # save model and metrics
    joblib.dump(clf, model_path)
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
