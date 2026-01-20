import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate, learning_curve

from src.data_loader import load_data
from src.train import build_pipeline


def main():
    parser = argparse.ArgumentParser(description='Evaluate model for overfitting')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV data file')
    parser.add_argument('--target', type=str, default=None, help='Name of the target column')
    parser.add_argument('--cv', type=int, default=5, help='Number of CV folds')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    preferred = base / 'data' / 'education_career_success_satisfaction.csv'
    models_dir = base / 'models'
    models_dir.mkdir(exist_ok=True)

    data_path = args.data if args.data else (str(preferred) if preferred.exists() else None)
    if data_path is None:
        raise SystemExit('No data file found; pass --data')

    print('Loading data from', data_path)
    X, y = load_data(str(data_path), target_cols=args.target)

    print('Building fresh pipeline for evaluation')
    pipeline = build_pipeline(X)

    print(f'Running cross-validation (cv={args.cv}) with train scores...')
    cv_results = cross_validate(pipeline, X, y, cv=args.cv, return_train_score=True, scoring='accuracy', n_jobs=-1)

    train_scores = cv_results['train_score']
    test_scores = cv_results['test_score']

    print('Train accuracy: mean={:.4f} std={:.4f}'.format(np.mean(train_scores), np.std(train_scores)))
    print('Validation accuracy: mean={:.4f} std={:.4f}'.format(np.mean(test_scores), np.std(test_scores)))

    # Save numeric summary
    summary = {
        'cv': args.cv,
        'train_score_mean': float(np.mean(train_scores)),
        'train_score_std': float(np.std(train_scores)),
        'val_score_mean': float(np.mean(test_scores)),
        'val_score_std': float(np.std(test_scores)),
    }
    with open(models_dir / 'overfit_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Learning curve
    print('Computing learning curve (this may take a moment)')
    train_sizes, train_scores_lc, val_scores_lc = learning_curve(
        pipeline, X, y, cv=args.cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )

    train_scores_mean = np.mean(train_scores_lc, axis=1)
    val_scores_mean = np.mean(val_scores_lc, axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training score')
    plt.plot(train_sizes, val_scores_mean, 'o-', label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    lc_path = models_dir / 'learning_curve.png'
    plt.savefig(lc_path, bbox_inches='tight')
    plt.close()

    print('Learning curve saved to', lc_path)
    print('Summary written to', models_dir / 'overfit_summary.json')


if __name__ == '__main__':
    main()
