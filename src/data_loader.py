import pandas as pd
from typing import Tuple


def load_data(path: str, target_cols=None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV, do minimal cleaning, and return X, y.
    - Detects the target column case-insensitively (e.g. 'Target', 'target').
    - Drops rows with any NA by default.
    - Encodes object columns to category codes.
    """
    df = pd.read_csv(path)
    # handle files that include a title row before the header (one-column read)
    # e.g. first line is a dataset title and second line is the real header
    if len(df.columns) == 1:
        first_col = df.columns[0].strip()
        # if the single column name looks like a title (no commas inside), try skipping first row
        if ' ' in first_col or '_' in first_col:
            try:
                df = pd.read_csv(path, skiprows=1)
            except Exception:
                pass

    # normalize column names to strip whitespace
    df.columns = [c.strip() for c in df.columns]

    def _find_target_in_columns(columns, target_cols_arg=None):
        # search for explicit names or a set of common synonyms
        synonyms = ['target', 'outcome', 'success', 'label', 'y', 'satisfaction']
        if target_cols_arg is None:
            want_list = synonyms
        else:
            if isinstance(target_cols_arg, str):
                want_list = [target_cols_arg]
            else:
                want_list = list(target_cols_arg)

        for want in want_list:
            for c in columns:
                if c and c.lower() == want.lower():
                    return c
        return None

    # first attempt: look in the parsed columns
    target_col = _find_target_in_columns(df.columns, target_cols)

    # If not found, try re-reading by skipping the first row (common when CSV has a title row)
    if target_col is None:
        try:
            df2 = pd.read_csv(path, skiprows=1)
            df2.columns = [c.strip() for c in df2.columns]
            target_col = _find_target_in_columns(df2.columns, target_cols)
            if target_col is not None:
                df = df2
        except Exception:
            # ignore read errors and fall through to raising a ValueError below
            pass

    if target_col is None:
        raise ValueError('CSV must contain a target column (e.g. "Target" or "target").')

    df = df.dropna()
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Simple encoding: convert object columns to category codes
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    return X, y


if __name__ == '__main__':
    import os
    sample = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.csv')
    sample = os.path.normpath(sample)
    try:
        X, y = load_data(sample)
        print('Loaded', X.shape, 'X and', y.shape, 'y')
    except Exception as e:
        print('Could not load sample data:', e)
