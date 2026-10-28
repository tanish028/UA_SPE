"""Data pipeline: load -> check -> remove exact duplicates -> split.

P (minority) = label 1 (fraud); N (majority) = label 0 (genuine).
Every row keeps its original CSV row number as `row_id` (the DataFrame index), so any
training or test row can be traced back to the raw file.

Duplicates are removed and the split is made using only the raw table and its labels.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from uaspe import settings


def load_raw_data(path=settings.DATA_PATH,
                  feature_columns=settings.FEATURE_COLUMNS,
                  target_column=settings.TARGET_COLUMN):
    """Read and check the CSV; rows are indexed by their original position (row_id)."""
    df = pd.read_csv(path)
    expected = list(feature_columns) + [target_column]

    # check the columns
    missing = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in expected]
    if missing or unexpected:
        raise ValueError(f"Column mismatch. Missing: {missing}. Unexpected: {unexpected}.")

    # check for missing values and non-numeric features
    n_missing_values = int(df[expected].isna().sum().sum())
    if n_missing_values > 0:
        raise ValueError(f"{n_missing_values} missing values found; no imputation rule is defined.")

    non_numeric = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}.")

    # labels: exactly 0 and 1, and 1 must be the minority class
    labels = set(df[target_column].unique())
    if labels != {settings.MAJORITY_LABEL, settings.MINORITY_LABEL}:
        raise ValueError(f"Target must contain exactly the labels 0 and 1, found {sorted(labels)}.")

    n_minority = (df[target_column] == settings.MINORITY_LABEL).sum()
    n_majority = (df[target_column] == settings.MAJORITY_LABEL).sum()
    if n_minority >= n_majority:
        raise ValueError("Label 1 must be the minority class (P).")

    df = df[expected]
    df.index = pd.RangeIndex(len(df), name="row_id")
    return df


def class_counts(y):
    """Counts of N and P, and the imbalance ratio IR = |N| / |P|."""
    y = np.asarray(y)
    n_majority = int((y == settings.MAJORITY_LABEL).sum())
    n_minority = int((y == settings.MINORITY_LABEL).sum())

    if n_minority:
        imbalance_ratio = n_majority / n_minority
    else:
        imbalance_ratio = float("inf")

    return {"rows": int(len(y)), "N": n_majority, "P": n_minority,
            "imbalance_ratio": imbalance_ratio}


def remove_exact_duplicates(df, target_column=settings.TARGET_COLUMN):
    """Drop rows identical in every column (features and label), keeping the first.

    Identical rows on both sides of the split would put training rows in the test set.
    Returns the reduced table and the counts before, removed and after.
    """
    # keep the first copy of each duplicate row
    is_repeat = df.duplicated(keep="first")
    removed = df[is_repeat]
    kept = df[~is_repeat]

    summary = {
        "before": class_counts(df[target_column]),
        "removed": {
            "rows": int(is_repeat.sum()),
            "N": int((removed[target_column] == settings.MAJORITY_LABEL).sum()),
            "P": int((removed[target_column] == settings.MINORITY_LABEL).sum()),
        },
        "after": class_counts(kept[target_column]),
    }
    return kept, summary


def stratified_split_indices(y, seed=settings.SPLIT_SEED):
    """Positions (0..len(y)-1) of the train / validation / test rows, each sorted.

    Two stratified steps with the same seed (fractions from settings.py):
      all rows -> train (60%) | holdout (40%)
      holdout  -> validation (20%) | test (20%)
    """
    y = np.asarray(y)
    positions = np.arange(len(y))

    # stratify so every split keeps the same fraud share; the fixed seed gives the
    # same split in every experiment
    holdout_fraction = settings.VAL_FRACTION + settings.TEST_FRACTION

    train_pos, holdout_pos = train_test_split(
        positions,
        test_size=holdout_fraction,
        stratify=y,
        random_state=seed,
    )
    val_pos, test_pos = train_test_split(
        holdout_pos,
        test_size=settings.TEST_FRACTION / holdout_fraction,
        stratify=y[holdout_pos],
        random_state=seed,
    )
    return np.sort(train_pos), np.sort(val_pos), np.sort(test_pos)


def load_and_split(path=settings.DATA_PATH, seed=settings.SPLIT_SEED):
    """Load, remove duplicates and split. Returns a dict with keys:

    X_train, X_val, X_test                     feature DataFrames, indexed by row_id
    y_train, y_val, y_test                     int label arrays (1 = P, 0 = N)
    train_row_ids, val_row_ids, test_row_ids   original row_ids of each split
    deduplication                              counts before, removed and after
    """
    raw = load_raw_data(path)
    data, dedup_summary = remove_exact_duplicates(raw)

    X = data[settings.FEATURE_COLUMNS].astype(np.float64)
    y = data[settings.TARGET_COLUMN].to_numpy(dtype=np.int64)
    row_ids = X.index.to_numpy()

    train_pos, val_pos, test_pos = stratified_split_indices(y, seed)

    split = {
        "X_train": X.iloc[train_pos],
        "y_train": y[train_pos],
        "train_row_ids": row_ids[train_pos],
        "X_val": X.iloc[val_pos],
        "y_val": y[val_pos],
        "val_row_ids": row_ids[val_pos],
        "X_test": X.iloc[test_pos],
        "y_test": y[test_pos],
        "test_row_ids": row_ids[test_pos],
        "deduplication": dedup_summary,
    }
    return split
