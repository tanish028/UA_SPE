"""Baselines with the same base classifier as SPE."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

from uaspe import settings


def train_ordinary(X_train, y_train, seed=0):
    """One classifier on the full, imbalanced training split, without any rebalancing."""
    classifier = HistGradientBoostingClassifier(**settings.HGB_PARAMS, random_state=seed)
    classifier.fit(np.asarray(X_train, dtype=float), np.asarray(y_train))
    return [classifier]


def train_random_undersampling(X_train, y_train, n_classifiers=1, seed=0):
    """Each classifier sees all minority rows plus |P| majority rows drawn uniformly.

    Every classifier gets its own draw. n_classifiers=10 matches SPE's budget.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train)
    minority = np.flatnonzero(y == settings.MINORITY_LABEL)
    majority = np.flatnonzero(y == settings.MAJORITY_LABEL)
    rng = np.random.default_rng(seed)

    classifiers = []
    for _ in range(n_classifiers):
        sampled_majority = rng.choice(majority, size=len(minority), replace=False)
        rows = np.concatenate([minority, sampled_majority])

        classifier = HistGradientBoostingClassifier(**settings.HGB_PARAMS, random_state=seed)
        classifier.fit(X[rows], y[rows])
        classifiers.append(classifier)

    return classifiers


def train_class_weighted(X_train, y_train, seed=0):
    """One classifier on the full training split.

    class_weight="balanced" gives each class the same total weight instead of resampling.
    """
    classifier = HistGradientBoostingClassifier(
        **settings.HGB_PARAMS,
        class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(np.asarray(X_train, dtype=float), np.asarray(y_train))
    return [classifier]


def random_oversample(X, y, rng):
    """Copy random minority rows (with replacement) until both classes have the same size."""
    minority = np.flatnonzero(y == settings.MINORITY_LABEL)
    n_majority = int((y == settings.MAJORITY_LABEL).sum())
    n_new = n_majority - len(minority)

    copies = rng.choice(minority, size=n_new, replace=True)
    return np.vstack([X, X[copies]]), np.concatenate([y, y[copies]])


def smote(X, y, rng, k=5):
    """SMOTE (Chawla et al., 2002).

    Each new minority row lies on the line between a random minority row and one of
    its k nearest minority neighbours. New rows are added until the classes are equal.

    Neighbours are found on standardised features, because Time and Amount are on much
    larger scales than V1-V28. The new rows are built in the original feature space.
    """
    minority_rows = X[y == settings.MINORITY_LABEL]
    n_majority = int((y == settings.MAJORITY_LABEL).sum())
    n_new = n_majority - len(minority_rows)

    # find the k nearest minority neighbours on standardised features
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    scaled = (minority_rows - X.mean(axis=0)) / scale

    nearest = NearestNeighbors(n_neighbors=k + 1).fit(scaled)
    neighbours = nearest.kneighbors(scaled, return_distance=False)
    neighbours = neighbours[:, 1:]   # the nearest point is the row itself

    # build each new row between a random minority row and one of its neighbours
    start = rng.integers(0, len(minority_rows), size=n_new)
    partner = neighbours[start, rng.integers(0, k, size=n_new)]
    gap = rng.random((n_new, 1))
    synthetic = minority_rows[start] + gap * (minority_rows[partner] - minority_rows[start])

    X_new = np.vstack([X, synthetic])
    y_new = np.concatenate([y, np.full(n_new, settings.MINORITY_LABEL)])
    return X_new, y_new


def train_oversampled(X_train, y_train, seed=0, method="smote"):
    """One classifier on the training split after SMOTE or random oversampling.

    Only the training split is oversampled.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train)
    rng = np.random.default_rng(seed)

    if method == "smote":
        X_fit, y_fit = smote(X, y, rng)
    else:
        X_fit, y_fit = random_oversample(X, y, rng)

    classifier = HistGradientBoostingClassifier(**settings.HGB_PARAMS, random_state=seed)
    classifier.fit(X_fit, y_fit)
    return [classifier]
