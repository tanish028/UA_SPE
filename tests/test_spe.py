"""SPE / UA-SPE: formulas, allocation, sampling and the training loop."""
import math

import numpy as np
import pytest

from uaspe import baselines, spe
from uaspe.spe import (allocate_samples, assign_hardness_bins, bin_weights, normalized_uncertainty,
                       predict_spe, select_within_bin, self_paced_alpha, train_spe, within_bin_weights)

BIN_ROWS = np.arange(100, 150)       # 50 row ids in one hardness bin
U_TILDE = np.linspace(0.1, 1.0, 50)


# hardness bins, alpha, bin weights and U~
def test_hardness_bin_boundaries():
    hardness = np.array([0.0, 0.049, 0.05, 0.55, 0.95, 0.999, 1.0])
    assert list(assign_hardness_bins(hardness)) == [0, 0, 1, 11, 19, 19, 19]   # H = 1 -> last bin
    assert list(assign_hardness_bins(np.arange(20) / 20)) == list(range(20))  # every l/20 starts bin l


def test_self_paced_alpha():
    assert self_paced_alpha(1, 10) == pytest.approx(math.tan(math.pi / 20))
    assert self_paced_alpha(9, 10) == pytest.approx(math.tan(9 * math.pi / 20))


def test_bin_weights_use_mean_hardness_and_ignore_empty_bins():
    hardness = np.array([0.01, 0.03, 0.52])
    weights = bin_weights(hardness, assign_hardness_bins(hardness), alpha=0.5)
    assert weights[0] == pytest.approx(1 / (0.02 + 0.5))
    assert weights[10] == pytest.approx(1 / (0.52 + 0.5))
    assert np.count_nonzero(weights) == 2


def test_normalized_uncertainty_matches_hand_calculation():
    # five example rows (columns), four classifiers each
    history = np.array([[0.05, 0.45, 0.5, 0.10, 0.90],
                        [0.06, 0.48, 0.5, 0.90, 0.92],
                        [0.04, 0.46, 0.5, 0.15, 0.91],
                        [0.05, 0.47, 0.5, 0.85, 0.89]])
    expected = [0.9989, 0.9995, 1.0, 0.4350, 0.9985]   # U / (H(1-H)), worked out by hand
    assert normalized_uncertainty(history) == pytest.approx(expected, abs=1e-4)


def test_normalized_uncertainty_zero_denominator_is_zero():
    # H = 0 and H = 1 give H(1-H) = 0, and we set U~ = 0 for these rows
    assert list(normalized_uncertainty([[0.0, 1.0], [0.0, 1.0]])) == [0.0, 0.0]


# sharing the budget between bins
def test_allocation_by_hand():
    # ideal = 7 * [0, .5, .25, .25] = [0, 3.5, 1.75, 1.75], rounded down [0, 3, 1, 1]
    # the 2 rows left over go to the largest fractional parts (bins 2 and 3)
    allocation, redistributions = allocate_samples(np.array([0, 2.0, 1, 1]), np.array([0, 10, 10, 10]), 7)
    assert list(allocation) == [0, 3, 2, 2] and redistributions == 0


def test_allocation_ties_go_to_lower_bin_index():
    allocation, _ = allocate_samples(np.array([1.0, 1, 1]), np.array([9, 9, 9]), 2)
    assert list(allocation) == [1, 1, 0]


def test_allocation_redistributes_when_a_bin_is_too_small():
    # ideal [2, 2, 2, 2] but bin 0 has only 1 row, so its missing row goes to
    # bins 1-3 (equal weights, so the tie goes to bin 1)
    allocation, redistributions = allocate_samples(np.array([1.0, 1, 1, 1]), np.array([1, 10, 10, 10]), 8)
    assert list(allocation) == [1, 3, 2, 2] and redistributions == 1


# selecting rows inside a bin
def test_gamma_zero_selects_exactly_what_original_spe_selects():
    # passing equal weights instead would not: numpy's weighted draw uses the random
    # stream differently
    original_spe = select_within_bin(BIN_ROWS, 12, np.random.default_rng(3))
    ua_spe_gamma0 = select_within_bin(BIN_ROWS, 12, np.random.default_rng(3),
                                      weights=within_bin_weights(U_TILDE, gamma=0))
    assert within_bin_weights(U_TILDE, gamma=0) is None
    assert np.array_equal(original_spe, ua_spe_gamma0)


def test_selection_is_without_replacement():
    assert sorted(select_within_bin(BIN_ROWS, 50, np.random.default_rng(0))) == list(BIN_ROWS)
    weighted = select_within_bin(BIN_ROWS, 40, np.random.default_rng(0), weights=U_TILDE ** 2)
    assert len(set(weighted)) == 40 and set(weighted) <= set(BIN_ROWS)


def test_all_zero_weights_fall_back_to_uniform():
    fallback = select_within_bin(BIN_ROWS, 5, np.random.default_rng(9), weights=np.zeros(50))
    uniform = select_within_bin(BIN_ROWS, 5, np.random.default_rng(9))
    assert np.array_equal(fallback, uniform)


def test_zero_weight_rows_are_used_only_when_positive_rows_run_out():
    rows = np.arange(10)
    weights = np.array([1.0, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert set(select_within_bin(rows, 3, np.random.default_rng(1), weights)) <= {0, 1, 2, 3, 4}
    chosen = select_within_bin(rows, 7, np.random.default_rng(1), weights)
    assert {0, 1, 2, 3, 4} <= set(chosen) and len(set(chosen)) == 7


def test_weighted_selection_is_successive_sampling():
    # weights [3, 1, 1], choose 2. successive sampling gives inclusion probabilities
    # P(row 0) = 3/5 + 2 * (1/5)(3/4) = 0.90 and P(row 1) = 1/5 + (3/5)(1/2) + (1/5)(1/4) = 0.55,
    # which are not proportional to the weights
    rng = np.random.default_rng(0)
    counts = np.zeros(3)
    trials = 20000
    for _ in range(trials):
        counts[select_within_bin(np.arange(3), 2, rng, np.array([3.0, 1, 1]))] += 1
    assert counts[0] / trials == pytest.approx(0.90, abs=0.01)
    assert counts[1] / trials == pytest.approx(0.55, abs=0.015)


def test_requesting_more_rows_than_the_bin_has_is_an_error():
    with pytest.raises(ValueError):
        select_within_bin(BIN_ROWS, 51, np.random.default_rng(0))


# training loop, using a fake classifier so the tests are fast
class FakeClassifier:
    """Stand-in for HistGradientBoostingClassifier: records what it was fitted on and
    predicts a probability that depends on its training subset."""
    fitted = []

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        FakeClassifier.fitted.append((X.copy(), y.copy()))
        self.centre = X[y == 0, 0].mean()
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(X[:, 0] - self.centre)))
        return np.column_stack([1 - p, p])


def small_training_set(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(600, 3))
    y = np.zeros(600, dtype=int)
    y[:30] = 1
    X[:30, 0] += 2.0
    return X, y


@pytest.fixture
def fake_classifier(monkeypatch):
    FakeClassifier.fitted = []
    monkeypatch.setattr(spe, "HistGradientBoostingClassifier", FakeClassifier)
    return FakeClassifier


def test_every_classifier_gets_all_minority_rows_and_equal_majority(fake_classifier):
    X, y = small_training_set()
    classifiers, log, subsets = train_spe(X, y, gamma=2, n=5)
    assert len(classifiers) == 5 and len(fake_classifier.fitted) == 5
    assert len(log) == 4                      # one selection step between consecutive classifiers
    training_rows = {tuple(row) for row in X}
    for X_fit, y_fit in fake_classifier.fitted:
        assert (y_fit == 1).sum() == 30 and (y_fit == 0).sum() == 30
        assert {tuple(r) for r in X_fit[y_fit == 1]} == {tuple(r) for r in X[:30]}
        assert len({tuple(r) for r in X_fit}) == 60          # no row twice
        assert {tuple(r) for r in X_fit} <= training_rows    # only rows we passed in


def test_gamma_zero_reproduces_original_spe_subsets(fake_classifier):
    X, y = small_training_set()
    train_spe(X, y, gamma=None, n=5)
    original = list(fake_classifier.fitted)
    fake_classifier.fitted = []
    train_spe(X, y, gamma=0, n=5)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(original, fake_classifier.fitted))


def test_sampling_seed_controls_the_subsets(fake_classifier):
    X, y = small_training_set()
    _, _, first = train_spe(X, y, gamma=2, n=4, seed=0)
    _, _, again = train_spe(X, y, gamma=2, n=4, seed=0)
    _, _, other = train_spe(X, y, gamma=2, n=4, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not np.array_equal(first[0], other[0])


def test_gamma_two_changes_the_subsets(fake_classifier):
    X, y = small_training_set()
    train_spe(X, y, gamma=None, n=5)
    original = list(fake_classifier.fitted)
    fake_classifier.fitted = []
    train_spe(X, y, gamma=2, n=5)
    assert np.array_equal(original[0][0], fake_classifier.fitted[0][0])   # step 1 is shared
    assert not all(np.array_equal(a[0], b[0]) for a, b in zip(original, fake_classifier.fitted))


def test_ensemble_probability_is_the_mean_of_the_classifiers():
    class Constant:
        def __init__(self, p):
            self.p = p

        def predict_proba(self, X):
            return np.column_stack([1 - np.full(len(X), self.p), np.full(len(X), self.p)])

    X = np.zeros((3, 2))
    assert predict_spe([Constant(0.2), Constant(0.5), Constant(0.8)], X) == pytest.approx([0.5] * 3)


# training loop with the real classifier on a small dataset
def test_train_spe_runs_with_the_real_classifier():
    X, y = small_training_set()
    classifiers, log, subsets = train_spe(X, y, gamma=2, n=3)
    probabilities = predict_spe(classifiers, X)
    assert len(classifiers) == 3 and probabilities.shape == (600,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert sum(b["allocation"] for b in log[0]["bins"]) == 30
    assert len(subsets) == 3 and all(len(s) == 30 and len(set(s)) == 30 for s in subsets)
    assert all(y[s].sum() == 0 for s in subsets)          # subsets contain only majority rows


# baselines
def test_random_undersampling_baselines(fake_classifier, monkeypatch):
    monkeypatch.setattr(baselines, "HistGradientBoostingClassifier", fake_classifier)
    X, y = small_training_set()
    assert len(baselines.train_random_undersampling(X, y, n_classifiers=10, seed=0)) == 10
    for X_fit, y_fit in fake_classifier.fitted:
        assert (y_fit == 1).sum() == 30 and (y_fit == 0).sum() == 30
        assert len({tuple(r) for r in X_fit}) == 60
    first, second = fake_classifier.fitted[0][0], fake_classifier.fitted[1][0]
    assert not np.array_equal(first, second)                 # independent draw per classifier


def test_class_weighted_baseline_uses_all_training_rows(fake_classifier, monkeypatch):
    monkeypatch.setattr(baselines, "HistGradientBoostingClassifier", fake_classifier)
    X, y = small_training_set()
    assert len(baselines.train_class_weighted(X, y)) == 1
    assert len(fake_classifier.fitted[0][1]) == 600


def test_ordinary_baseline_uses_all_training_rows_unchanged(fake_classifier, monkeypatch):
    monkeypatch.setattr(baselines, "HistGradientBoostingClassifier", fake_classifier)
    X, y = small_training_set()
    assert len(baselines.train_ordinary(X, y)) == 1
    X_fit, y_fit = fake_classifier.fitted[0]
    assert np.array_equal(X_fit, X) and np.array_equal(y_fit, y)


def test_random_oversampling_copies_minority_rows_until_balanced():
    X, y = small_training_set()
    X_new, y_new = baselines.random_oversample(X, y, np.random.default_rng(0))
    assert (y_new == 1).sum() == (y_new == 0).sum() == 570
    assert np.array_equal(X_new[:600], X) and np.array_equal(y_new[:600], y)
    minority = {tuple(row) for row in X[y == 1]}
    assert all(tuple(row) in minority for row in X_new[600:])


def test_smote_rows_lie_between_a_minority_row_and_one_of_its_neighbours():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0],
                  [5.0, 5.0], [6.0, 5.0], [7.0, 5.0], [8.0, 5.0], [9.0, 5.0], [9.5, 5.0]])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    X_new, y_new = baselines.smote(X, y, np.random.default_rng(0), k=1)
    synthetic = X_new[10:]
    assert len(synthetic) == 2 and (y_new[10:] == 1).all()
    assert np.array_equal(X_new[:10], X)
    # with k = 1 the neighbour pairs are (0, 1) and (10, 11), so new rows stay on y = 0
    # and inside one of the two segments
    assert np.all(synthetic[:, 1] == 0)
    assert all(0 <= x <= 1 or 10 <= x <= 11 for x in synthetic[:, 0])


def test_smote_is_reproducible_and_balances_classes():
    X, y = small_training_set()
    first = baselines.smote(X, y, np.random.default_rng(3))
    second = baselines.smote(X, y, np.random.default_rng(3))
    assert np.array_equal(first[0], second[0])
    assert (first[1] == 1).sum() == (first[1] == 0).sum() == 570


def test_oversampled_baselines_train_one_classifier_on_balanced_data(fake_classifier, monkeypatch):
    monkeypatch.setattr(baselines, "HistGradientBoostingClassifier", fake_classifier)
    X, y = small_training_set()
    for method in ("random", "smote"):
        assert len(baselines.train_oversampled(X, y, seed=0, method=method)) == 1
    for X_fit, y_fit in fake_classifier.fitted:
        assert len(y_fit) == 1140 and (y_fit == 1).sum() == 570
