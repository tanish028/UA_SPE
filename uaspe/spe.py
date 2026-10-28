"""Original SPE and UA-SPE.

Both use the same training loop in train_spe. The only difference is how rows are
chosen inside each hardness bin: uniformly (SPE) or with weights U~^gamma (UA-SPE).
"""
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from uaspe import settings


def assign_hardness_bins(hardness, k=settings.N_HARDNESS_BINS):
    # bin l (0-based) is [l/k, (l+1)/k); H = 1 goes into the last bin
    bins = np.floor(hardness * k).astype(int)
    return np.minimum(bins, k - 1)


def self_paced_alpha(i, n=settings.N_CLASSIFIERS):
    return math.tan(i * math.pi / (2 * n))


def bin_weights(hardness, bins, alpha, k=settings.N_HARDNESS_BINS):
    """w_l = 1 / (h_l + alpha), where h_l is the mean hardness in bin l. Empty bins get 0."""
    weights = np.zeros(k)

    for l in range(k):
        in_bin = bins == l
        if in_bin.any():
            mean_hardness = hardness[in_bin].mean()
            weights[l] = 1.0 / (mean_hardness + alpha)

    return weights


def allocate_samples(weights, bin_sizes, total):
    """Number of rows to take from each bin, summing exactly to `total`.

    Ideal counts total * w_l / sum(w) are rounded down, and the rows left over go to
    the bins with the largest fractional parts (ties go to the lower bin index).
    A bin cannot give more rows than it has. Any shortfall is shared again, by the
    same rule, over bins that still have spare rows.

    Returns the allocation and the number of redistribution rounds.
    """
    k = len(weights)
    allocation = np.zeros(k, dtype=int)
    redistributions = 0
    remaining = total

    while remaining > 0:
        eligible = (weights > 0) & (allocation < bin_sizes)
        if not eligible.any():
            raise ValueError("Not enough majority rows to allocate.")

        # ideal share of the remaining rows, rounded down
        eligible_weights = np.where(eligible, weights, 0.0)
        ideal = remaining * eligible_weights / eligible_weights.sum()
        extra = np.floor(ideal).astype(int)

        # give the leftover rows to the largest fractional parts, then lower bin index
        leftover = remaining - extra.sum()
        fractional_part = ideal - extra
        order = np.lexsort((np.arange(k), -fractional_part))
        extra[order[:leftover]] += 1

        # no bin can give more rows than it has
        allocation = np.minimum(allocation + extra, bin_sizes)
        remaining = total - allocation.sum()
        if remaining > 0:
            redistributions += 1

    return allocation, redistributions


def normalized_uncertainty(prediction_history):
    """U~ = U / (H(1-H)) for each row; 0 where H(1-H) = 0.

    prediction_history has one row per classifier and one column per majority sample.
    """
    p = np.asarray(prediction_history, dtype=float)

    hardness = p.mean(axis=0)
    raw_uncertainty = (p * (1 - p)).mean(axis=0)
    denominator = hardness * (1 - hardness)

    u_tilde = np.zeros_like(hardness)
    np.divide(raw_uncertainty, denominator, out=u_tilde, where=denominator > 0)
    return u_tilde


def within_bin_weights(normalized_uncertainty, gamma):
    """U~^gamma. gamma = 0 returns None so the caller samples uniformly, exactly as SPE."""
    if gamma == 0:
        return None
    return np.asarray(normalized_uncertainty, dtype=float) ** gamma


def select_within_bin(bin_rows, n_to_select, rng, weights=None):
    """Pick n_to_select rows of one bin without replacement."""
    bin_rows = np.asarray(bin_rows)
    if n_to_select > len(bin_rows):
        raise ValueError("Allocation exceeds bin size; allocate_samples must redistribute first.")

    # uniform selection (SPE, gamma = 0, or a bin where every weight is 0)
    if weights is None or not np.any(weights > 0):
        return rng.choice(bin_rows, size=n_to_select, replace=False)

    # fewer positive-weight rows than needed: take all of them and fill the rest
    # uniformly from the zero-weight rows
    positive = weights > 0
    n_positive = positive.sum()
    if n_positive <= n_to_select:
        rest = rng.choice(bin_rows[~positive], size=n_to_select - n_positive, replace=False)
        return np.concatenate([bin_rows[positive], rest])

    # successive sampling: each draw picks one of the remaining rows with probability
    # proportional to its weight, so the final inclusion probabilities are NOT
    # proportional to the weights
    probabilities = weights / weights.sum()
    return rng.choice(bin_rows, size=n_to_select, replace=False, p=probabilities)


def select_majority_rows(bins, allocation, u_tilde, gamma, rng, k):
    """Positions (within the majority rows) chosen from every bin for the next classifier."""
    chosen = []

    for l in range(k):
        if allocation[l] == 0:
            continue

        in_bin = np.flatnonzero(bins == l)
        if gamma is None:
            row_weights = None
        else:
            row_weights = within_bin_weights(u_tilde[in_bin], gamma)

        chosen.append(select_within_bin(in_bin, allocation[l], rng, row_weights))

    return np.concatenate(chosen)


def train_spe(X_train, y_train, gamma=None, seed=settings.SPE_SEEDS[0],
              n=settings.N_CLASSIFIERS, k=settings.N_HARDNESS_BINS):
    """Train n classifiers. gamma=None is original SPE; gamma = 0, 1, 2, 4 is UA-SPE.

    Returns the classifiers, a per-iteration diagnostics log, and the majority rows
    (positions in X_train) used to train each classifier.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train)
    minority = np.flatnonzero(y == settings.MINORITY_LABEL)
    majority = np.flatnonzero(y == settings.MAJORITY_LABEL)
    rng = np.random.default_rng(seed)

    classifiers = []
    history = []   # predictions of each classifier on every majority row
    log = []
    subsets = []

    # first sample: |P| majority rows drawn uniformly, without replacement so no row
    # is selected twice
    selected_majority = rng.choice(majority, size=len(minority), replace=False)

    for i in range(1, n + 1):
        # train classifier i on all minority rows plus the selected majority rows
        subsets.append(selected_majority)
        rows = np.concatenate([minority, selected_majority])
        classifier = HistGradientBoostingClassifier(**settings.HGB_PARAMS, random_state=seed)
        classifier.fit(X[rows], y[rows])
        classifiers.append(classifier)

        if i == n:
            break

        # calculate hardness for each majority row
        history.append(classifier.predict_proba(X[majority])[:, 1])
        hardness = np.mean(history, axis=0)

        # split rows into bins and share the budget of |P| rows between the bins
        bins = assign_hardness_bins(hardness, k)
        bin_sizes = np.bincount(bins, minlength=k)
        alpha = self_paced_alpha(i, n)
        weights = bin_weights(hardness, bins, alpha, k)
        allocation, redistributions = allocate_samples(weights, bin_sizes, len(minority))

        # select the next sample inside each bin
        if gamma is None:
            u_tilde = None
        else:
            u_tilde = normalized_uncertainty(history)

        chosen = select_majority_rows(bins, allocation, u_tilde, gamma, rng, k)
        selected_majority = majority[chosen]

        log.append(iteration_diagnostics(i, alpha, hardness, bins, bin_sizes, weights,
                                         allocation, redistributions, history, u_tilde, gamma))

    return classifiers, log, subsets


def predict_spe(classifiers, X):
    """Final ensemble probability: the mean of the classifiers' probabilities."""
    X = np.asarray(X, dtype=float)
    probabilities = [classifier.predict_proba(X)[:, 1] for classifier in classifiers]
    return np.mean(probabilities, axis=0)


def iteration_diagnostics(i, alpha, hardness, bins, bin_sizes, weights, allocation,
                          redistributions, history, u_tilde, gamma):
    """Diagnostics for one selection step.

    Selection probabilities p(x) are single-draw probabilities: uniform 1/|B_l| for
    SPE, gamma = 0 and all-zero bins, otherwise U~^gamma normalised within the bin.
    """
    variance = np.var(history, axis=0)

    # distribution of hardness and prediction variance over all majority rows
    summary = {
        "iteration": i,
        "alpha": alpha,
        "occupied_bins": int((bin_sizes > 0).sum()),
        "redistributions": redistributions,
        "hardness_mean": hardness.mean(),
        "hardness_median": np.median(hardness),
        "hardness_p90": np.quantile(hardness, 0.9),
        "hardness_max": hardness.max(),
        "zero_denominator_frac": np.mean(hardness * (1 - hardness) == 0),
        "prediction_variance_mean": variance.mean(),
        "prediction_variance_median": np.median(variance),
        "prediction_variance_max": variance.max(),
    }

    if u_tilde is not None:
        summary.update({
            "u_tilde_mean": u_tilde.mean(),
            "u_tilde_p10": np.quantile(u_tilde, 0.1),
            "u_tilde_median": np.median(u_tilde),
            "u_tilde_p90": np.quantile(u_tilde, 0.9),
            "u_tilde_zero_frac": np.mean(u_tilde == 0),
            "u_tilde_near_one_frac": np.mean(u_tilde >= settings.NEAR_ONE_U_TILDE),
        })

    # per bin: how far the selection probabilities move away from uniform
    bins_log = []
    deviating_rows = 0

    for l in np.flatnonzero(bin_sizes > 0):
        in_bin = bins == l
        uniform = 1.0 / bin_sizes[l]
        p = np.full(bin_sizes[l], uniform)

        if gamma not in (None, 0):
            w = u_tilde[in_bin] ** gamma
            if w.sum() > 0:
                p = w / w.sum()

        relative_deviation = np.abs(p - uniform) / uniform
        is_deviating = relative_deviation > settings.RELATIVE_DEVIATION_THRESHOLD
        if allocation[l] > 0:
            deviating_rows += int(np.sum(is_deviating))

        bins_log.append({
            "iteration": i,
            "bin": int(l),
            "rows": int(bin_sizes[l]),
            "mean_hardness": hardness[in_bin].mean(),
            "weight": weights[l],
            "allocation": int(allocation[l]),
            "u_tilde_mean": np.nan if u_tilde is None else u_tilde[in_bin].mean(),
            "tv_from_uniform": 0.5 * np.abs(p - uniform).sum(),
            "rel_dev_frac": np.mean(is_deviating),
        })

    # averages over the bins that gave rows, weighted by their allocation
    used = [b for b in bins_log if b["allocation"] > 0]
    
    total_allocated = sum(b["allocation"] for b in used)
    weighted_tv = sum(b["allocation"] * b["tv_from_uniform"] for b in used)

    summary["tv_from_uniform_weighted"] = weighted_tv / total_allocated
    summary["rel_dev_frac_rows"] = deviating_rows / sum(b["rows"] for b in used)

    return {"summary": summary, "bins": bins_log}
