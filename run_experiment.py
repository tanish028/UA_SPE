"""Train SPE, UA-SPE (gamma = 0, 1, 2, 4) and the six baselines for every sampling seed
on the fixed training split, evaluate them on the test split and save the results.

Run from the repository root:  python run_experiment.py [output folder] [--overwrite]
The default output folder is results/. Existing results are only replaced with --overwrite.
"""
import sys
import time

import numpy as np
import pandas as pd
import sklearn

from uaspe import settings
from uaspe.baselines import (train_class_weighted, train_ordinary, train_oversampled,
                             train_random_undersampling)
from uaspe.data import class_counts, load_and_split
from uaspe.metrics import evaluate
from uaspe.spe import predict_spe, train_spe


# files written by this script
OUTPUT_FILES = [
    "results_per_seed.csv", "results_summary.csv", "control_check.csv",
    "diagnostics_iterations.csv", "diagnostics_bins.csv", "test_probabilities.npz",
    "run_info.txt",
]

# columns averaged over the seeds in results_summary.csv
METRICS = [
    "gmean_recall_specificity", "aucprc_average_precision", "f1", "mcc",
    "precision", "recall", "specificity", "gmean_precision_recall",
    "tp", "fp", "train_seconds",
    "tv_from_uniform_steps2plus", "u_tilde_near_one_frac_steps2plus",
]

# (method name, gamma); gamma = None is the original SPE
SPE_RUNS = [("SPE", None)] + [(f"UA-SPE gamma={g}", g) for g in settings.GAMMA_VALUES]

BASELINES = {
    "Ordinary HGB (no resampling)": train_ordinary,
    "RUS (1 classifier)":
        lambda X, y, seed: train_random_undersampling(X, y, 1, seed),
    "ROS (random oversampling)":
        lambda X, y, seed: train_oversampled(X, y, seed, method="random"),
    "SMOTE":
        lambda X, y, seed: train_oversampled(X, y, seed, method="smote"),
    "RUS ensemble (10 classifiers)":
        lambda X, y, seed: train_random_undersampling(X, y, settings.N_CLASSIFIERS, seed),
    "Class-weighted HGB (full data)": train_class_weighted,
}


def training_rows_per_classifier(method, y_train):
    n_minority = int((y_train == settings.MINORITY_LABEL).sum())

    if method.startswith(("Ordinary", "Class-weighted")):
        return len(y_train)
    if method.startswith(("ROS", "SMOTE")):
        return 2 * (len(y_train) - n_minority)
    return 2 * n_minority


def selection_summary(log):
    """Average selection diagnostics over steps 2 and later.

    Step 1 is skipped: with one classifier U~ = 1 everywhere and selection is uniform.
    """
    later = pd.DataFrame([entry["summary"] for entry in log[1:]])

    summary = {"tv_from_uniform_steps2plus": later["tv_from_uniform_weighted"].mean()}
    if "u_tilde_near_one_frac" in later:
        summary["u_tilde_near_one_frac_steps2plus"] = later["u_tilde_near_one_frac"].mean()
    return summary


def control_check(seed, subsets_by_method, test_probabilities):
    """SPE and UA-SPE gamma=0 must pick the same rows and give the same predictions."""
    spe_subsets = subsets_by_method["SPE"]
    gamma0_subsets = subsets_by_method["UA-SPE gamma=0"]
    same_subsets = all(np.array_equal(a, b) for a, b in zip(spe_subsets, gamma0_subsets))

    same_probabilities = np.array_equal(test_probabilities[f"SPE|{seed}"],
                                        test_probabilities[f"UA-SPE gamma=0|{seed}"])

    return {"seed": seed,
            "subsets_identical": same_subsets,
            "probabilities_identical": same_probabilities}


def save_results(results_dir, split, result_rows, control_rows, iteration_rows, bin_rows,
                 test_probabilities):
    # one row per method and seed
    results = pd.DataFrame(result_rows)
    results.to_csv(results_dir / "results_per_seed.csv", index=False)

    # mean and standard deviation over the seeds (std uses n - 1)
    ok = results[results["status"] == "ok"]
    metrics = [m for m in METRICS if m in ok]
    summary = ok.groupby("method", sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "completed_seeds", ok.groupby("method", sort=False).size())
    summary.to_csv(results_dir / "results_summary.csv")

    # control check, diagnostics and test probabilities
    pd.DataFrame(control_rows).to_csv(results_dir / "control_check.csv", index=False)
    pd.DataFrame(iteration_rows).to_csv(results_dir / "diagnostics_iterations.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(results_dir / "diagnostics_bins.csv", index=False)
    np.savez_compressed(results_dir / "test_probabilities.npz",
                        y_test=split["y_test"], **test_probabilities)

    # settings used for this run
    info = [
        f"scikit-learn {sklearn.__version__} (pinned {settings.SKLEARN_VERSION})",
        f"split seed {settings.SPLIT_SEED}, sampling seeds {settings.SPE_SEEDS}",
        f"n = {settings.N_CLASSIFIERS}, k = {settings.N_HARDNESS_BINS}, gammas = {settings.GAMMA_VALUES}",
        f"HGB parameters: {settings.HGB_PARAMS}",
        f"threshold {settings.CLASSIFICATION_THRESHOLD}",
        f"train {class_counts(split['y_train'])}",
        f"test {class_counts(split['y_test'])}",
        f"failed runs: {int((results['status'] != 'ok').sum())}",
    ]
    (results_dir / "run_info.txt").write_text("\n".join(info) + "\n")


def main(results_name="results", overwrite=False):
    # never replace earlier results by accident
    results_dir = settings.REPO_ROOT / results_name
    existing = [name for name in OUTPUT_FILES if (results_dir / name).exists()]
    if existing and not overwrite:
        sys.exit(f"{results_dir} already contains experiment results. "
                 f"Use another folder name, or add --overwrite to replace them.")

    split = load_and_split()
    X_train, y_train = split["X_train"], split["y_train"]
    results_dir.mkdir(exist_ok=True)

    result_rows = []
    control_rows = []
    iteration_rows = []
    bin_rows = []
    test_probabilities = {}

    for seed in settings.SPE_SEEDS:
        runs = SPE_RUNS + [(name, None) for name in BASELINES]
        subsets_by_method = {}

        for method, gamma in runs:
            log = None

            # train on the training split, then predict the test split
            try:
                start = time.perf_counter()
                if method in BASELINES:
                    classifiers = BASELINES[method](X_train, y_train, seed)
                else:
                    classifiers, log, subsets = train_spe(X_train, y_train, gamma=gamma, seed=seed)
                    subsets_by_method[method] = subsets
                train_seconds = time.perf_counter() - start

                # the test split is used only here, after training is finished
                probabilities = predict_spe(classifiers, split["X_test"])

            except Exception as error:
                result_rows.append({"method": method, "gamma": gamma, "seed": seed,
                                    "status": f"failed: {error!r}"})
                print(f"seed {seed}, {method}: FAILED ({error!r})")
                continue

            # score the test predictions
            test_probabilities[f"{method}|{seed}"] = probabilities
            row = {
                "method": method,
                "gamma": gamma,
                "seed": seed,
                "status": "ok",
                "n_classifiers": len(classifiers),
                "training_rows_per_classifier": training_rows_per_classifier(method, y_train),
                **evaluate(split["y_test"], probabilities),
                "train_seconds": train_seconds,
            }

            # selection diagnostics (SPE and UA-SPE only); full detail for the first seed
            if log is not None:
                row.update(selection_summary(log))

                if seed == settings.SPE_SEEDS[0]:
                    for entry in log:
                        iteration_rows.append({"method": method, "seed": seed, **entry["summary"]})
                        for b in entry["bins"]:
                            bin_rows.append({"method": method, "seed": seed, **b})

            result_rows.append(row)
            print(f"seed {seed}, {method}: G-mean {row['gmean_recall_specificity']:.4f}, "
                  f"AP {row['aucprc_average_precision']:.4f}, {train_seconds:.1f} s")

        if "SPE" in subsets_by_method and "UA-SPE gamma=0" in subsets_by_method:
            control_rows.append(control_check(seed, subsets_by_method, test_probabilities))

    save_results(results_dir, split, result_rows, control_rows, iteration_rows, bin_rows,
                 test_probabilities)

    print(pd.DataFrame(control_rows).to_string(index=False))
    print(f"Results written to {results_dir}")


if __name__ == "__main__":
    args = sys.argv[1:]
    overwrite = "--overwrite" in args
    folder_names = [a for a in args if a != "--overwrite"]
    main(*folder_names, overwrite=overwrite)
