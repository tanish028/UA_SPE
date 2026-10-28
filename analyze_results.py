"""Read the saved outputs of run_experiment.py and write comparison tables and figures
to <results folder>/analysis/.

Run after run_experiment.py has finished:  python analyze_results.py [results folder]
The default folder is results/.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uaspe import settings
from uaspe.data import load_and_split
from uaspe.metrics import evaluate
from uaspe.spe import normalized_uncertainty, predict_spe, train_spe


SPE_FAMILY = ["SPE", "UA-SPE gamma=1", "UA-SPE gamma=2", "UA-SPE gamma=4"]
BASELINES = [
    "Ordinary HGB (no resampling)", "RUS (1 classifier)", "ROS (random oversampling)",
    "SMOTE", "RUS ensemble (10 classifiers)", "Class-weighted HGB (full data)",
]

# short labels and colours for the figures
SHORT = {
    "SPE": "SPE",
    "UA-SPE gamma=0": "γ=0",
    "UA-SPE gamma=1": "γ=1",
    "UA-SPE gamma=2": "γ=2",
    "UA-SPE gamma=4": "γ=4",
    "Ordinary HGB (no resampling)": "Ordinary",
    "RUS (1 classifier)": "RUS×1",
    "ROS (random oversampling)": "ROS",
    "SMOTE": "SMOTE",
    "RUS ensemble (10 classifiers)": "RUS×10",
    "Class-weighted HGB (full data)": "Class-wt",
}
COLORS = {
    "SPE": "#2f5d8a",
    "UA-SPE gamma=1": "#5fa8a0",
    "UA-SPE gamma=2": "#e0a33a",
    "UA-SPE gamma=4": "#c8553d",
    "Ordinary HGB (no resampling)": "#3b3f46",
    "RUS (1 classifier)": "#8a8f98",
    "ROS (random oversampling)": "#7f9c6a",
    "SMOTE": "#4f7f3a",
    "RUS ensemble (10 classifiers)": "#5e6470",
    "Class-weighted HGB (full data)": "#a37fb5",
}

N_BOOTSTRAP = 1000

# saved outputs of run_experiment.py that this script reads
INPUT_FILES = ["results_per_seed.csv", "diagnostics_iterations.csv", "diagnostics_bins.csv",
               "test_probabilities.npz"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150})


def check_complete(results_dir, probabilities):
    """Stop with a short message if the folder does not hold a complete experiment run."""
    expected = [f"{m}|{s}" for m in SPE_FAMILY + BASELINES for s in settings.SPE_SEEDS]
    missing = [key for key in expected if key not in probabilities]
    if missing:
        sys.exit(f"{results_dir} has incomplete results ({len(missing)} of {len(expected)} "
                 f"method/seed runs missing, e.g. {missing[0]}). "
                 f"Run run_experiment.py again to produce a complete set.")


def paired_differences(results):
    """Per-seed difference of each method from SPE (same seed)."""
    rows = []
    spe = results[results["method"] == "SPE"].set_index("seed")

    for method in SPE_FAMILY[1:] + BASELINES:
        other = results[results["method"] == method].set_index("seed")

        for metric in ["gmean_recall_specificity", "aucprc_average_precision", "f1", "mcc"]:
            d = other[metric] - spe[metric]
            rows.append({
                "method": method,
                "metric": metric,
                "mean_difference": d.mean(),
                "min": d.min(),
                "max": d.max(),
                "seeds_better": int((d > 1e-12).sum()),
                "seeds_tied": int((d.abs() <= 1e-12).sum()),
                "seeds_worse": int((d < -1e-12).sum()),
            })

    return pd.DataFrame(rows)


def bootstrap(probabilities, y_test, rng):
    """Resample test rows; for each resample, average each method's metric over the seeds.

    This shows how much the metrics move because the test set has only 95 frauds.
    """
    methods = SPE_FAMILY + ["Ordinary HGB (no resampling)", "SMOTE", "RUS ensemble (10 classifiers)"]
    draws = {m: {"gmean": [], "ap": []} for m in methods}

    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y_test), len(y_test))

        for m in methods:
            scores = [evaluate(y_test[idx], probabilities[f"{m}|{s}"][idx])
                      for s in settings.SPE_SEEDS]
            draws[m]["gmean"].append(np.mean([x["gmean_recall_specificity"] for x in scores]))
            draws[m]["ap"].append(np.mean([x["aucprc_average_precision"] for x in scores]))

    # 95% percentile intervals for each method and for its difference from SPE
    rows = []
    for m in methods:
        for metric in ["gmean", "ap"]:
            values = np.array(draws[m][metric])
            diff = values - np.array(draws["SPE"][metric])
            rows.append({
                "method": m,
                "metric": metric,
                "ci_low": np.quantile(values, 0.025),
                "ci_high": np.quantile(values, 0.975),
                "diff_vs_spe_ci_low": np.quantile(diff, 0.025),
                "diff_vs_spe_ci_high": np.quantile(diff, 0.975),
            })

    return pd.DataFrame(rows)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_results_per_seed(results, out):
    """Figure 1: per-seed G-mean and AP."""
    order = SPE_FAMILY + BASELINES
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    panels = [(axes[0], "gmean_recall_specificity", "G-mean = √(recall·specificity)"),
              (axes[1], "aucprc_average_precision", "Average precision")]

    for ax, metric, title in panels:
        for x, m in enumerate(order):
            values = results.loc[results["method"] == m, metric]
            jitter = np.linspace(-0.12, 0.12, len(values))
            ax.scatter(np.full(len(values), x) + jitter, values, color=COLORS[m], s=22, alpha=0.8)
            ax.hlines(values.mean(), x - 0.28, x + 0.28, color="#222", lw=2)

        ax.set_xticks(range(len(order)), [SHORT[m] for m in order], rotation=45, ha="right")
        ax.set_title(title)
        ax.axvline(3.5, color="#bbb", lw=1, ls="--")

    axes[1].set_ylim(0, 1)
    fig.suptitle("Test-set results, 5 sampling seeds (dots) and mean (bar)", y=1.02)
    save_figure(fig, out / "fig_results_per_seed.png")


def plot_operating_points(results, out):
    """Figure 2: operating points at threshold 0.5 (recall against false positives)."""
    fig, ax = plt.subplots(figsize=(6.4, 4))

    for m in SPE_FAMILY + BASELINES:
        sub = results[results["method"] == m]
        ax.scatter(sub["fp"], sub["recall"], color=COLORS[m], label=SHORT[m], s=30, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("False positives on the test set (log scale, 56,651 genuine)")
    ax.set_ylabel("Recall (95 frauds)")
    ax.set_title("Operating point at threshold 0.5")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    save_figure(fig, out / "fig_operating_points.png")


def plot_u_tilde_distribution(hardness, u_tilde, out):
    """Figure 3: U~ at the last selection step of SPE, seed 0."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    axes[0].hist(u_tilde, bins=np.linspace(0.8, 1.0, 81), color="#2f5d8a")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Ũ at the last selection step (169,951 majority rows)")
    axes[0].set_ylabel("Rows (log scale)")
    axes[0].set_title("Ũ is concentrated near 1")

    hb = axes[1].hexbin(hardness, u_tilde, gridsize=45, bins="log", cmap="Blues", mincnt=1)
    axes[1].set_xlabel("Hardness H")
    axes[1].set_ylabel("Ũ")
    axes[1].set_title("Ũ against hardness")
    fig.colorbar(hb, ax=axes[1], label="rows (log)")

    save_figure(fig, out / "fig_u_tilde_distribution.png")


def plot_selection_change(iterations, out):
    """Figure 4: how far each gamma moves the selection away from uniform."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    for m in ["UA-SPE gamma=1", "UA-SPE gamma=2", "UA-SPE gamma=4"]:
        sub = iterations[iterations["method"] == m]
        axes[0].plot(sub["iteration"], sub["tv_from_uniform_weighted"], "o-",
                     color=COLORS[m], label=SHORT[m])
        axes[1].plot(sub["iteration"], sub["rel_dev_frac_rows"], "o-",
                     color=COLORS[m], label=SHORT[m])

    axes[0].set_title("TV distance from uniform selection")
    axes[1].set_title("Share of rows whose probability moves >10%")
    for ax in axes:
        ax.set_xlabel("Selection step i (seed 0)")
        ax.legend(frameon=False)

    save_figure(fig, out / "fig_selection_change.png")


def plot_bins_allocation(bins, iterations, out):
    """Figure 5: rows and allocation per hardness bin (SPE, seed 0, steps 1 and 9)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)

    for ax, step in zip(axes, [1, settings.N_CLASSIFIERS - 1]):
        sub = bins[(bins["method"] == "SPE") & (bins["iteration"] == step)]

        # rows in each bin (left axis) and rows selected from it (right axis)
        rows_bars = ax.bar(sub["bin"] - 0.2, sub["rows"], width=0.4, color="#c9d6e5")
        ax.set_yscale("log")
        ax2 = ax.twinx()
        selected_bars = ax2.bar(sub["bin"] + 0.2, sub["allocation"], width=0.4, color="#2f5d8a")

        ax.set_xticks(range(0, 20, 2))
        ax.set_xlim(-0.6, 19.6)
        ax.set_xlabel("Hardness bin l (width 0.05)")

        is_step = (iterations["method"] == "SPE") & (iterations["iteration"] == step)
        alpha = iterations[is_step]["alpha"].iloc[0]
        ax.set_title(f"Step {step}: α = {alpha:.2f}")
        ax.set_ylabel("Rows in bin (log)")
        ax2.set_ylabel("Rows selected")
        ax2.spines["right"].set_visible(True)

        if step == 1:
            ax.legend([rows_bars, selected_bars], ["majority rows in bin", "rows selected"],
                      loc="upper right", frameon=False, fontsize=8)

    save_figure(fig, out / "fig_bins_allocation.png")


def plot_gamma_effect(hardness, u_tilde, out):
    """Figure 6: effect of gamma, using real U~ values from one bin of the last step."""
    in_bin = (hardness >= 0.30) & (hardness < 0.35)
    example = np.sort(np.random.default_rng(1).choice(u_tilde[in_bin], 8, replace=False))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 0.2
    colors = ["#8a8f98", "#5fa8a0", "#e0a33a", "#c8553d"]

    for j, g in enumerate([0, 1, 2, 4]):
        if g == 0:
            w = np.ones_like(example)
        else:
            w = example ** g
        ax.bar(np.arange(8) + (j - 1.5) * width, w / w.sum(), width, label=f"γ={g}", color=colors[j])

    ax.set_xticks(range(8), [f"{u:.3f}" for u in example])
    ax.set_xlabel("Ũ of 8 rows sampled from bin [0.30, 0.35) at the last step")
    ax.set_ylabel("Single-draw probability")
    ax.set_title("Effect of γ on within-bin probabilities")
    ax.legend(frameon=False, ncol=4)
    save_figure(fig, out / "fig_gamma_effect.png")


def plot_class_imbalance(y_train, out):
    """Figure 7: class imbalance in the training split."""
    counts = pd.Series(y_train).value_counts()
    fig, ax = plt.subplots(figsize=(5.5, 2.2))

    ax.barh(["Fraud (P)", "Genuine (N)"], [counts[1], counts[0]], color=["#c8553d", "#2f5d8a"])
    for y_pos, v in enumerate([counts[1], counts[0]]):
        ax.text(v, y_pos, f"  {v:,}", va="center")

    ax.set_xscale("log")
    ax.set_title(f"Training split: {counts[0] / counts[1]:.0f} genuine per fraud")
    save_figure(fig, out / "fig_class_imbalance.png")


def main(results_name="results"):
    results_dir = settings.REPO_ROOT / results_name

    missing_files = [name for name in INPUT_FILES if not (results_dir / name).exists()]
    if missing_files:
        sys.exit(f"{results_dir} is missing {', '.join(missing_files)}. "
                 f"Run run_experiment.py first and let it finish.")

    # saved outputs of run_experiment.py
    results = pd.read_csv(results_dir / "results_per_seed.csv")
    results = results[results["status"] == "ok"]
    iterations = pd.read_csv(results_dir / "diagnostics_iterations.csv")
    bins = pd.read_csv(results_dir / "diagnostics_bins.csv")
    saved = np.load(results_dir / "test_probabilities.npz")
    probabilities = {k: saved[k] for k in saved.files if k != "y_test"}
    y_test = saved["y_test"]
    check_complete(results_dir, probabilities)

    out = results_dir / "analysis"
    out.mkdir(exist_ok=True)

    # comparison tables
    paired_differences(results).to_csv(out / "paired_differences_vs_spe.csv", index=False)
    bootstrap(probabilities, y_test, np.random.default_rng(0)).to_csv(
        out / "bootstrap_test_set.csv", index=False)

    # retrain SPE (seed 0), check it reproduces the saved run, and rebuild H and U~
    # at its last selection step
    split = load_and_split()
    first_seed = settings.SPE_SEEDS[0]
    classifiers, _, _ = train_spe(split["X_train"], split["y_train"], gamma=None, seed=first_seed)

    rebuilt = predict_spe(classifiers, split["X_test"])
    assert np.array_equal(rebuilt, probabilities[f"SPE|{first_seed}"]), "SPE run is not reproducible"

    is_majority = split["y_train"] == settings.MAJORITY_LABEL
    X_majority = np.asarray(split["X_train"], dtype=float)[is_majority]
    history = [c.predict_proba(X_majority)[:, 1] for c in classifiers[:-1]]
    hardness = np.mean(history, axis=0)
    u_tilde = normalized_uncertainty(history)

    last_step = pd.DataFrame({"hardness": hardness, "u_tilde": u_tilde})
    last_step.describe(percentiles=[0.01, 0.1, 0.5, 0.9]).to_csv(
        out / "last_step_hardness_u_tilde_summary.csv")

    # figures
    plot_results_per_seed(results, out)
    plot_operating_points(results, out)
    plot_u_tilde_distribution(hardness, u_tilde, out)
    plot_selection_change(iterations, out)
    plot_bins_allocation(bins, iterations, out)
    plot_gamma_effect(hardness, u_tilde, out)
    plot_class_imbalance(split["y_train"], out)

    print(f"Analysis written to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
