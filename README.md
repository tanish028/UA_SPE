# UA-SPE: Uncertainty-Aware Self-Paced Ensemble

## Project overview

Self-Paced Ensemble (SPE) (Liu et al., ICDE 2020) handles imbalanced data by training
10 classifiers one after another, each on all fraud rows plus an equal number of
genuine rows. The genuine rows are chosen through hardness bins: rows that the earlier
classifiers find harder get more attention in later rounds. Inside each bin, SPE picks
rows uniformly at random.

UA-SPE changes only that last step. Inside a bin, rows are drawn with probability
proportional to Ũ^γ, where Ũ measures how much the earlier classifiers agree about a
row, relative to its hardness. γ = 0 gives exactly the same selection as SPE.

The project uses the ULB credit-card fraud dataset and a 10-round
`HistGradientBoostingClassifier`. It compares SPE with six baselines (plain training,
random undersampling with 1 and 10 classifiers, random oversampling, SMOTE and class
weighting) and compares SPE with UA-SPE for γ = 0, 1, 2 and 4.

## Files

| File | Purpose |
|---|---|
| `uaspe/settings.py` | All fixed settings: data path, split, seeds, number of classifiers and bins, γ values, classifier parameters |
| `uaspe/data.py` | Loads and checks the CSV, removes exact duplicates, makes the 60/20/20 split |
| `uaspe/spe.py` | The SPE and UA-SPE implementation: hardness, bins, budget allocation, Ũ and row selection |
| `uaspe/baselines.py` | The six baseline methods |
| `uaspe/metrics.py` | Average precision, G-mean and the other metrics |
| `run_experiment.py` | Runs the experiments and saves the results |
| `analyze_results.py` | Processes the saved results into comparison tables and figures |
| `tests/test_data.py` | Tests for data loading, duplicate removal and the split, including checks that the dataset and split are the expected ones |
| `tests/test_spe.py` | Tests for SPE, UA-SPE and the baselines, including the γ = 0 control |
| `requirements.txt` | Pinned package versions |

## Setup

You need Python 3.10. Open PowerShell in the repository folder, then create an
environment and install the packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use the pinned scikit-learn 1.7.2. Other versions can give slightly different results.

The dataset is not included in the repository. Download `creditcard.csv` and place it
in a `data` folder next to the repository folder. The code looks for
`../data/creditcard.csv`:

```
UA-SPE/
├── data/
│   └── creditcard.csv
└── UA_SPE/
```

## How to run

Run these commands from the repository folder, in this order:

```powershell
python run_experiment.py
python analyze_results.py
python -m pytest
```

1. `run_experiment.py` loads the dataset, removes duplicates and makes the split. It
   then trains and evaluates all 11 methods with 5 sampling seeds (55 runs) and saves
   the results in `results/`. If the dataset is not at `../data/creditcard.csv`, it
   stops with a "file not found" error. If `results/` already contains results, it
   stops instead of replacing them. To keep both, give a new folder name
   (`python run_experiment.py results_2`). To replace them, add `--overwrite`.
2. `analyze_results.py` reads the saved results and writes comparison tables and
   figures to `results/analysis/`. Wait until `run_experiment.py` has finished before
   running it. If you used another folder name in step 1, pass the same name here.
3. `python -m pytest` runs the tests. The tests that need the dataset are skipped if
   it is missing.

## Results summary

- SPE had a higher Average Precision (AP) than every tested baseline on all five
  seeds: 0.842, against 0.552–0.735 for the baselines. Most baselines had a higher
  G-mean at the 0.5 threshold, but only by flagging many more genuine transactions as
  fraud.
- UA-SPE with γ = 2 had a slightly higher mean AP than SPE (0.849 against 0.842), but
  it was better on only 3 of 5 seeds. The bootstrap interval for the difference,
  [−0.003, 0.018], includes zero, so the improvement is not conclusive. γ = 1 and γ = 4
  did not improve on SPE.
- Main limitation: everything was measured on one dataset, one train/test split and
  five seeds.
