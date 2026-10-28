"""Experiment settings shared by every script."""
from pathlib import Path


# data
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT.parent / "data" / "creditcard.csv"  # kept outside the repository

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
MINORITY_LABEL = 1  # P = fraud
MAJORITY_LABEL = 0  # N = genuine

# exact duplicate rows are removed before the split
TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION = 0.6, 0.2, 0.2
SPLIT_SEED = 42

# SPE / UA-SPE
N_CLASSIFIERS = 10      # n
N_HARDNESS_BINS = 20    # k, fixed bins over [0, 1]
GAMMA_VALUES = (0, 1, 2, 4)
CLASSIFICATION_THRESHOLD = 0.5
SPE_SEEDS = (0, 1, 2, 3, 4)   # sampling seeds; the data split stays fixed (SPLIT_SEED)

# thresholds used only in the selection diagnostics
NEAR_ONE_U_TILDE = 0.99
RELATIVE_DEVIATION_THRESHOLD = 0.10

# base classifier: HistGradientBoostingClassifier
# 10 boosting rounds, as in the SPE paper's 10-round GBDT (but not the paper's implementation)
# no early stopping; all 30 features are numeric, so none is treated as categorical
# every other parameter is the scikit-learn default for SKLEARN_VERSION
SKLEARN_VERSION = "1.7.2"
HGB_PARAMS = {"max_iter": 10, "early_stopping": False, "categorical_features": None}
