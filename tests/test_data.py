""" this is the Data pipeline: schema checks, duplicate removal, split integrity, leakage."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from uaspe import settings
from uaspe.data import (class_counts, load_and_split, load_raw_data,
                        remove_exact_duplicates, stratified_split_indices)

# SHA-256 fingerprints of the validated CSV file and of the split row ids
# if either changes, the tests fail because the data or the split is different
EXPECTED_DATA_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
EXPECTED_SPLIT_FINGERPRINT = "22f9b6db604923ca"

FEATURES = ["Time", "V1", "Amount"]


def small_table(n_rows=2000, n_minority=40, seed=0):
    """Hand-controlled imbalanced table shaped like Credit Fraud."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Time": np.arange(n_rows, dtype=float),
                       "V1": rng.normal(size=n_rows),
                       "Amount": rng.exponential(50.0, size=n_rows)})
    df["Class"] = 0
    df.loc[rng.choice(n_rows, n_minority, replace=False), "Class"] = 1
    return df


def write_and_load(tmp_path, df):
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    return load_raw_data(path, FEATURES, "Class")


# load and check the data
def test_loader_keeps_columns_and_row_ids(tmp_path):
    df = write_and_load(tmp_path, small_table())
    assert list(df.columns) == FEATURES + ["Class"]
    assert np.array_equal(df.index.to_numpy(), np.arange(2000))   # row_id = CSV order


def test_loader_rejects_missing_column(tmp_path):
    df = small_table().drop(columns="V1")
    with pytest.raises(ValueError, match="Column mismatch"):
        write_and_load(tmp_path, df)


def test_loader_rejects_unexpected_column(tmp_path):
    df = small_table()
    df["extra"] = 1.0
    with pytest.raises(ValueError, match="Column mismatch"):
        write_and_load(tmp_path, df)


def test_loader_rejects_missing_values(tmp_path):
    df = small_table()
    df.loc[3, "V1"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        write_and_load(tmp_path, df)


def test_loader_rejects_non_numeric_feature(tmp_path):
    df = small_table()
    df["V1"] = "abc"
    with pytest.raises(ValueError, match="Non-numeric"):
        write_and_load(tmp_path, df)


def test_loader_rejects_labels_other_than_0_and_1(tmp_path):
    df = small_table()
    df.loc[df["Class"] == 1, "Class"] = 2
    with pytest.raises(ValueError, match="exactly the labels"):
        write_and_load(tmp_path, df)


def test_loader_rejects_label_1_as_majority(tmp_path):
    df = small_table()
    df["Class"] = 1 - df["Class"]
    with pytest.raises(ValueError, match="must be the minority"):
        write_and_load(tmp_path, df)


def test_class_counts_by_hand():
    assert class_counts([0, 0, 0, 0, 0, 0, 1, 1]) == {"rows": 8, "N": 6, "P": 2, "imbalance_ratio": 3.0}


# remove duplicates
def test_exact_duplicates_removed_first_occurrence_kept():
    df = pd.DataFrame({"Time": [0., 1., 0., 2., 1., 1.],
                       "V1": [5., 6., 5., 7., 6., 6.],
                       "Class": [0, 1, 0, 0, 1, 0]})
    # row 2 repeats row 0 and row 4 repeats row 1. row 5 has the features of row 1
    # but a different label, so it is not an exact duplicate and must be kept
    kept, summary = remove_exact_duplicates(df)
    assert list(kept.index) == [0, 1, 3, 5]            # original row ids preserved
    assert summary["removed"] == {"rows": 2, "N": 1, "P": 1}
    assert summary["before"]["rows"] == 6 and summary["after"]["rows"] == 4
    assert summary["after"]["N"] == 3 and summary["after"]["P"] == 1


# split the dataset
@pytest.fixture
def labels():
    return small_table()["Class"].to_numpy()


def test_split_is_a_partition_with_60_20_20_sizes(labels):
    tr, va, te = stratified_split_indices(labels, seed=1)
    assert (len(tr), len(va), len(te)) == (1200, 400, 400)
    assert np.array_equal(np.sort(np.concatenate([tr, va, te])), np.arange(len(labels)))


def test_split_is_stratified(labels):
    tr, va, te = stratified_split_indices(labels, seed=1)
    assert (labels[tr].sum(), labels[va].sum(), labels[te].sum()) == (24, 8, 8)


def test_split_is_deterministic_in_the_seed(labels):
    a = stratified_split_indices(labels, seed=7)
    b = stratified_split_indices(labels, seed=7)
    c = stratified_split_indices(labels, seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[2], c[2])


# checks on the real creditcard.csv (skipped if the file is missing)
needs_csv = pytest.mark.skipif(not settings.DATA_PATH.exists(), reason="creditcard.csv not found")


@pytest.fixture(scope="module")
def real_split():
    return load_and_split()


@needs_csv
def test_real_deduplication_counts(real_split):
    d = real_split["deduplication"]
    assert (d["before"]["rows"], d["before"]["N"], d["before"]["P"]) == (284807, 284315, 492)
    assert (d["removed"]["rows"], d["removed"]["N"], d["removed"]["P"]) == (1081, 1062, 19)
    assert (d["after"]["rows"], d["after"]["N"], d["after"]["P"]) == (283726, 283253, 473)


@needs_csv
def test_real_file_is_the_validated_version():
    """Catches a changed CSV that still has the right shape (e.g. re-saved in Excel)."""
    with open(settings.DATA_PATH, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_DATA_SHA256


@needs_csv
def test_real_split_sizes_counts_and_fingerprint(real_split):
    s = real_split
    assert [len(s[f"y_{k}"]) for k in ("train", "val", "test")] == [170235, 56745, 56746]
    assert [int(s[f"y_{k}"].sum()) for k in ("train", "val", "test")] == [284, 94, 95]
    assert list(s["X_train"].columns) == settings.FEATURE_COLUMNS   # Time kept, Class absent
    # the sizes and counts could match even if different rows ended up in each split,
    # so we also hash the row ids of the three splits and compare the fingerprint
    digest = hashlib.sha256()
    for ids in (s["train_row_ids"], s["val_row_ids"], s["test_row_ids"]):
        digest.update(np.asarray(ids, dtype=np.int64).tobytes() + b"|")
    assert digest.hexdigest()[:16] == EXPECTED_SPLIT_FINGERPRINT


@needs_csv
def test_real_split_has_no_leakage(real_split):
    s = real_split
    tr, va, te = (set(s[f"{k}_row_ids"]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == 283726
    # X rows really are the rows named by the ids
    assert np.array_equal(s["X_test"].index.to_numpy(), s["test_row_ids"])
