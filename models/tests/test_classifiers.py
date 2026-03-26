"""Classification Tests focus on data pipeline correctness (shapes, types, no leakage),
not model accuracy (which is the experiment's result, not an assertion).
Run:
    python -m pytest models/tests/test_classifiers.py -v
"""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from api.app.config import PROJECT_ROOT, RANDOM_SEED
from models.classifiers import (
    ALL_FEATURES,
    ALL_FEATURES_LR,
    TIMING_FEATURES,
    STRUCTURAL_FEATURES,
    ABLATION_SETS,
    load_features,
    split_data,
    prepare_features,
    make_lr,
    make_rf,
    evaluate_model,
    run_ablation,
    analyze_errors,
)

#  Fixtures 

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"

@pytest.fixture(scope="module")
def raw_df():
    """Load the features CSV for all tests"""
    return load_features(FEATURES_PATH)


@pytest.fixture(scope="module")
def split(raw_df):
    """Stratified split, reused across tests."""
    train_df, test_df, y_train, y_test, le = split_data(raw_df)
    return train_df, test_df, y_train, y_test, le


@pytest.fixture(scope="module")
def class_names(split):
    """Sorted class label strings."""
    _, _, _, _, le = split
    return list(le.classes_)


#  TEST 1: Train/test split shapes

def test_load_and_split_shapes(raw_df, split):
    """Train + test = total rows; both splits are non-empty;
    all 3 classes present in both splits."""
    train_df, test_df, y_train, y_test, le = split
    n = len(raw_df)

    # 80/20 split
    assert len(y_train) + len(y_test) == n
    assert len(y_train) == pytest.approx(n * 0.8, abs=2)
    assert len(y_test) == pytest.approx(n * 0.2, abs=2)

    # All 3 classes in both splits
    assert set(np.unique(y_train)) == {0, 1, 2}
    assert set(np.unique(y_test)) == {0, 1, 2}


#  TEST 2: Stratification preserves class ratios

def test_stratification_preserves_ratios(raw_df, split):
    """Class proportions in train and test are within 5% of full dataset."""
    _, _, y_train, y_test, le = split
    full_y = le.transform(raw_df["actor_type"])

    for cls in range(3):
        full_ratio = (full_y == cls).mean()
        train_ratio = (y_train == cls).mean()
        test_ratio = (y_test == cls).mean()
        assert abs(train_ratio - full_ratio) < 0.05, (
            f"Train ratio for class {cls} deviates >5% from full dataset"
        )
        assert abs(test_ratio - full_ratio) < 0.05, (
            f"Test ratio for class {cls} deviates >5% from full dataset"
        )


#  TEST 3: Feature groups are disjoint and complete

def test_feature_groups_no_overlap():
    """TIMING and STRUCTURAL are disjoint; their union = ALL_FEATURES.
    ALL_FEATURES_LR = ALL minus request_rate."""
    timing_set = set(TIMING_FEATURES)
    struct_set = set(STRUCTURAL_FEATURES)

    # Disjoint
    assert timing_set & struct_set == set(), "TIMING and STRUCTURAL overlap"

    # Union = ALL
    assert timing_set | struct_set == set(ALL_FEATURES)

    # LR excludes exactly request_rate
    assert set(ALL_FEATURES_LR) == set(ALL_FEATURES) - {"request_rate"}


#  TEST 4: No NaN after imputation

def test_imputer_no_nans_after_transform(split):
    """After prepare_features, X_train and X_test have zero NaN."""
    train_df, test_df, _, _, _ = split
    X_train, X_test, _ = prepare_features(train_df, test_df, ALL_FEATURES)

    assert not np.isnan(X_train).any(), "NaN found in X_train after imputation"
    assert not np.isnan(X_test).any(), "NaN found in X_test after imputation"


#  TEST 5: Imputer fitted on train only (no data leakage)

def test_imputer_fitted_on_train_only(split):
    """Changing test set values doesn't affect training imputation.
    Strategy: corrupt test_df with extreme NaN, verify X_train unchanged."""
    train_df, test_df, _, _, _ = split

    # Baseline: normal prep
    X_train_base, _, _ = prepare_features(train_df, test_df, ALL_FEATURES)

    # Corrupt test: set all recovery_time to NaN
    test_corrupted = test_df.copy()
    test_corrupted["recovery_time"] = np.nan
    X_train_corrupted, _, _ = prepare_features(
        train_df, test_corrupted, ALL_FEATURES
    )

    # Training arrays must be identical — imputer doesn't see test data
    np.testing.assert_array_equal(
        X_train_base, X_train_corrupted,
        err_msg="X_train changed when test data was modified — data leakage!"
    )


#  TEST 6: LR feature set excludes request_rate

def test_lr_excludes_request_rate(split):
    """LR's prepared feature matrix has fewer columns than RF's,
    and 'request_rate' is not among LR's feature names."""
    train_df, test_df, _, _, _ = split

    _, _, lr_feat = prepare_features(train_df, test_df, ALL_FEATURES_LR)
    _, _, rf_feat = prepare_features(train_df, test_df, ALL_FEATURES)

    # LR has fewer features (no request_rate + potentially fewer indicators)
    assert len(lr_feat) < len(rf_feat)

    # No raw or indicator column for request_rate in LR
    assert "request_rate" not in lr_feat
    assert "request_rate_missing" not in lr_feat


#  TEST 7: Model output shapes

def test_model_output_shapes(split):
    """predict() returns 1D array of len(X_test);
    predict_proba() returns (len(X_test), 3)."""
    train_df, test_df, y_train, y_test, _ = split
    X_train, X_test, _ = prepare_features(train_df, test_df, ALL_FEATURES)

    for model in [make_lr(), make_rf()]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        assert preds.shape == (len(y_test),)
        assert proba.shape == (len(y_test), 3)
        # Probabilities sum to 1 per row
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


#  TEST 8: Metrics JSON schema

def test_metrics_json_schema():
    """Saved JSON files contain expected keys."""
    for name in ["lr_results.json", "rf_results.json"]:
        path = PROJECT_ROOT / "report" / "metrics" / name
        if not path.exists():
            pytest.skip(f"{path} not found — run pipeline first")
        with open(path) as f:
            data = json.load(f)
        # Top-level keys: per-class dicts + summary keys
        assert "accuracy" in data
        assert "macro avg" in data
        assert "weighted avg" in data
        # Per-class keys
        for cls in ["bot", "human", "llm_agent"]:
            assert cls in data, f"Missing class '{cls}' in {name}"
            assert "precision" in data[cls]
            assert "recall" in data[cls]
            assert "f1-score" in data[cls]


#  TEST 9: Ablation produces 6 results

def test_ablation_produces_six_results(split, class_names):
    """3 feature sets × 2 models = 6 entries in ablation results."""
    train_df, test_df, y_train, y_test, _ = split
    results = run_ablation(train_df, test_df, y_train, y_test, class_names)

    assert len(results) == 6

    expected_keys = [
        "ALL_LR", "ALL_RF",
        "TIMING_LR", "TIMING_RF",
        "STRUCTURAL_LR", "STRUCTURAL_RF",
    ]
    for key in expected_keys:
        assert key in results, f"Missing ablation key: {key}"
        assert "macro_f1" in results[key]
        assert 0.0 <= results[key]["macro_f1"] <= 1.0


#  TEST 10: Seed determinism

def test_seed_determinism(raw_df):
    """Two identical runs produce identical predictions."""
    # Run 1
    train1, test1, yt1, yts1, _ = split_data(raw_df, seed=RANDOM_SEED)
    X_tr1, X_ts1, _ = prepare_features(train1, test1, ALL_FEATURES)
    rf1 = make_rf(RANDOM_SEED)
    rf1.fit(X_tr1, yt1)
    pred1 = rf1.predict(X_ts1)

    # Run 2
    train2, test2, yt2, yts2, _ = split_data(raw_df, seed=RANDOM_SEED)
    X_tr2, X_ts2, _ = prepare_features(train2, test2, ALL_FEATURES)
    rf2 = make_rf(RANDOM_SEED)
    rf2.fit(X_tr2, yt2)
    pred2 = rf2.predict(X_ts2)

    np.testing.assert_array_equal(pred1, pred2,
                                  err_msg="Same seed produced different predictions")


#  TEST 11: Error analysis returns misclassified rows

def test_error_analysis_returns_misclassified():
    """analyze_errors returns only rows where y_true != y_pred,
    with true_label and predicted columns. Also handles zero-error case."""
    # Synthetic data: 6 samples, 2 misclassified
    test_df = pd.DataFrame({
        "session_id": [f"s{i}" for i in range(6)],
        "actor_type": ["bot", "bot", "human", "human", "llm_agent", "llm_agent"],
        "request_count": [10, 10, 10, 10, 10, 10],
        "mean_gap_time": [50, 50, 5000, 5000, 12000, 12000],
        "std_gap_time": [5, 5, 3000, 3000, 1500, 1500],
        "max_delay": [80, 80, 10000, 10000, 15000, 15000],
        "session_duration": [1000, 1000, 20000, 20000, 90000, 90000],
        "request_rate": [18, 18, 0.3, 0.3, 0.1, 0.1],
        "burstiness": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "recovery_time": [50, 50, 6000, 6000, 11000, 11000],
        "session_entropy": [1.0, 1.0, 1.2, 1.2, 1.1, 1.1],
        "payload_entropy": [0.0, 0.0, 0.7, 0.7, 1.7, 1.7],
        "path_repetition": [1.0, 1.0, 0.1, 0.1, 0.5, 0.5],
    })
    class_names = ["bot", "human", "llm_agent"]

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 1])  # 2 wrong: human→agent, agent→human

    errors = analyze_errors(test_df, y_true, y_pred, class_names)
    assert len(errors) == 2
    assert "true_label" in errors.columns
    assert "predicted" in errors.columns
    assert set(errors["true_label"]) == {"human", "llm_agent"}

    # Zero-error case
    y_perfect = y_true.copy()
    errors_empty = analyze_errors(test_df, y_true, y_perfect, class_names)
    assert len(errors_empty) == 0