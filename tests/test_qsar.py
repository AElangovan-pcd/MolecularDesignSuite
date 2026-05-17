"""Tests for utils/qsar.py."""

import math
import pandas as pd
import pytest

from utils.qsar import (
    ModelArtifact, train_qsar,
)


def test_train_qsar_returns_artifact_and_metrics(sample_dataset_df):
    artifact, metrics = train_qsar(
        sample_dataset_df, smiles_col="SMILES", activity_col="Activity",
    )
    assert isinstance(artifact, ModelArtifact)
    assert len(artifact.feature_columns) > 0
    assert artifact.scaler is not None
    assert artifact.model is not None
    assert "cv_r2_mean" in metrics
    assert "cv_r2_std" in metrics
    assert "n_molecules" in metrics
    assert metrics["n_molecules"] == len(sample_dataset_df)
    # cv_y_actual / cv_y_predicted let the UI plot pred-vs-actual without
    # re-deriving descriptors and re-applying the activity transform.
    assert "cv_y_actual" in metrics
    assert "cv_y_predicted" in metrics
    assert len(metrics["cv_y_actual"]) == len(sample_dataset_df)
    assert len(metrics["cv_y_predicted"]) == len(sample_dataset_df)


def test_train_qsar_raises_on_all_invalid_smiles():
    df = pd.DataFrame({
        "SMILES": ["not", "valid", "smiles", "here", "either", "x", "y", "z", "q", "p"],
        "Activity": list(range(10)),
    })
    with pytest.raises(ValueError, match="No valid molecules"):
        train_qsar(df, smiles_col="SMILES", activity_col="Activity")


def test_train_qsar_records_versions_and_hashes(sample_dataset_df):
    artifact, _ = train_qsar(
        sample_dataset_df, smiles_col="SMILES", activity_col="Activity",
    )
    assert artifact.rdkit_version  # non-empty string
    assert artifact.sklearn_version  # non-empty string
    assert len(artifact.training_smiles_hashes) == len(sample_dataset_df)


def test_train_qsar_records_y_range(sample_dataset_df):
    artifact, _ = train_qsar(
        sample_dataset_df, smiles_col="SMILES", activity_col="Activity",
    )
    # With activity_transform='none', y_min/y_max match raw activity range.
    activities = sample_dataset_df["Activity"].tolist()
    assert artifact.training_y_min == pytest.approx(min(activities))
    assert artifact.training_y_max == pytest.approx(max(activities))


def test_train_qsar_log10_transform_changes_y(sample_dataset_df):
    artifact_none, _ = train_qsar(
        sample_dataset_df, "SMILES", "Activity", activity_transform="none",
    )
    artifact_log, _ = train_qsar(
        sample_dataset_df, "SMILES", "Activity", activity_transform="log10",
    )
    # log10 transform must produce a different y range
    assert artifact_log.training_y_min != pytest.approx(artifact_none.training_y_min)
    assert artifact_log.training_y_max != pytest.approx(artifact_none.training_y_max)


def test_train_qsar_pic50_transform(pic50_dataset_df):
    artifact, _ = train_qsar(
        pic50_dataset_df, "SMILES", "Activity", activity_transform="pIC50",
    )
    # pIC50 = -log10(IC50_in_M). For IC50=1e-9 M, pIC50=9; for IC50=1e-5, pIC50=5.
    # max pIC50 corresponds to min IC50 (most potent), so y_max should be ~9.
    ic50s = pic50_dataset_df["Activity"].tolist()
    expected_pic50_max = -math.log10(min(ic50s))
    expected_pic50_min = -math.log10(max(ic50s))
    assert artifact.training_y_max == pytest.approx(expected_pic50_max, rel=1e-6)
    assert artifact.training_y_min == pytest.approx(expected_pic50_min, rel=1e-6)


def test_train_qsar_pic50_raises_on_nonpositive():
    df = pd.DataFrame({
        "SMILES": ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
                   "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "CCCCc1ccccc1"],
        "Activity": [-1.0, 0.0, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9],
    })
    with pytest.raises(ValueError, match="pIC50 transform requires positive"):
        train_qsar(df, "SMILES", "Activity", activity_transform="pIC50")


def test_train_qsar_log10_raises_on_nonpositive():
    df = pd.DataFrame({
        "SMILES": ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
                   "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "CCCCc1ccccc1"],
        "Activity": [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })
    with pytest.raises(ValueError, match="log10 transform requires positive"):
        train_qsar(df, "SMILES", "Activity", activity_transform="log10")
