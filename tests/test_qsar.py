"""Tests for utils/qsar.py."""

import math
import pandas as pd
import pytest

from utils.qsar import (
    ModelArtifact, train_qsar,
)
from utils.qsar import save_model_artifact, load_model_artifact


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


def test_save_load_round_trip(tmp_db, tmp_path, sample_dataset_df):
    from utils.rdkit_utils import mol_from_smiles, calculate_descriptor_set

    artifact, metrics = train_qsar(sample_dataset_df, "SMILES", "Activity")
    models_dir = tmp_path / "qsar_models"
    meta = {
        "name": "RoundTrip", "dataset_name": "sample.csv",
        "activity_label": "pIC50", "activity_transform": "none",
        "higher_is_better": 1,
        "cv_r2_mean": metrics["cv_r2_mean"], "cv_r2_std": metrics["cv_r2_std"],
        "project_id": None,
    }
    model_id = save_model_artifact(artifact, meta, tmp_db, models_dir=models_dir)
    assert model_id > 0
    assert (models_dir / f"{model_id}.joblib").exists()

    loaded = load_model_artifact(model_id, tmp_db, models_dir=models_dir)
    assert loaded.feature_columns == artifact.feature_columns
    assert loaded.training_smiles_hashes == artifact.training_smiles_hashes
    assert loaded.training_y_min == artifact.training_y_min
    assert loaded.training_y_max == artifact.training_y_max

    # Predictions must match exactly across the round-trip.
    desc = calculate_descriptor_set(mol_from_smiles("CCO"))
    X = artifact.scaler.transform(
        pd.DataFrame([desc])[artifact.feature_columns].values
    )
    orig_pred = artifact.model.predict(X)
    loaded_pred = loaded.model.predict(X)
    assert (orig_pred == loaded_pred).all()


def test_save_persists_metadata_to_db(tmp_db, tmp_path, sample_dataset_df):
    artifact, metrics = train_qsar(sample_dataset_df, "SMILES", "Activity")
    meta = {
        "name": "MetaTest", "dataset_name": "x.csv",
        "activity_label": "pIC50", "activity_transform": "none",
        "higher_is_better": 1,
        "cv_r2_mean": metrics["cv_r2_mean"], "cv_r2_std": metrics["cv_r2_std"],
        "project_id": None,
    }
    model_id = save_model_artifact(artifact, meta, tmp_db, models_dir=tmp_path)
    rows = tmp_db.get_qsar_models()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == model_id
    assert row["name"] == "MetaTest"
    assert row["n_molecules"] == len(sample_dataset_df)
    assert row["model_type"] == "RandomForestRegressor"
    assert row["rdkit_version"] == artifact.rdkit_version
    assert row["sklearn_version"] == artifact.sklearn_version
    assert row["artifact_path"].endswith(f"{model_id}.joblib")


def test_save_rolls_back_db_row_on_dump_failure(tmp_db, tmp_path, sample_dataset_df, monkeypatch):
    artifact, metrics = train_qsar(sample_dataset_df, "SMILES", "Activity")
    meta = {
        "name": "RollbackTest", "dataset_name": "x.csv",
        "activity_label": "pIC50", "activity_transform": "none",
        "higher_is_better": 1,
        "cv_r2_mean": 0.0, "cv_r2_std": 0.0,
        "project_id": None,
    }

    def boom(*args, **kwargs):
        raise IOError("simulated disk failure")

    monkeypatch.setattr("utils.qsar.joblib.dump", boom)

    with pytest.raises(IOError, match="simulated disk failure"):
        save_model_artifact(artifact, meta, tmp_db, models_dir=tmp_path)

    # DB row must not exist after rollback
    assert tmp_db.get_qsar_models() == []


def test_load_missing_artifact_file_raises(tmp_db, tmp_path, sample_dataset_df):
    artifact, metrics = train_qsar(sample_dataset_df, "SMILES", "Activity")
    meta = {
        "name": "WillVanish", "dataset_name": "x.csv",
        "activity_label": "pIC50", "activity_transform": "none",
        "higher_is_better": 1, "cv_r2_mean": 0.0, "cv_r2_std": 0.0,
        "project_id": None,
    }
    model_id = save_model_artifact(artifact, meta, tmp_db, models_dir=tmp_path)
    # User deletes the .joblib out from under us
    (tmp_path / f"{model_id}.joblib").unlink()

    with pytest.raises(FileNotFoundError):
        load_model_artifact(model_id, tmp_db, models_dir=tmp_path)
