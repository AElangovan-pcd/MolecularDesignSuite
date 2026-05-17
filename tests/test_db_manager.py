"""Tests for database/db_manager.py."""

import pytest


def test_add_molecule_duplicate_returns_existing_id(tmp_db):
    """The INSERT OR IGNORE contract: adding the same SMILES twice returns
    the original id, never raises and never creates a duplicate row.
    """
    id1 = tmp_db.add_molecule(
        smiles="CCO", canonical_smiles="CCO", name="ethanol",
        formula="C2H6O", mw=46.07,
    )
    id2 = tmp_db.add_molecule(
        smiles="CCO", canonical_smiles="CCO", name="ethanol-again",
        formula="C2H6O", mw=46.07,
    )
    assert id1 == id2
    assert tmp_db.get_molecule_count() == 1


def test_get_molecule_by_smiles_returns_dict(tmp_db):
    mol_id = tmp_db.add_molecule(
        smiles="CCO", canonical_smiles="CCO", name="ethanol",
    )
    row = tmp_db.get_molecule_by_smiles("CCO")
    assert row is not None
    assert row["id"] == mol_id
    assert row["name"] == "ethanol"


def test_get_molecule_by_smiles_missing_returns_none(tmp_db):
    assert tmp_db.get_molecule_by_smiles("NEVER_INSERTED") is None


def test_add_qsar_model_returns_id(tmp_db):
    meta = {
        "name": "TestRF",
        "dataset_name": "test.csv",
        "n_molecules": 13,
        "activity_label": "pIC50",
        "activity_transform": "none",
        "higher_is_better": 1,
        "cv_r2_mean": 0.71,
        "cv_r2_std": 0.05,
        "model_type": "RandomForestRegressor",
        "artifact_path": "data/qsar_models/1.joblib",
        "rdkit_version": "2023.9.5",
        "sklearn_version": "1.4.0",
        "project_id": None,
    }
    model_id = tmp_db.add_qsar_model(meta)
    assert isinstance(model_id, int)
    assert model_id > 0


def test_get_qsar_models_returns_inserted_row(tmp_db):
    meta = {
        "name": "TestRF", "dataset_name": "x.csv", "n_molecules": 5,
        "activity_label": "pIC50", "activity_transform": "log10",
        "higher_is_better": 0, "cv_r2_mean": 0.3, "cv_r2_std": 0.1,
        "model_type": "RandomForestRegressor",
        "artifact_path": "data/qsar_models/1.joblib",
        "rdkit_version": "x", "sklearn_version": "y", "project_id": None,
    }
    tmp_db.add_qsar_model(meta)
    rows = tmp_db.get_qsar_models()
    assert len(rows) == 1
    assert rows[0]["name"] == "TestRF"
    assert rows[0]["activity_transform"] == "log10"
    assert rows[0]["higher_is_better"] == 0


def test_get_qsar_models_filters_by_project(tmp_db):
    project_id = tmp_db.create_project("p1")
    base = {
        "dataset_name": "x.csv", "n_molecules": 5,
        "activity_label": "pIC50", "activity_transform": "none",
        "higher_is_better": 1, "cv_r2_mean": 0.5, "cv_r2_std": 0.1,
        "model_type": "RandomForestRegressor",
        "artifact_path": "p", "rdkit_version": "x", "sklearn_version": "y",
    }
    tmp_db.add_qsar_model({**base, "name": "ProjectModel", "project_id": project_id})
    tmp_db.add_qsar_model({**base, "name": "OrphanModel", "project_id": None})

    project_rows = tmp_db.get_qsar_models(project_id=project_id)
    assert len(project_rows) == 1
    assert project_rows[0]["name"] == "ProjectModel"

    all_rows = tmp_db.get_qsar_models()
    assert len(all_rows) == 2


def test_delete_qsar_model_removes_row_and_file(tmp_db, tmp_path):
    artifact_file = tmp_path / "1.joblib"
    artifact_file.write_bytes(b"fake-joblib-payload")
    meta = {
        "name": "DeleteMe", "dataset_name": "x.csv", "n_molecules": 1,
        "activity_label": "pIC50", "activity_transform": "none",
        "higher_is_better": 1, "cv_r2_mean": 0.0, "cv_r2_std": 0.0,
        "model_type": "RandomForestRegressor",
        "artifact_path": str(artifact_file),
        "rdkit_version": "x", "sklearn_version": "y", "project_id": None,
    }
    model_id = tmp_db.add_qsar_model(meta)
    assert artifact_file.exists()

    tmp_db.delete_qsar_model(model_id)
    assert tmp_db.get_qsar_models() == []
    assert not artifact_file.exists()


def test_delete_qsar_model_tolerant_of_missing_file(tmp_db):
    """Rollback path: file never written, delete must not raise."""
    meta = {
        "name": "Orphan", "dataset_name": "x.csv", "n_molecules": 1,
        "activity_label": "pIC50", "activity_transform": "none",
        "higher_is_better": 1, "cv_r2_mean": 0.0, "cv_r2_std": 0.0,
        "model_type": "RandomForestRegressor",
        "artifact_path": "/nonexistent/path/to/missing.joblib",
        "rdkit_version": "x", "sklearn_version": "y", "project_id": None,
    }
    model_id = tmp_db.add_qsar_model(meta)
    # Must not raise
    tmp_db.delete_qsar_model(model_id)
    assert tmp_db.get_qsar_models() == []
