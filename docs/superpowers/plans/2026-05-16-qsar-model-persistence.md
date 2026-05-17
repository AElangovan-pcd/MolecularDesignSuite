# QSAR Model Persistence + Predict — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist trained QSAR models to disk + DB, add a Predict tab in SAR Analysis, wire predicted activity into Drug Optimization's MPO, and backfill pytest coverage.

**Architecture:** A new module `utils/qsar.py` is the single owner of sklearn/joblib persistence (trains a RandomForestRegressor on RDKit descriptors, saves a `ModelArtifact` joblib alongside a `qsar_models` SQLite metadata row). `modules/sar_analysis.py` gets a Save Model section plus a new "QSAR Predict" tab; `modules/drug_optimization.py`'s MPO gets an optional predicted-activity desirability axis; `app.py` gets a "QSAR Models" tab in Data Management. Pytest covers the new module + backfills the existing pure-function surface.

**Tech Stack:** Python 3.11 (conda env `moldesign`), RDKit (conda-forge), scikit-learn, joblib, pandas, Streamlit, SQLite, pytest.

**Spec:** `docs/superpowers/specs/2026-05-16-qsar-model-persistence-design.md`

---

## File structure

**New files:**
- `pyproject.toml` — minimal pytest config.
- `tests/__init__.py` — empty.
- `tests/conftest.py` — fixtures (`tmp_db`, `sample_dataset_df`, `pic50_dataset_df`).
- `tests/test_rdkit_utils.py` — backfill for `utils/rdkit_utils.py`.
- `tests/test_db_manager.py` — `qsar_models` CRUD + `add_molecule` duplicate contract.
- `tests/test_qsar.py` — `utils/qsar.py`.
- `utils/qsar.py` — `ModelArtifact`, `train_qsar`, `save_model_artifact`, `load_model_artifact`, `predict`.
- `data/qsar_models/.gitkeep` — directory placeholder.

**Modified files:**
- `database/schema.sql` — add `qsar_models` table + index.
- `database/db_manager.py` — add `add_qsar_model`, `get_qsar_models`, `delete_qsar_model`.
- `modules/sar_analysis.py` — extend `_qsar_tab` with Save Model section; add `_predict_tab` and a 6th tab.
- `modules/drug_optimization.py` — add optional QSAR axis to `_mpo_tab`.
- `app.py` — add "QSAR Models" tab in `render_data_management`.
- `.gitignore` — add `data/qsar_models/*.joblib`.
- `CLAUDE.md` — note `utils/qsar.py` boundary, pytest install/run.

**Planning-time discovery:** `ModelArtifact` needs two extra fields not listed in the spec — `training_y_min: float` and `training_y_max: float` — so the MPO desirability axis can normalize a raw prediction into [0,1] without depending on the current candidate batch. This is an internal artifact detail; it doesn't change any user-visible spec contract.

---

### Task 1: Set up pytest config and shared fixtures

**Files:**
- Create: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1.1: Install pytest into the conda env**

Run (Anaconda Prompt with `moldesign` activated):

```
pip install pytest
```

Expected: `Successfully installed pytest-<version>`.

- [ ] **Step 1.2: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 1.3: Create `tests/__init__.py`** (empty file)

- [ ] **Step 1.4: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures for the Molecular Design Suite test suite."""

import pandas as pd
import pytest

from database.db_manager import DatabaseManager


@pytest.fixture
def tmp_db(tmp_path):
    """Fresh DatabaseManager backed by an isolated temp SQLite file."""
    return DatabaseManager(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def sample_dataset_df():
    """13-row dataset with valid SMILES and activity correlated to MW
    so a RandomForestRegressor can actually learn a signal.
    """
    smiles = [
        "CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
        "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "CCCCc1ccccc1",
        "CC(=O)O", "CCC(=O)O", "CCCC(=O)O",
    ]
    activity = [3.0 + 0.5 * i for i in range(len(smiles))]
    return pd.DataFrame({"SMILES": smiles, "Activity": activity})


@pytest.fixture
def pic50_dataset_df():
    """Same SMILES as sample_dataset_df with positive IC50 values in molar units,
    used to exercise the pIC50 transform path.
    """
    smiles = [
        "CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
        "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "CCCCc1ccccc1",
        "CC(=O)O", "CCC(=O)O", "CCCC(=O)O",
    ]
    # IC50 in molar units, ranging 1e-9 to 1e-5
    ic50 = [10.0 ** (-9 + 0.3 * i) for i in range(len(smiles))]
    return pd.DataFrame({"SMILES": smiles, "Activity": ic50})
```

- [ ] **Step 1.5: Smoke-run pytest**

Run:

```
pytest -v
```

Expected: `no tests ran in 0.XXs` (zero-test success, not an error).

- [ ] **Step 1.6: Commit**

```
git add pyproject.toml tests/__init__.py tests/conftest.py
git commit -m "Add pytest config and shared fixtures

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Backfill tests for `utils/rdkit_utils.py`

These tests target code that already exists. They should PASS on first run — they pin the contract for the existing pure-function surface so future refactors don't drift silently.

**Files:**
- Create: `tests/test_rdkit_utils.py`

- [ ] **Step 2.1: Write the test file**

```python
"""Backfill tests for the existing utils/rdkit_utils.py surface."""

import pytest
from rdkit import Chem

from utils.rdkit_utils import (
    mol_from_smiles, canonical_smiles, validate_smiles,
    calculate_basic_properties,
    lipinski_rule_of_five, veber_rules, ghose_filter,
    egan_rules, muegge_rules, all_drug_likeness_filters,
    tanimoto_similarity, check_pains,
)


ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
LONG_ALKANE = "C" * 60  # deliberate Lipinski violator (MW > 500)


def test_mol_from_smiles_valid():
    assert mol_from_smiles(ASPIRIN) is not None


def test_mol_from_smiles_invalid_returns_none():
    assert mol_from_smiles("not a smiles") is None
    assert mol_from_smiles("") is None
    assert mol_from_smiles("   ") is None


def test_canonical_smiles_idempotent():
    canon = canonical_smiles(ASPIRIN)
    assert canon is not None
    assert canonical_smiles(canon) == canon


def test_validate_smiles_returns_tuple():
    ok, msg = validate_smiles(ASPIRIN)
    assert ok is True
    assert isinstance(msg, str)

    bad, msg = validate_smiles("xyz123")
    assert bad is False
    assert isinstance(msg, str)


def test_calculate_basic_properties_aspirin():
    props = calculate_basic_properties(mol_from_smiles(ASPIRIN))
    expected_keys = {
        "molecular_weight", "exact_mass", "formula", "logP", "tpsa",
        "hbd", "hba", "rotatable_bonds", "aromatic_rings", "rings",
        "heavy_atoms", "fraction_csp3", "num_heteroatoms",
        "molar_refractivity", "qed",
    }
    assert expected_keys.issubset(props.keys())
    # Aspirin MW = 180.16 g/mol
    assert abs(props["molecular_weight"] - 180.16) < 1.0
    assert props["formula"] == "C9H8O4"


def test_lipinski_pass_and_fail():
    aspirin = lipinski_rule_of_five(mol_from_smiles(ASPIRIN))
    assert aspirin["passes"] is True
    assert aspirin["violations"] == 0

    fail = lipinski_rule_of_five(mol_from_smiles(LONG_ALKANE))
    assert fail["passes"] is False


def test_veber_rules_aspirin_passes():
    result = veber_rules(mol_from_smiles(ASPIRIN))
    assert result["passes"] is True


def test_ghose_filter_aspirin():
    # Aspirin MW is below Ghose minimum (160), so this should fail on MW
    result = ghose_filter(mol_from_smiles(ASPIRIN))
    assert result["160 <= MW <= 480"] is True  # 180 is in range
    assert "passes" in result


def test_egan_rules_aspirin_passes():
    result = egan_rules(mol_from_smiles(ASPIRIN))
    assert result["passes"] is True


def test_muegge_rules_aspirin():
    result = muegge_rules(mol_from_smiles(ASPIRIN))
    assert "passes" in result
    assert isinstance(result["passes"], bool)


def test_all_drug_likeness_filters_returns_all_five():
    result = all_drug_likeness_filters(mol_from_smiles(ASPIRIN))
    assert set(result.keys()) == {"Lipinski", "Veber", "Ghose", "Egan", "Muegge"}
    for name, checks in result.items():
        assert "passes" in checks


def test_tanimoto_self_similarity_is_one():
    mol = mol_from_smiles(ASPIRIN)
    assert tanimoto_similarity(mol, mol) == 1.0


def test_tanimoto_different_molecules_below_one():
    mol1 = mol_from_smiles(ASPIRIN)
    mol2 = mol_from_smiles("c1ccccc1")  # benzene
    sim = tanimoto_similarity(mol1, mol2)
    assert 0.0 <= sim < 1.0


def test_check_pains_aspirin_clean():
    # Aspirin is not a PAINS hit
    assert check_pains(mol_from_smiles(ASPIRIN)) == []


def test_check_pains_known_alert():
    # Quinone / catechol style structures often hit PAINS catalogs.
    # 1,2-dihydroxybenzene (catechol) hits the catechol_A PAINS pattern.
    catechol = mol_from_smiles("Oc1ccccc1O")
    matches = check_pains(catechol)
    assert len(matches) >= 1
```

- [ ] **Step 2.2: Run the tests**

```
pytest tests/test_rdkit_utils.py -v
```

Expected: all tests PASS. If `test_check_pains_known_alert` fails because catechol isn't in the local PAINS catalog, replace `"Oc1ccccc1O"` with another known PAINS hit: `"O=C1C=CC(=O)C=C1"` (benzoquinone). Re-run.

- [ ] **Step 2.3: Commit**

```
git add tests/test_rdkit_utils.py
git commit -m "Backfill tests for utils/rdkit_utils.py pure-function surface

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Backfill test for `DatabaseManager.add_molecule` duplicate contract

The `INSERT OR IGNORE` + lookup pattern is a load-bearing contract — callers depend on "duplicate returns existing id, never raises." Pin it before adding new methods on top.

**Files:**
- Create: `tests/test_db_manager.py`

- [ ] **Step 3.1: Write the test file**

```python
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
```

- [ ] **Step 3.2: Run the tests**

```
pytest tests/test_db_manager.py -v
```

Expected: all PASS (these test existing code).

- [ ] **Step 3.3: Commit**

```
git add tests/test_db_manager.py
git commit -m "Pin add_molecule INSERT OR IGNORE duplicate contract with tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add `qsar_models` table to schema and DB methods (TDD)

**Files:**
- Modify: `database/schema.sql` (append at end, before the indexes section)
- Modify: `database/db_manager.py` (add three methods)
- Modify: `tests/test_db_manager.py` (add tests)

- [ ] **Step 4.1: Write failing tests in `tests/test_db_manager.py`**

Append to the existing file:

```python
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
```

- [ ] **Step 4.2: Run the failing tests**

```
pytest tests/test_db_manager.py -v
```

Expected: 5 new tests FAIL with `AttributeError: 'DatabaseManager' object has no attribute 'add_qsar_model'`.

- [ ] **Step 4.3: Extend `database/schema.sql`**

Append before the `-- Indexes for performance` line block at the bottom:

```sql
CREATE TABLE IF NOT EXISTS qsar_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    dataset_name TEXT,
    n_molecules INTEGER NOT NULL,
    activity_label TEXT NOT NULL,
    activity_transform TEXT DEFAULT 'none',
    higher_is_better INTEGER NOT NULL,
    cv_r2_mean REAL,
    cv_r2_std REAL,
    model_type TEXT DEFAULT 'RandomForestRegressor',
    artifact_path TEXT NOT NULL,
    rdkit_version TEXT,
    sklearn_version TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    project_id INTEGER,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
);
```

Then add a new index line in the `-- Indexes for performance` block:

```sql
CREATE INDEX IF NOT EXISTS idx_qsar_models_project ON qsar_models(project_id);
```

- [ ] **Step 4.4: Add the three methods to `database/db_manager.py`**

Add to the top of the file (after `from typing import Optional`):

```python
from pathlib import Path as _Path
```

Then append at the end of the `DatabaseManager` class (after the `get_sar_dataset_molecules` method, before any closing-of-file content):

```python
    # ── QSAR Models ───────────────────────────────────────────

    def add_qsar_model(self, meta: dict) -> int:
        """Insert a qsar_models row from a metadata dict. Returns new id.

        Expected keys: name, dataset_name, n_molecules, activity_label,
        activity_transform, higher_is_better, cv_r2_mean, cv_r2_std,
        model_type, artifact_path, rdkit_version, sklearn_version, project_id.
        """
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """INSERT INTO qsar_models
                   (name, dataset_name, n_molecules, activity_label,
                    activity_transform, higher_is_better, cv_r2_mean, cv_r2_std,
                    model_type, artifact_path, rdkit_version, sklearn_version,
                    project_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    meta["name"], meta.get("dataset_name"), meta["n_molecules"],
                    meta["activity_label"], meta.get("activity_transform", "none"),
                    int(meta["higher_is_better"]),
                    meta.get("cv_r2_mean"), meta.get("cv_r2_std"),
                    meta.get("model_type", "RandomForestRegressor"),
                    meta["artifact_path"],
                    meta.get("rdkit_version"), meta.get("sklearn_version"),
                    meta.get("project_id"),
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_qsar_models(self, project_id: Optional[int] = None) -> list[dict]:
        conn = self._get_connection()
        try:
            if project_id is not None:
                rows = conn.execute(
                    "SELECT * FROM qsar_models WHERE project_id = ? ORDER BY created_date DESC",
                    (project_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM qsar_models ORDER BY created_date DESC"
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_qsar_model(self, model_id: int) -> None:
        """Delete the qsar_models row and unlink the artifact file.
        Tolerant of a missing file so it's safe as a rollback path.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT artifact_path FROM qsar_models WHERE id = ?", (model_id,)
            ).fetchone()
            conn.execute("DELETE FROM qsar_models WHERE id = ?", (model_id,))
            conn.commit()
        finally:
            conn.close()
        if row is not None and row["artifact_path"]:
            _Path(row["artifact_path"]).unlink(missing_ok=True)
```

- [ ] **Step 4.5: Re-run the tests**

```
pytest tests/test_db_manager.py -v
```

Expected: all 8 tests PASS (3 existing + 5 new).

- [ ] **Step 4.6: Commit**

```
git add database/schema.sql database/db_manager.py tests/test_db_manager.py
git commit -m "Add qsar_models table and CRUD methods

Adds qsar_models schema (auto-applied on next DatabaseManager init),
plus add_qsar_model/get_qsar_models/delete_qsar_model. delete is
tolerant of missing artifact files so it's safe as a save-rollback path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Create `utils/qsar.py` skeleton + `train_qsar` (TDD)

**Files:**
- Create: `tests/test_qsar.py`
- Create: `utils/qsar.py`

- [ ] **Step 5.1: Write failing tests in `tests/test_qsar.py`**

```python
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
```

- [ ] **Step 5.2: Run the failing tests**

```
pytest tests/test_qsar.py -v
```

Expected: collection error (module `utils.qsar` does not exist).

- [ ] **Step 5.3: Create `utils/qsar.py`**

```python
"""QSAR model training, persistence, and prediction.

Single owner of joblib and sklearn-persistence imports in this codebase.
Consumed by modules/sar_analysis.py and modules/drug_optimization.py.
"""

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

import rdkit
from rdkit import Chem

from utils.rdkit_utils import mol_from_smiles, calculate_descriptor_set, canonical_smiles


VALID_TRANSFORMS = ("none", "log10", "pIC50")


@dataclass
class ModelArtifact:
    """In-memory bundle of everything needed to predict from a trained QSAR model."""
    model: Any
    scaler: Any
    feature_columns: list[str]
    training_smiles_hashes: frozenset[str]
    training_y_min: float
    training_y_max: float
    rdkit_version: str
    sklearn_version: str


def _hash_smiles(smiles: str) -> str:
    """Canonical-SMILES md5; used as a stable training-set membership marker."""
    canon = canonical_smiles(smiles) or smiles
    return hashlib.md5(canon.encode("utf-8")).hexdigest()


def _apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
    """Apply the activity transform to a numeric array."""
    if transform == "none":
        return values
    if transform == "log10":
        if (values <= 0).any():
            raise ValueError("log10 transform requires positive activity values")
        return np.log10(values)
    if transform == "pIC50":
        if (values <= 0).any():
            raise ValueError("pIC50 transform requires positive IC50 values (molar units)")
        return -np.log10(values)
    raise ValueError(f"Unknown activity_transform: {transform!r} (must be one of {VALID_TRANSFORMS})")


def train_qsar(
    df: pd.DataFrame,
    smiles_col: str,
    activity_col: str,
    activity_transform: str = "none",
) -> tuple[ModelArtifact, dict]:
    """Train a RandomForestRegressor on RDKit descriptors of SMILES + activity.

    Returns (artifact, metrics) where metrics is {cv_r2_mean, cv_r2_std, n_molecules}.
    Raises ValueError if no valid molecules survive parsing, or if the transform
    encounters non-positive values.
    """
    if activity_transform not in VALID_TRANSFORMS:
        raise ValueError(f"Unknown activity_transform: {activity_transform!r}")

    # Parse SMILES, compute descriptors, drop invalid rows
    descriptor_rows = []
    activities = []
    smiles_kept = []
    for _, row in df.iterrows():
        smi = str(row[smiles_col])
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        desc = calculate_descriptor_set(mol)
        descriptor_rows.append(desc)
        activities.append(float(row[activity_col]))
        smiles_kept.append(smi)

    if not descriptor_rows:
        raise ValueError("No valid molecules in training set")

    desc_df = pd.DataFrame(descriptor_rows)
    feature_columns = list(desc_df.columns)
    X = desc_df[feature_columns].values
    y_raw = np.asarray(activities, dtype=float)
    y = _apply_transform(y_raw, activity_transform)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # 5-fold CV (or fewer folds if dataset is tiny)
    cv = min(5, len(y))
    if cv < 2:
        cv_mean = float("nan")
        cv_std = float("nan")
        cv_predictions = np.full_like(y, np.nan, dtype=float)
    else:
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")
        cv_mean = float(scores.mean())
        cv_std = float(scores.std())
        cv_predictions = cross_val_predict(model, X_scaled, y, cv=cv)

    model.fit(X_scaled, y)

    artifact = ModelArtifact(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        training_smiles_hashes=frozenset(_hash_smiles(s) for s in smiles_kept),
        training_y_min=float(y.min()),
        training_y_max=float(y.max()),
        rdkit_version=rdkit.__version__,
        sklearn_version=sklearn.__version__,
    )
    metrics = {
        "cv_r2_mean": cv_mean,
        "cv_r2_std": cv_std,
        "n_molecules": len(smiles_kept),
        "cv_y_actual": y,
        "cv_y_predicted": cv_predictions,
    }
    return artifact, metrics
```

- [ ] **Step 5.4: Run the tests**

```
pytest tests/test_qsar.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5.5: Commit**

```
git add utils/qsar.py tests/test_qsar.py
git commit -m "Add utils/qsar.py with train_qsar and ModelArtifact

Trains a RandomForestRegressor on RDKit descriptors with optional
log10 or pIC50 activity transforms. Records training SMILES hashes
and y-range on the artifact for downstream leak-check and MPO
normalization.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Add `save_model_artifact` / `load_model_artifact` round-trip (TDD)

**Files:**
- Modify: `utils/qsar.py`
- Modify: `tests/test_qsar.py`

- [ ] **Step 6.1: Append failing tests to `tests/test_qsar.py`**

```python
from utils.qsar import save_model_artifact, load_model_artifact


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
```

- [ ] **Step 6.2: Run the failing tests**

```
pytest tests/test_qsar.py -v
```

Expected: 4 new tests FAIL with ImportError or similar.

- [ ] **Step 6.3: Add `save_model_artifact` and `load_model_artifact` to `utils/qsar.py`**

Append at the end of `utils/qsar.py`:

```python
def save_model_artifact(
    artifact: ModelArtifact,
    meta: dict,
    db,
    models_dir: Path = Path("data/qsar_models"),
) -> int:
    """Persist a ModelArtifact to disk and insert its metadata row.

    `meta` is the user/UI-supplied subset:
        name, dataset_name, activity_label, activity_transform,
        higher_is_better, cv_r2_mean, cv_r2_std, project_id.
    The artifact-derived fields (n_molecules, model_type, versions,
    artifact_path) are filled in here.

    Rolls back the DB row if joblib.dump fails, so callers don't get
    orphan rows pointing at nonexistent files.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Insert DB row first (we need the id to name the file).
    full_meta = {
        **meta,
        "n_molecules": len(artifact.training_smiles_hashes),
        "model_type": type(artifact.model).__name__,
        "rdkit_version": artifact.rdkit_version,
        "sklearn_version": artifact.sklearn_version,
        "artifact_path": "",  # placeholder, updated after dump succeeds
    }
    model_id = db.add_qsar_model(full_meta)
    artifact_path = models_dir / f"{model_id}.joblib"

    try:
        joblib.dump(artifact, artifact_path)
    except Exception:
        # Roll back the DB row; delete_qsar_model is tolerant of missing files.
        db.delete_qsar_model(model_id)
        raise

    # Patch the artifact_path on the row to the real location.
    conn = db._get_connection()
    try:
        conn.execute(
            "UPDATE qsar_models SET artifact_path = ? WHERE id = ?",
            (str(artifact_path), model_id),
        )
        conn.commit()
    finally:
        conn.close()

    return model_id


def load_model_artifact(
    model_id: int,
    db,
    models_dir: Path = Path("data/qsar_models"),
) -> ModelArtifact:
    """Load a saved ModelArtifact from disk by DB id.

    Raises FileNotFoundError if the .joblib is gone (e.g. user deleted it
    or the data/ directory was not copied alongside the DB).
    """
    models_dir = Path(models_dir)
    artifact_path = models_dir / f"{model_id}.joblib"
    if not artifact_path.exists():
        # Fall back to the recorded path on the row, in case models_dir differs.
        rows = [r for r in db.get_qsar_models() if r["id"] == model_id]
        if rows and rows[0]["artifact_path"]:
            artifact_path = Path(rows[0]["artifact_path"])
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"QSAR model artifact missing on disk: {artifact_path}"
        )
    return joblib.load(artifact_path)
```

- [ ] **Step 6.4: Run the tests**

```
pytest tests/test_qsar.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 6.5: Commit**

```
git add utils/qsar.py tests/test_qsar.py
git commit -m "Add save_model_artifact / load_model_artifact round-trip

save_model_artifact rolls back the DB row if joblib.dump fails so
callers never get orphan rows. load_model_artifact raises
FileNotFoundError if the .joblib is gone, with a fallback to the
recorded artifact_path on the metadata row.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Add `predict` function with invalid-SMILES handling + leak flag (TDD)

**Files:**
- Modify: `utils/qsar.py`
- Modify: `tests/test_qsar.py`

- [ ] **Step 7.1: Append failing tests to `tests/test_qsar.py`**

```python
from utils.qsar import predict


def test_predict_valid_smiles_returns_floats(sample_dataset_df):
    artifact, _ = train_qsar(sample_dataset_df, "SMILES", "Activity")
    results = predict(artifact, ["CCO", "CCCO"])
    assert len(results) == 2
    for r in results:
        assert isinstance(r["predicted_value"], float)
        assert r["error"] is None
        assert "smiles" in r
        assert "in_training" in r


def test_predict_invalid_smiles_returns_none_and_error(sample_dataset_df):
    artifact, _ = train_qsar(sample_dataset_df, "SMILES", "Activity")
    results = predict(artifact, ["CCO", "not-a-smiles", "CCCO"])
    assert results[0]["predicted_value"] is not None
    assert results[1]["predicted_value"] is None
    assert "Invalid SMILES" in results[1]["error"]
    assert results[2]["predicted_value"] is not None


def test_predict_flags_training_set_members(sample_dataset_df):
    artifact, _ = train_qsar(sample_dataset_df, "SMILES", "Activity")
    # "CCO" is in sample_dataset_df, "CCCCCCCCO" (octanol) is not.
    results = predict(artifact, ["CCO", "CCCCCCCCO"])
    assert results[0]["in_training"] is True
    assert results[1]["in_training"] is False


def test_predict_empty_list_returns_empty(sample_dataset_df):
    artifact, _ = train_qsar(sample_dataset_df, "SMILES", "Activity")
    assert predict(artifact, []) == []
```

- [ ] **Step 7.2: Run the failing tests**

```
pytest tests/test_qsar.py -v
```

Expected: 4 new tests FAIL with ImportError.

- [ ] **Step 7.3: Append `predict` to `utils/qsar.py`**

```python
def predict(artifact: ModelArtifact, smiles_list: list[str]) -> list[dict]:
    """Predict activity for each SMILES. Never raises on per-row errors.

    Returns list of dicts: {smiles, predicted_value, in_training, error}.
    `predicted_value` is a float for valid SMILES, None for invalid.
    `in_training` flags whether the canonical SMILES was in the training set.
    """
    results: list[dict] = []
    for smi in smiles_list:
        row: dict = {
            "smiles": smi,
            "predicted_value": None,
            "in_training": False,
            "error": None,
        }
        mol = mol_from_smiles(smi)
        if mol is None:
            row["error"] = "Invalid SMILES"
            results.append(row)
            continue
        try:
            desc = calculate_descriptor_set(mol)
            desc_df = pd.DataFrame([desc])[artifact.feature_columns]
            X = artifact.scaler.transform(desc_df.values)
            pred = float(artifact.model.predict(X)[0])
            row["predicted_value"] = pred
            row["in_training"] = _hash_smiles(smi) in artifact.training_smiles_hashes
        except Exception as e:
            row["error"] = f"Prediction failed: {e}"
        results.append(row)
    return results
```

- [ ] **Step 7.4: Run the tests**

```
pytest tests/test_qsar.py -v
```

Expected: all 16 tests PASS.

- [ ] **Step 7.5: Commit**

```
git add utils/qsar.py tests/test_qsar.py
git commit -m "Add predict() with per-row error handling and training-set leak flag

Batch prediction never raises on invalid SMILES — bad inputs get
predicted_value=None and a descriptive error string. Each result
also carries an in_training flag from a canonical-SMILES hash check
against the training set.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Wire Save Model section into `modules/sar_analysis.py::_qsar_tab`

No pytest for Streamlit UI — verify manually by launching the app at the end of the task.

**Files:**
- Modify: `modules/sar_analysis.py`

- [ ] **Step 8.1: Update the imports at the top of `modules/sar_analysis.py`**

Replace the existing `from database.db_manager import DatabaseManager` line with:

```python
from database.db_manager import DatabaseManager
from utils.qsar import (
    train_qsar, save_model_artifact, load_model_artifact, predict, VALID_TRANSFORMS,
)
```

- [ ] **Step 8.2: Refactor `_qsar_tab` to use the new helpers and add a Save Model section**

Replace the entire `_qsar_tab` function (currently ~lines 248–335) with:

```python
def _qsar_tab(db: DatabaseManager):
    """Simple QSAR model building with persistence."""
    st.subheader("QSAR Modeling")
    st.caption("Build a QSAR model from molecular descriptors and activity data, then save it for later prediction.")

    st.markdown("#### Input Data")
    st.markdown("Upload a CSV with columns: `SMILES`, `Activity`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="qsar_csv")
    if uploaded is None:
        st.code("SMILES,Activity\nCCO,3.5\nc1ccccc1,5.2\nCC(=O)O,2.1", language="csv")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    smiles_col = None
    activity_col = None
    for col in df.columns:
        if col.lower() in ("smiles", "smi", "molecule"):
            smiles_col = col
        if col.lower() in ("activity", "pic50", "ic50", "ec50", "potency", "value"):
            activity_col = col
    if smiles_col is None:
        smiles_col = st.selectbox("SMILES column", df.columns.tolist(), key="qsar_smiles_col")
    if activity_col is None:
        activity_col = st.selectbox(
            "Activity column",
            [c for c in df.columns if c != smiles_col],
            key="qsar_activity_col",
        )

    transform = st.radio(
        "Activity transform (applied to the activity column before training)",
        VALID_TRANSFORMS,
        horizontal=True,
        key="qsar_transform",
        help="'pIC50' computes -log10(IC50 in molar units). 'log10' computes log10(value). 'none' uses values as-is.",
    )

    if not st.button("Build QSAR Model", key="qsar_build_btn"):
        return

    try:
        artifact, metrics = train_qsar(df, smiles_col, activity_col, activity_transform=transform)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Training failed: {e}")
        return

    st.session_state["qsar_last_artifact"] = artifact
    st.session_state["qsar_last_metrics"] = metrics
    st.session_state["qsar_last_dataset_name"] = uploaded.name
    st.session_state["qsar_last_transform"] = transform

    st.markdown("#### Model Performance (5-fold CV)")
    st.metric("Mean R²", f"{metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
    st.caption(f"Trained on {metrics['n_molecules']} valid molecules")

    importances = pd.DataFrame({
        "Feature": artifact.feature_columns,
        "Importance": artifact.model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    st.markdown("#### Top Feature Importances")
    st.dataframe(importances.head(10), hide_index=True, use_container_width=True)

    # Predicted vs actual comes from cross_val_predict run inside train_qsar.
    pred_df = pd.DataFrame({
        "Actual": metrics["cv_y_actual"],
        "Predicted": metrics["cv_y_predicted"],
    })
    st.plotly_chart(
        scatter_plot(pred_df, "Actual", "Predicted", title="Predicted vs Actual Activity"),
        use_container_width=True,
    )

    # ── Save Model ───────────────────────────────────────────
    st.divider()
    st.markdown("#### Save This Model")

    default_name = uploaded.name.rsplit(".", 1)[0] + "-RF"
    save_col1, save_col2 = st.columns(2)
    with save_col1:
        save_name = st.text_input("Model name", value=default_name, key="qsar_save_name")
        save_activity_label = st.text_input(
            "Activity label (required)",
            placeholder="e.g., pIC50, log(EC50) [nM], % inhibition",
            key="qsar_save_label",
        )
    with save_col2:
        higher_is_better = st.checkbox(
            "Higher activity = better potency?",
            value=True,
            key="qsar_save_higher_better",
            help="Used by Drug Optimization MPO to know whether to maximize or minimize predictions.",
        )

    if st.button("Save Model", key="qsar_save_btn", type="primary"):
        if not save_activity_label.strip():
            st.error("Activity label is required.")
            return
        if "qsar_last_artifact" not in st.session_state:
            st.error("Train a model first.")
            return
        meta = {
            "name": save_name.strip() or default_name,
            "dataset_name": st.session_state["qsar_last_dataset_name"],
            "activity_label": save_activity_label.strip(),
            "activity_transform": st.session_state["qsar_last_transform"],
            "higher_is_better": 1 if higher_is_better else 0,
            "cv_r2_mean": st.session_state["qsar_last_metrics"]["cv_r2_mean"],
            "cv_r2_std": st.session_state["qsar_last_metrics"]["cv_r2_std"],
            "project_id": st.session_state.get("current_project_id"),
        }
        try:
            model_id = save_model_artifact(
                st.session_state["qsar_last_artifact"], meta, db,
            )
            st.success(f"Saved as model #{model_id}: {meta['name']}")
        except Exception as e:
            st.error(f"Failed to save model: {e}")
```

- [ ] **Step 8.3: Sanity-run the tests (no regressions)**

```
pytest -v
```

Expected: all tests still PASS.

- [ ] **Step 8.4: Manual smoke test**

Launch the app:

```
run.bat
```

(or `streamlit run app.py --server.port 8501 --server.headless true`)

1. Open <http://localhost:8501>.
2. Navigate to "SAR Analysis" → "QSAR Modeling".
3. Prepare a tiny CSV (paste into a file `smoke.csv`):
   ```
   SMILES,Activity
   CCO,3.0
   CCCO,3.5
   CCCCO,4.0
   CCCCCO,4.5
   CCCCCCO,5.0
   c1ccccc1,5.5
   Cc1ccccc1,6.0
   CCc1ccccc1,6.5
   CCCc1ccccc1,7.0
   CCCCc1ccccc1,7.5
   ```
4. Upload it, leave transform = `none`, click "Build QSAR Model".
5. Verify R² + feature importances + scatter render.
6. Fill in "Activity label" = `pIC50`, leave Higher-is-better checked, click "Save Model".
7. Verify `st.success` with a model id, and `data/qsar_models/<id>.joblib` exists on disk.
8. Stop the app (Ctrl+C).

- [ ] **Step 8.5: Commit**

```
git add modules/sar_analysis.py
git commit -m "Wire Save Model section into SAR Analysis QSAR tab

Refactors _qsar_tab to use utils.qsar.train_qsar and adds a 'Save This
Model' section that captures activity label + direction and persists via
save_model_artifact under the active project.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Add "QSAR Predict" as a 6th tab in SAR Analysis

**Files:**
- Modify: `modules/sar_analysis.py`

- [ ] **Step 9.1: Update the `render_sar_analysis` function**

Replace the `tab1, tab2, tab3, tab4, tab5 = st.tabs([...])` block (and its `with tabN:` callers) with:

```python
def render_sar_analysis(db: DatabaseManager):
    """Render the SAR Analysis module."""
    st.header("Structure-Activity Relationship Analysis")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Molecular Descriptors", "Similarity Analysis",
        "Scaffold Analysis", "Chemical Space",
        "QSAR Modeling", "QSAR Predict",
    ])

    with tab1:
        _descriptors_tab(db)
    with tab2:
        _similarity_tab(db)
    with tab3:
        _scaffold_tab(db)
    with tab4:
        _chemical_space_tab(db)
    with tab5:
        _qsar_tab(db)
    with tab6:
        _predict_tab(db)
```

- [ ] **Step 9.2: Add `_predict_tab` function**

Append to `modules/sar_analysis.py` (after `_qsar_tab`):

```python
def _predict_tab(db: DatabaseManager):
    """Predict activity using a saved QSAR model."""
    st.subheader("QSAR Predict")
    st.caption("Pick a saved model and score new SMILES.")

    project_id = st.session_state.get("current_project_id")
    models = db.get_qsar_models(project_id=project_id)
    if not models:
        st.info(
            "No saved QSAR models for this project. "
            "Train one in the 'QSAR Modeling' tab and click 'Save Model'."
        )
        return

    def _label(m: dict) -> str:
        r2 = m.get("cv_r2_mean")
        r2_str = f"R²={r2:.2f}" if r2 is not None else "R²=n/a"
        return f"#{m['id']} {m['name']} ({r2_str}, {m['created_date']})"

    options = {_label(m): m for m in models}
    chosen_label = st.selectbox("Saved model", list(options.keys()), key="predict_model_select")
    chosen = options[chosen_label]

    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        st.metric("Activity label", chosen["activity_label"])
    with meta_col2:
        st.metric("Higher is better", "Yes" if chosen["higher_is_better"] else "No")
    with meta_col3:
        st.metric("Training n", chosen["n_molecules"])
    if chosen.get("activity_transform") and chosen["activity_transform"] != "none":
        st.caption(f"Predictions are in *{chosen['activity_transform']}-transformed* space.")

    source = st.radio("Input", ["Paste SMILES", "From Database"], horizontal=True, key="predict_source")
    smiles_list: list[str] = []
    if source == "Paste SMILES":
        text = st.text_area(
            "SMILES (one per line)",
            height=150,
            key="predict_text",
            placeholder="CCO\nc1ccccc1\nCC(=O)O",
        )
        if text:
            smiles_list = [line.strip() for line in text.strip().split("\n") if line.strip()]
    else:
        db_mols = db.get_molecules(project_id=project_id, limit=500)
        if not db_mols:
            st.info("No molecules in the database for this project.")
            return
        labels = {f"{m['name'] or m['canonical_smiles'][:25]} (#{m['id']})": m["smiles"] for m in db_mols}
        chosen_dbs = st.multiselect("Molecules", list(labels.keys()), key="predict_db_select")
        smiles_list = [labels[c] for c in chosen_dbs]

    if not smiles_list:
        return

    if not st.button("Predict", key="predict_btn", type="primary"):
        return

    try:
        artifact = load_model_artifact(chosen["id"], db)
    except FileNotFoundError:
        st.error(
            "Model artifact missing on disk. "
            "Delete this row from Data Management → QSAR Models, then re-train."
        )
        return

    # Version skew warning, non-blocking
    import rdkit, sklearn
    if chosen.get("rdkit_version") and chosen["rdkit_version"] != rdkit.__version__:
        st.warning(
            f"Model was trained with RDKit {chosen['rdkit_version']}, "
            f"current is {rdkit.__version__}. Predictions may differ slightly."
        )
    if chosen.get("sklearn_version") and chosen["sklearn_version"] != sklearn.__version__:
        st.warning(
            f"Model was trained with scikit-learn {chosen['sklearn_version']}, "
            f"current is {sklearn.__version__}. Predictions may differ slightly."
        )

    results = predict(artifact, smiles_list)
    pred_col_label = f"Predicted {chosen['activity_label']}"
    results_df = pd.DataFrame([
        {
            "SMILES": r["smiles"],
            pred_col_label: r["predicted_value"],
            "in_training?": "yes ⚠" if r["in_training"] else "no",
            "error": r["error"] or "",
        }
        for r in results
    ])
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    csv = results_df.to_csv(index=False)
    st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
```

- [ ] **Step 9.3: Sanity-run the tests**

```
pytest -v
```

Expected: all tests still PASS.

- [ ] **Step 9.4: Manual smoke test**

Launch the app, go to SAR Analysis → QSAR Predict. The saved model from Task 8's smoke test should appear in the dropdown.

1. Paste `CCO\nCCCCCCCCO\nnot-a-smiles\n` and click Predict.
2. Verify the table shows: `CCO` with `in_training? = yes ⚠`, `CCCCCCCCO` with `in_training? = no`, `not-a-smiles` with empty prediction and an error.

- [ ] **Step 9.5: Commit**

```
git add modules/sar_analysis.py
git commit -m "Add QSAR Predict tab in SAR Analysis

New 6th tab loads a saved QSAR model, accepts pasted SMILES or
DB-stored molecules, and renders predictions with an in-training
leak flag and per-row error messages. Surfaces RDKit/sklearn
version skew as non-blocking warnings.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Wire optional QSAR axis into Drug Optimization MPO

**Files:**
- Modify: `modules/drug_optimization.py`

- [ ] **Step 10.1: Update imports**

Replace the existing `from database.db_manager import DatabaseManager` line with:

```python
from database.db_manager import DatabaseManager
from utils.qsar import load_model_artifact, predict as qsar_predict
```

- [ ] **Step 10.2: Replace `_mpo_tab` with the extended version**

Find `_mpo_tab` (currently ~lines 322–402) and replace the function body with:

```python
def _mpo_tab(db: DatabaseManager):
    """Multi-parameter optimization with optional QSAR-predicted activity axis."""
    st.subheader("Multi-Parameter Optimization (MPO)")
    st.caption("Score molecules against multiple property targets using desirability functions.")

    molecules = []
    source = st.radio("Source", ["Database", "Paste SMILES"], horizontal=True, key="mpo_source")
    if source == "Database":
        db_mols = db.get_molecules(limit=200)
        for m in db_mols:
            mol = mol_from_smiles(m["smiles"])
            if mol:
                molecules.append({"name": m["name"] or m["smiles"][:20],
                                  "smiles": m["smiles"], "mol": mol})
    else:
        text = st.text_area("SMILES (one per line)", height=100, key="mpo_text")
        if text:
            for line in text.strip().split("\n"):
                parts = line.strip().split("\t")
                smi = parts[0].strip()
                name = parts[1].strip() if len(parts) > 1 else smi[:20]
                mol = mol_from_smiles(smi)
                if mol:
                    molecules.append({"name": name, "smiles": smi, "mol": mol})

    if not molecules:
        st.info("Load molecules for MPO analysis.")
        return

    # Define target property ranges
    st.markdown("#### Define Property Targets")
    st.caption("Set ideal ranges for each property (molecules score higher within range).")

    col1, col2, col3 = st.columns(3)
    with col1:
        mw_range = st.slider("MW", 100.0, 800.0, (200.0, 500.0), key="mpo_mw")
        logp_range = st.slider("LogP", -3.0, 8.0, (0.0, 4.0), key="mpo_logp")
    with col2:
        tpsa_range = st.slider("TPSA", 0.0, 250.0, (40.0, 130.0), key="mpo_tpsa")
        hbd_max = st.slider("Max HBD", 0, 10, 5, key="mpo_hbd")
    with col3:
        hba_max = st.slider("Max HBA", 0, 15, 10, key="mpo_hba")
        qed_min = st.slider("Min QED", 0.0, 1.0, 0.4, key="mpo_qed")

    # ── Optional QSAR axis ────────────────────────────────────
    st.markdown("#### QSAR Predicted Activity (optional)")
    project_id = st.session_state.get("current_project_id")
    saved_models = db.get_qsar_models(project_id=project_id)
    qsar_options = {"None": None}
    for m in saved_models:
        r2 = m.get("cv_r2_mean")
        label = f"#{m['id']} {m['name']} (R²={r2:.2f})" if r2 is not None else f"#{m['id']} {m['name']}"
        qsar_options[label] = m
    qsar_choice_label = st.selectbox(
        "Use QSAR model", list(qsar_options.keys()), key="mpo_qsar_model_select"
    )
    qsar_meta = qsar_options[qsar_choice_label]
    qsar_weight = 0.0
    if qsar_meta is not None:
        qsar_weight = st.slider(
            "QSAR axis weight (relative to other axes)",
            0.0, 1.0, 0.5, 0.1, key="mpo_qsar_weight",
        )

    if not st.button("Calculate MPO Scores", key="mpo_calc_btn"):
        return

    # If a QSAR model is selected, predict once for the whole batch.
    qsar_predictions: dict[str, Optional[float]] = {}
    if qsar_meta is not None:
        try:
            artifact = load_model_artifact(qsar_meta["id"], db)
        except FileNotFoundError:
            st.error(
                "QSAR model artifact missing on disk. Delete it from Data Management or pick another."
            )
            return
        results = qsar_predict(artifact, [md["smiles"] for md in molecules])
        for r in results:
            qsar_predictions[r["smiles"]] = r["predicted_value"]
        y_min = artifact.training_y_min
        y_max = artifact.training_y_max
        y_span = (y_max - y_min) if y_max > y_min else 1.0
        higher_is_better = bool(qsar_meta["higher_is_better"])

    rows = []
    for md in molecules:
        props = calculate_basic_properties(md["mol"])
        scores = {}
        mw = props["molecular_weight"]
        scores["MW"] = 1.0 if mw_range[0] <= mw <= mw_range[1] else max(0, 1 - abs(mw - np.mean(mw_range)) / 200)
        logp = props["logP"]
        scores["LogP"] = 1.0 if logp_range[0] <= logp <= logp_range[1] else max(0, 1 - abs(logp - np.mean(logp_range)) / 3)
        tpsa = props["tpsa"]
        scores["TPSA"] = 1.0 if tpsa_range[0] <= tpsa <= tpsa_range[1] else max(0, 1 - abs(tpsa - np.mean(tpsa_range)) / 80)
        scores["HBD"] = 1.0 if props["hbd"] <= hbd_max else max(0, 1 - (props["hbd"] - hbd_max) / 3)
        scores["HBA"] = 1.0 if props["hba"] <= hba_max else max(0, 1 - (props["hba"] - hba_max) / 5)
        scores["QED"] = 1.0 if props["qed"] >= qed_min else props["qed"] / qed_min

        # Equal-weighted mean of the property axes (existing behavior)
        property_score = float(np.mean(list(scores.values())))

        row = {"Name": md["name"], "SMILES": md["smiles"]}

        if qsar_meta is not None:
            pred = qsar_predictions.get(md["smiles"])
            if pred is None:
                # Invalid SMILES for the QSAR model — score this axis as 0
                qsar_desirability = 0.0
                row[f"Predicted {qsar_meta['activity_label']}"] = None
            else:
                norm = (pred - y_min) / y_span
                norm = max(0.0, min(1.0, norm))
                qsar_desirability = norm if higher_is_better else (1.0 - norm)
                row[f"Predicted {qsar_meta['activity_label']}"] = round(pred, 3)
            row["QSAR Score"] = round(qsar_desirability, 3)
            # Weighted combination: w * qsar + (1 - w) * property_score
            mpo_score = qsar_weight * qsar_desirability + (1.0 - qsar_weight) * property_score
        else:
            mpo_score = property_score

        row["MPO Score"] = round(mpo_score, 3)
        row.update({f"{k} Score": round(v, 3) for k, v in scores.items()})
        row.update({"MW": mw, "LogP": logp, "TPSA": tpsa,
                    "HBD": props["hbd"], "HBA": props["hba"],
                    "QED": props["qed"]})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("MPO Score", ascending=False)
    st.dataframe(df, use_container_width=True)

    st.plotly_chart(
        scatter_plot(df, "QED", "MPO Score", hover_data=["Name", "SMILES"],
                     title="MPO Score vs QED"),
        use_container_width=True,
    )
```

- [ ] **Step 10.3: Sanity-run the tests**

```
pytest -v
```

Expected: all tests still PASS.

- [ ] **Step 10.4: Manual smoke test**

Launch the app, go to Drug Optimization → Multi-Parameter Optimization.

1. Source = Paste SMILES, paste:
   ```
   CCO
   CCCCCCCCO
   c1ccccc1
   Cc1ccccc1
   ```
2. Use QSAR model = the model saved in Task 8.
3. Set QSAR axis weight = 0.5.
4. Click "Calculate MPO Scores".
5. Verify the table includes `Predicted pIC50`, `QSAR Score`, and `MPO Score`; sort order is by MPO Score desc.

- [ ] **Step 10.5: Commit**

```
git add modules/drug_optimization.py
git commit -m "Add optional QSAR-predicted-activity axis to Drug Optimization MPO

When a saved QSAR model is selected, MPO predicts activity once per
candidate batch, normalizes against the training y-range (flipping
for higher_is_better=False), and blends into the existing
property-desirability score via a user-controlled weight slider.
Invalid SMILES score the QSAR axis as 0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Add "QSAR Models" tab to Data Management

**Files:**
- Modify: `app.py`

- [ ] **Step 11.1: Update `render_data_management`**

Find the `tab1, tab2, tab3 = st.tabs(["Molecules", "Proteins", "Experiments"])` line (currently in `app.py:render_data_management`) and replace with:

```python
    tab1, tab2, tab3, tab4 = st.tabs(["Molecules", "Proteins", "Experiments", "QSAR Models"])
```

Then add a new `with tab4:` block after the existing `with tab3:` block:

```python
    with tab4:
        st.subheader("Saved QSAR Models")
        project_id = st.session_state.get("current_project_id")
        models = db.get_qsar_models(project_id=project_id)
        if models:
            import pandas as pd
            df = pd.DataFrame(models)
            display_cols = [
                "id", "name", "activity_label", "activity_transform",
                "higher_is_better", "cv_r2_mean", "n_molecules",
                "rdkit_version", "sklearn_version", "created_date",
            ]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_cols], use_container_width=True, hide_index=True)

            model_ids = df["id"].tolist()
            del_id = st.selectbox("Select model id to delete", model_ids, key="del_qsar_model")
            if st.button("Delete Selected Model", key="del_qsar_model_btn"):
                db.delete_qsar_model(del_id)
                st.success(f"Deleted model id {del_id}")
                st.rerun()
        else:
            st.info("No saved QSAR models for this project.")
```

- [ ] **Step 11.2: Sanity-run the tests**

```
pytest -v
```

Expected: all tests still PASS.

- [ ] **Step 11.3: Manual smoke test**

Launch the app, go to Data Management → QSAR Models. The saved model from Task 8 should appear. Click "Delete Selected Model" and verify both the row disappears and the `.joblib` file is gone from `data/qsar_models/`.

- [ ] **Step 11.4: Commit**

```
git add app.py
git commit -m "Add QSAR Models tab to Data Management

Lists saved models for the active project with metadata columns and
a delete control that removes both the DB row and the .joblib file.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Wrap-up — directory placeholder, gitignore, CLAUDE.md notes

**Files:**
- Create: `data/qsar_models/.gitkeep`
- Modify: `.gitignore`
- Modify: `CLAUDE.md`

- [ ] **Step 12.1: Create the directory placeholder**

```
mkdir -p data/qsar_models
```

Create empty file `data/qsar_models/.gitkeep` (so the directory is tracked but its contents aren't).

- [ ] **Step 12.2: Extend `.gitignore`**

Append to `.gitignore`:

```
data/qsar_models/*.joblib
```

- [ ] **Step 12.3: Update `CLAUDE.md`**

Find the "Cheminformatics layer" section header in `CLAUDE.md` and append the following paragraph immediately after that section's existing body:

```
**`utils/qsar.py`** is the second cheminformatics-adjacent module — and the only file that should import `joblib` or call `sklearn.*` for model persistence. It owns `train_qsar`, `save_model_artifact`, `load_model_artifact`, and `predict`. `modules/sar_analysis.py` and `modules/drug_optimization.py` consume it; neither imports joblib directly.
```

Find the "Run / develop" section. Find the paragraph that says "There is no lint, type-check, or formatter configured. There is no real test suite ...". Replace that paragraph with:

```
There is no lint, type-check, or formatter configured. The pytest suite lives in `tests/` — run with `pytest` from the repo root (config is in `pyproject.toml`). One-time setup in the conda env: `pip install pytest`. `test_imports.py` and `test_rdkit_imports.py` at the repo root remain as smoke scripts (and are gitignored), unrelated to the pytest suite.
```

- [ ] **Step 12.4: Final pytest run from scratch**

```
pytest -v
```

Expected: all 30+ tests PASS.

- [ ] **Step 12.5: Commit**

```
git add data/qsar_models/.gitkeep .gitignore CLAUDE.md
git commit -m "Document utils/qsar.py boundary and pytest setup

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Done criteria

- `pytest -v` runs green from the repo root.
- The app launches via `run.bat` and:
  - SAR Analysis → QSAR Modeling can train a model and save it.
  - SAR Analysis → QSAR Predict can score new SMILES against a saved model, flag training-set leaks, and download a CSV.
  - Drug Optimization → MPO can include an optional QSAR-predicted-activity axis.
  - Data Management → QSAR Models can list and delete saved models, and delete removes both the DB row and the `.joblib` file.
- `data/qsar_models/` exists; `*.joblib` is gitignored; `.gitkeep` is tracked.
- `CLAUDE.md` reflects `utils/qsar.py`'s role and pytest's existence.
