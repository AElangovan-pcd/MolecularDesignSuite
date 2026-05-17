# QSAR Model Persistence & Prediction — Design Spec

**Date:** 2026-05-16
**Status:** Approved (sections §1–§4)
**Scope:** Persist trained QSAR models, add a Predict tab, wire predictions into Drug Optimization MPO, and backfill pytest coverage for `utils/rdkit_utils.py` plus the new code.

## Problem

`modules/sar_analysis.py::_qsar_tab` trains a `RandomForestRegressor` from molecular descriptors + an activity column, reports 5-fold CV R², and throws the model away. There is no way to predict activity on a new molecule, no way to compare two models, and no way for `drug_optimization.py`'s Multi-Parameter Optimization (MPO) to use a learned activity model as a scoring axis. The QSAR feature is half-built.

## Goals

1. Save trained models to disk (with metadata in the SQLite DB) so they survive restarts.
2. Add a "QSAR Predict" tab in `sar_analysis.py` that loads a saved model and scores arbitrary SMILES.
3. Add an optional "Predicted Activity" axis to `drug_optimization.py`'s MPO desirability scoring.
4. Cover the new `utils/qsar.py` module with pytest, and backfill tests for the existing pure-function surface in `utils/rdkit_utils.py` and `database/db_manager.py`.

## Non-goals

- Hyperparameter tuning UI. The model stays `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`.
- Multiple model families (XGBoost, neural nets) — single-family persistence keeps the artifact schema closed.
- Back-transforming predictions (e.g. pIC50 → IC50 nM). Predictions are reported in the same units as the training activity, with the user-supplied `activity_label` shown alongside.
- Cross-session concurrency, disk-full handling, GitHub Actions CI — out of scope for a local single-user app.

## Architecture

A new domain concept — **QSAR Model** — is a first-class persisted artifact with three storage planes:

1. **Artifact on disk** — `data/qsar_models/<id>.joblib`, written via `joblib.dump`. Contains a single dict:
   ```python
   {
     "model": RandomForestRegressor,
     "scaler": StandardScaler,
     "feature_columns": list[str],          # column order is part of the contract
     "training_smiles_hashes": frozenset[str],  # canonical SMILES md5 hashes for leak check
     "rdkit_version": str,
     "sklearn_version": str,
   }
   ```
2. **Metadata in SQLite** — a new `qsar_models` table (schema below) holding everything else.
3. **A new module `utils/qsar.py`** — the only file that imports `joblib` and `sklearn`. Public API:
   ```python
   @dataclass
   class ModelArtifact:
       model: Any
       scaler: Any
       feature_columns: list[str]
       training_smiles_hashes: frozenset[str]
       rdkit_version: str
       sklearn_version: str

   def train_qsar(
       df: pd.DataFrame,
       smiles_col: str,
       activity_col: str,
       activity_transform: str = "none",  # "none" | "log10" | "pIC50"
   ) -> tuple[ModelArtifact, dict]:  # (artifact, {"cv_r2_mean": ..., "cv_r2_std": ..., "n_molecules": ...})
       ...

   def save_model_artifact(
       artifact: ModelArtifact,
       meta: dict,           # user/UI-supplied: name, dataset_name, activity_label,
                             # activity_transform, higher_is_better, cv_r2_mean,
                             # cv_r2_std, project_id
       db: DatabaseManager,
       models_dir: Path = Path("data/qsar_models"),
   ) -> int: ...             # returns the new qsar_models.id
                             # Internally derived from artifact:
                             #   n_molecules = len(artifact.training_smiles_hashes)
                             #   rdkit_version, sklearn_version (from artifact fields)
                             #   model_type = type(artifact.model).__name__
                             #   artifact_path = str(models_dir / f"{id}.joblib")

   def load_model_artifact(
       model_id: int,
       db: DatabaseManager,
       models_dir: Path = Path("data/qsar_models"),
   ) -> ModelArtifact: ...

   def predict(
       artifact: ModelArtifact,
       smiles_list: list[str],
   ) -> list[dict]: ...       # [{"smiles": ..., "predicted_value": float|None, "in_training": bool, "error": str|None}, ...]
   ```

`sar_analysis.py` and `drug_optimization.py` import only from `utils.qsar`. Neither file gains a `joblib` or sklearn-persistence import.

### Data flow

```
[Train tab]
  CSV upload ─► calculate_descriptor_set for each row ─► utils.qsar.train_qsar
            └─► artifact lives in st.session_state["qsar_last_artifact"]
            └─► metrics shown (R², feature importances, pred-vs-actual)

  [Save Model] ─► db.add_qsar_model(meta) ─► joblib.dump(artifact, data/qsar_models/<id>.joblib)
              └─► on dump failure: db.delete_qsar_model(id) to roll back the row

[Predict tab]
  Dropdown reads db.get_qsar_models(project_id)
  Selection ─► utils.qsar.load_model_artifact(model_id, db)
  SMILES input (paste or from DB) ─► utils.qsar.predict(artifact, smiles_list)
  Table render with predicted_value, in_training flag, errors

[Drug Optimization MPO]
  Existing axes (MW, LogP, TPSA, QED) unchanged.
  New optional axis: "QSAR Predicted Activity" dropdown over db.get_qsar_models.
  When selected, MPO loop calls utils.qsar.predict once per candidate batch.
  Desirability direction = "max" if higher_is_better else "min".
  Result cached via @st.cache_data keyed on (model_id, tuple(smiles_list)).
```

## Schema change

One additive table in `database/schema.sql`. `DatabaseManager._init_db` re-runs the schema with `CREATE TABLE IF NOT EXISTS`, so existing DBs auto-upgrade on next launch — no manual migration step.

```sql
CREATE TABLE IF NOT EXISTS qsar_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,                     -- user-supplied, NOT unique (append-only)
    dataset_name TEXT,                      -- source CSV filename for provenance
    n_molecules INTEGER NOT NULL,           -- training set size after invalid-SMILES drop
    activity_label TEXT NOT NULL,           -- free text: "pIC50", "log(EC50) [nM]", etc.
    activity_transform TEXT DEFAULT 'none', -- 'none' | 'log10' | 'pIC50'
    higher_is_better INTEGER NOT NULL,      -- 1 = maximize predicted (for MPO), 0 = minimize
    cv_r2_mean REAL,
    cv_r2_std REAL,
    model_type TEXT DEFAULT 'RandomForestRegressor',
    artifact_path TEXT NOT NULL,            -- "data/qsar_models/<id>.joblib"
    rdkit_version TEXT,
    sklearn_version TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    project_id INTEGER,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_qsar_models_project ON qsar_models(project_id);
```

Three new `DatabaseManager` methods:

```python
def add_qsar_model(self, meta: dict) -> int: ...           # INSERT, return lastrowid
def get_qsar_models(self, project_id: int | None = None) -> list[dict]: ...
def delete_qsar_model(self, model_id: int) -> None: ...    # DELETE row + Path(artifact_path).unlink(missing_ok=True)
                                                            # (tolerant of missing file so it's safe as a rollback path)
```

Design choices to note:

- **`name` is not unique** — per the append-only versioning decision. UI shows `"Aspirin-RF (R²=0.71, 2026-05-16 14:32)"` in dropdowns to disambiguate.
- **`activity_transform` applies at train time only.** The stored model maps descriptors → transformed-y. Predict outputs are in transformed space and surfaced as e.g. `"Predicted pIC50: 6.42"` — no back-transformation.
- **`higher_is_better`** is captured at save time and consumed by MPO; it has no effect on training or single-molecule prediction display.

## UI changes

### `modules/sar_analysis.py` — extend `_qsar_tab`

After the existing model-build flow renders R² + feature importances + pred-vs-actual, append:

```
─── Save This Model ─────────────────────────
Name:               [<csv-stem>-RF       ]
Activity label:     [                    ]   (required)
Activity transform: (●) None  ( ) log10  ( ) pIC50
Higher activity = better potency?  [✓]
                                  [ Save Model ]
```

`st.session_state["qsar_last_artifact"]` holds the trained artifact between the build and save clicks. Save validates that `activity_label` is non-empty, then calls `save_model_artifact`. On success: `st.success(f"Saved as model #{id}")`.

### `modules/sar_analysis.py` — new 6th tab `"QSAR Predict"`

```
Model: [Aspirin-RF (R²=0.71, 2026-05-16 14:32) ▾]
       Activity label: pIC50    Higher is better: yes    n_train: 47

Input: (●) Paste SMILES  ( ) From Database
[ multiline text or db selector ]

[ Predict ]

Results table: SMILES | Predicted <activity_label> | in_training? | error
[ Download CSV ]
```

The `in_training?` column compares each input's canonical-SMILES md5 against `artifact.training_smiles_hashes` — a cheap data-leak flag.

### `modules/drug_optimization.py` — wire into MPO

Add one optional axis in the MPO tab UI:

```
─── QSAR Predicted Activity (optional) ─────
Use QSAR model: [None ▾] / [Aspirin-RF (R²=0.71) ▾]
   Weight: [────●────] 0.5
```

When a model is selected, the existing MPO scoring loop adds a `predicted_<activity_label>` column to the candidate DataFrame and a desirability function (`max` if `higher_is_better else min`) with the slider weight. Predictions are cached via `@st.cache_data` keyed on `(model_id, tuple(smiles_list))` — the model id is itself the cache invalidation key.

### `app.py::render_data_management` — new 4th tab

`"QSAR Models"` listing `db.get_qsar_models()` (id, name, activity_label, cv_r2_mean, n_molecules, created_date) with a per-row delete button that calls `db.delete_qsar_model(id)` (which removes both the row and the `.joblib` file).

No sidebar entry. Models surface only where they are consumed.

## Error handling

Only cases that can actually happen are handled. Everything else is a bug and should crash visibly.

- **`train_qsar` — empty descriptor matrix** (all SMILES invalid) → `ValueError("No valid molecules in training set")`. UI catches and shows `st.error`.
- **`train_qsar` — activity column non-numeric** → `pd.to_numeric(errors="raise")` lets the exception propagate.
- **`train_qsar` — `activity_transform="pIC50"` or `"log10"` with non-positive values** → `ValueError(f"{transform} transform requires positive values")`, raised before training.
- **`save_model_artifact` — `joblib.dump` raises after the DB row exists** → `try/except` rolls back the row via `db.delete_qsar_model(id)` and re-raises. `delete_qsar_model`'s `unlink(missing_ok=True)` makes this safe even though the file was never written. This is the only orphan-prevention path that matters.
- **`load_model_artifact` — `.joblib` missing on disk** → `FileNotFoundError`. UI shows `st.error("Model artifact missing on disk — delete this row from Data Management")` and the dropdown filters broken rows on next render.
- **`predict` — invalid SMILES in input list** → that row gets `predicted_value=None, error="Invalid SMILES"`. No exception; batch prediction must not abort on one bad input.
- **Version skew on load** — compare stored `rdkit_version` / `sklearn_version` with current; mismatch shows a yellow `st.warning` but does **not** block prediction.

Deliberately not handled: file-permission errors on `data/qsar_models/`, disk full, concurrent saves from two browser tabs.

## Testing

Tooling:

- Add `pytest` to the conda env: `pip install pytest`. Not added to `requirements.txt` (which is documentation only — the conda env is the source of truth per the repo CLAUDE.md), but document the install in CLAUDE.md.
- New `pyproject.toml` at repo root with the minimum pytest config:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  pythonpath = ["."]
  ```
- New `tests/conftest.py` providing two fixtures:
  - `tmp_db` — `DatabaseManager` backed by `tmp_path / "test.db"`.
  - `sample_dataset_df` — 10-row pandas DataFrame with `SMILES` and `Activity` columns (linear-ish synthetic data in the 3–7 range to keep RF R² nontrivial).

Test files:

```
tests/
├── conftest.py
├── test_qsar.py          # NEW MODULE — covers utils/qsar.py
├── test_db_manager.py    # qsar_models CRUD + add_molecule INSERT OR IGNORE contract
└── test_rdkit_utils.py   # backfill for the existing pure-function surface
```

`tests/test_qsar.py`:

1. `train_qsar` on `sample_dataset_df` returns a `ModelArtifact` with non-empty `feature_columns` and a fitted `scaler`.
2. `train_qsar` raises `ValueError` on a dataset where every SMILES is invalid.
3. Each `activity_transform` branch (`none` / `log10` / `pIC50`) trains on different y-vectors — assert the resulting models' `predict` outputs differ.
4. `activity_transform="pIC50"` with a non-positive activity column raises `ValueError`.
5. `save_model_artifact` + `load_model_artifact` round-trip: train, save, load, predict on the same input — predictions identical to the in-memory artifact's predictions.
6. `save_model_artifact` rollback: monkeypatch `joblib.dump` to raise `IOError`; assert the DB row does not exist afterwards.
7. `predict` on a mix of valid + invalid SMILES — valid rows have `float` predictions, invalid rows have `None` + non-empty `error`.
8. `predict` flags training-set members via `training_smiles_hashes` (use one molecule from `sample_dataset_df` as the predict input).

`tests/test_db_manager.py`:

1. `add_qsar_model` returns an integer id, `get_qsar_models()` includes the row.
2. `get_qsar_models(project_id=X)` filters by project.
3. `delete_qsar_model(id)` removes the row **and** removes the artifact file (use a `tmp_path` `.joblib` stub).
4. `add_molecule` with a duplicate SMILES returns the existing row id (pins the `INSERT OR IGNORE` contract).

`tests/test_rdkit_utils.py` — one test per pure helper:

1. `mol_from_smiles` returns a Mol for aspirin, `None` for `"not a smiles"`.
2. `canonical_smiles` is idempotent.
3. `calculate_basic_properties` returns all expected keys for aspirin and the MW is within 1 amu of 180.16.
4. `lipinski_rule_of_five` — aspirin `passes=True`, a deliberate failure (long alkane like `C` * 60) `passes=False`.
5. `veber_rules`, `ghose_filter`, `egan_rules`, `muegge_rules` — one passing case each (skip exhaustive cross-checks).
6. `tanimoto_similarity(mol, mol) == 1.0`.
7. `check_pains` returns a non-empty list for a known PAINS structure (e.g. a quinone-style false-positive scaffold), empty list for aspirin.

## Files touched

**New:**
- `utils/qsar.py`
- `tests/conftest.py`
- `tests/test_qsar.py`
- `tests/test_db_manager.py`
- `tests/test_rdkit_utils.py`
- `pyproject.toml`
- `data/qsar_models/` (directory created at first save, with a `.gitkeep`)

**Modified:**
- `database/schema.sql` — add `qsar_models` table + index
- `database/db_manager.py` — add `add_qsar_model`, `get_qsar_models`, `delete_qsar_model`
- `modules/sar_analysis.py` — extend `_qsar_tab` with Save section, add `_predict_tab` and a 6th tab
- `modules/drug_optimization.py` — add optional QSAR axis to MPO
- `app.py` — add "QSAR Models" tab to `render_data_management`
- `.gitignore` — add `data/qsar_models/*.joblib`
- `CLAUDE.md` — note `utils/qsar.py` as the single sklearn-persistence entry point; mention `pytest` install + `pytest tests/` to run

## Out of scope (future work)

- Predict tab support for SDF/CSV upload (paste + DB are enough for v1).
- A "compare models" UI (overlay pred-vs-actual scatters from two saved models).
- Active-learning workflow (predict on a virtual library, surface uncertain ones for assay).
- Model export to ONNX / sharing.
- A "duplicate detection at predict time" warning that flags input SMILES exactly matching molecules already in the user's DB.
