# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single Streamlit app — **Molecular Design Suite** — for drug-discovery workflows over RDKit. One process, one SQLite DB, no separate backend. Five functional modules sit behind a sidebar `st.radio` in `app.py`; everything else (utilities, DB layer, file/visualization helpers) is dependency-injected via the `DatabaseManager` passed into each `render_*` entrypoint.

Author: A. Elangovan. The workspace-level `~/Documents/ClaudeProjects/CLAUDE.md` and `~/Documents/CLAUDE.md` apply on top of this file — read those for shell/env conventions before doing system work.

## Run / develop

The conda env `moldesign` (at `C:\Users\easam\.conda\envs\moldesign`) is the source of truth, **not** `requirements.txt`. RDKit must come from `conda-forge`; pip-installed `rdkit-pypi` on Windows fails to find its DLLs.

```bash
# One-time setup (Anaconda Prompt — not PowerShell, not Git Bash):
conda create -n moldesign python=3.11 rdkit -c conda-forge -y
conda activate moldesign
pip install streamlit plotly seaborn biopython py3Dmol stmol pubchempy chembl-webresource-client scikit-learn pillow streamlit-ketcher
```

Launch:

- **Windows preferred**: double-click or run `run.bat` from Command Prompt — it prepends `<env>`, `<env>\Library\bin`, and `<env>\Library\mingw-w64\bin` to PATH so RDKit's DLLs load, then `python -m streamlit run app.py`. If you ever move the conda env, edit the `CONDA_ENV` line at the top of `run.bat`.
- **Manual**: `conda activate moldesign && streamlit run app.py --server.port 8501 --server.headless true`. App is at <http://localhost:8501>.

There is no lint, type-check, or formatter configured. There is no real test suite — `test_imports.py` and `test_rdkit_imports.py` are smoke scripts that print `[OK]/[FAIL]` per import (`python test_imports.py`). Both are gitignored.

## Architecture

### Module routing
`app.py` is the single entrypoint. The sidebar holds:

1. **Project selector** — reads from `db.get_projects()`; the selected id lives in `st.session_state["current_project_id"]` and gates DB reads in every module (project = `None` means "all").
2. **Navigation radio** — picks one of `modules/{molecular_input, property_calc, protein_analysis, sar_analysis, drug_optimization}.py`. Each exposes a single `render_X(db: DatabaseManager)` function; `app.py` dispatches with an `if/elif` chain. The 6th option, "Data Management", is defined inline in `app.py:render_data_management` rather than in `modules/`.
3. **Structure Editor** — a `streamlit_ketcher.st_ketcher` instance writing to `st.session_state["editor_smiles"]`, plus a "Save to Database" button that calls `calculate_basic_properties` and `db.add_molecule` directly.

Each module file follows the same shape: a `render_X(db)` entry that creates `st.tabs([...])` and dispatches each tab to a `_underscore_tab(db)` private function. When adding a feature, match that shape — don't introduce class-based pages or a routing framework.

### Sidebar ↔ module data flow (important)
There are **two** distinct session-state slots that form a one-way pipe between the sidebar editor and the active module:

- `editor_smiles` — owned by Ketcher in the sidebar. Always reflects what's drawn. Modules can **write** to it (see `utils/editor_helpers.edit_in_editor_button`) to load an existing molecule back into the editor; the next rerun shows it on the canvas.
- `active_smiles` — set only when the user clicks "Use This Molecule". `modules/molecular_input.py` consumes it via `st.session_state.pop("active_smiles", "")` so it fires exactly once on the next rerun and doesn't override later user edits to the text box.

When wiring a new module that should accept molecules from the editor, read `active_smiles` with `.pop()`, not `.get()`. When adding a new "Edit in Structure Editor" affordance anywhere in the app, use `utils.editor_helpers.edit_in_editor_button(smiles, key=...)` — don't re-implement the `editor_smiles = smiles; st.rerun()` dance.

### Cheminformatics layer
`utils/rdkit_utils.py` is the only place that should import from `rdkit.*` for chemistry logic. It centralizes parsing (`mol_from_smiles`, `canonical_smiles`, `validate_smiles`), property calc (`calculate_basic_properties`, `calculate_descriptor_set`), drug-likeness filters (`lipinski_rule_of_five`, `veber_rules`, `ghose_filter`, `egan_rules`, `muegge_rules`, `all_drug_likeness_filters`), ADMET estimates (`admet_properties` + helpers), Morgan fingerprints + Tanimoto, Murcko scaffolds, 3D conformer embedding, and PAINS alerts via `FilterCatalog`.

`synthetic_accessibility_score()` opportunistically loads `sascorer.py` from `rdkit.Chem.RDConfig.RDContribDir/SA_Score`; if the contrib dir isn't shipped with the conda build, it falls back to `_estimate_sa_score` (a basic heuristic). Don't replace the fallback without checking that the contrib package is reliably present.

Visualization helpers (`utils/visualization.py`), file parsers (`utils/file_handlers.py`), and editor helpers (`utils/editor_helpers.py`) sit alongside but never import each other — they fan out from `rdkit_utils`.

### Data layer
`database/db_manager.py` wraps a single SQLite file at `database/molecular_design.db`. `DatabaseManager.__init__` runs `schema.sql` via `executescript` on every construction, so the schema file (`database/schema.sql`) is authoritative — edit the schema there, not via ad-hoc `ALTER TABLE`s in code. Foreign keys are on; `molecules.smiles` is `UNIQUE`, and `add_molecule()` uses `INSERT OR IGNORE` then looks up the existing row to return its id — so the function never raises on duplicates, it returns the pre-existing id. Callers depend on that. The DB file is gitignored.

The schema covers `projects`, `molecules`, `molecular_properties` (EAV side table keyed by `(molecule_id, property_name, calculation_method)`), `proteins`, `experiments`, and `sar_datasets` + `sar_dataset_molecules` (many-to-many with activity).

### Protein workflow
`modules/protein_analysis.py` is the only module that calls external HTTP services. It posts to RCSB's REST search (`search.rcsb.org/rcsbsearch/v2/query`) and then to the RCSB GraphQL endpoint (`data.rcsb.org/graphql`) to hydrate metadata. Downloaded PDB/CIF files land in `data/protein_structures/` which is gitignored (`*.pdb`, `*.ent`) — assume the directory may be empty on a fresh clone. 3D visualization uses `py3Dmol` + `stmol`.

## Conventions to match

- **Don't bypass `utils/rdkit_utils.py`.** New chemistry code goes there, even if it's used in only one module. Modules should import named helpers, not call `Chem.*` directly except for trivial things like `Chem.MolToSmiles(mol)` on an object they already have.
- **Don't introduce module state outside `st.session_state`.** No module-level globals; Streamlit reruns the whole script per interaction.
- **Project-id scoping** is opt-in per query (`project_id=None` means all). Match existing call sites — most modules read `st.session_state.get("current_project_id")` once and pass it down.
- **Empty `__init__.py`** files exist solely to make `modules/`, `database/`, `utils/` importable. Don't put re-exports or import side-effects there; `app.py` adds the project root to `sys.path` itself.
- **Drug-likeness dicts** all use a `"passes": bool` key alongside the individual checks. Keep that contract — the UI iterates over the dict and treats `passes` as the summary row.

## Gotchas

- The `.streamlit/` config directory is gitignored — server settings come from CLI flags in `run.bat`, not from a checked-in `config.toml`.
- `streamlit-ketcher`'s output is `None` on first render; the sidebar guards with `if editor_result is not None` before writing to `editor_smiles`. Preserve that guard or you'll wipe the drawing on every rerun.
- `check_pains()` deliberately breaks after the first match (the loop `entry = catalog.GetFirstMatch(mol); break` is intentional — repeated `GetFirstMatch` on the same `mol` would loop forever). Don't "fix" it into `GetMatches` without verifying behavior.
- `docs/superpowers/{plans,specs}/` contains historical design notes for the molecular editor feature — useful context, not source of truth for current code.
