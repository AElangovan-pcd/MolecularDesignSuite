# Molecular Structure Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Ketcher-based molecular structure editor in the sidebar that enables draw-to-SMILES input and bi-directional molecule loading across all app modules.

**Architecture:** Ketcher editor embedded in `st.sidebar` via `streamlit-ketcher`. Real-time SMILES preview + "Use This Molecule" button pushes SMILES to modules via `st.session_state["active_smiles"]`. "Edit in Structure Editor" buttons across modules set `st.session_state["editor_smiles"]` to load molecules back into the editor. A shared helper function avoids repeating button logic.

**Tech Stack:** Streamlit, streamlit-ketcher, RDKit (existing)

**Conda environment:** `moldesign` at `C:\Users\easam\.conda\envs\moldesign`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `requirements.txt` | Modify | Add streamlit-ketcher dependency |
| `app.py` | Modify | Add sidebar editor section + session state init |
| `utils/editor_helpers.py` | Create | Shared "Edit in Structure Editor" button helper |
| `modules/molecular_input.py` | Modify | Read active_smiles, add edit button |
| `modules/property_calc.py` | Modify | Add edit button to molecule displays |
| `modules/drug_optimization.py` | Modify | Add edit button to lead/analog displays |
| `modules/sar_analysis.py` | Modify | Add edit button to selected molecules |

---

### Task 1: Install Dependency and Add Session State

**Files:**
- Modify: `requirements.txt`
- Modify: `app.py:18-28` (init_session_state)

- [ ] **Step 1: Add streamlit-ketcher to requirements.txt**

Add this line at the end of `requirements.txt`:

```
streamlit-ketcher>=0.0.1
```

- [ ] **Step 2: Install the package**

Run from Windows (the conda env is on Windows side):

```bash
powershell.exe -Command "& 'C:\Users\easam\.conda\envs\moldesign\Scripts\pip.exe' install streamlit-ketcher"
```

Expected: successful installation.

- [ ] **Step 3: Add session state keys to app.py**

In `app.py`, add `editor_smiles` and `active_smiles` to the defaults dict in `init_session_state()`. Replace lines 20-25:

```python
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "current_project_id": None,
        "current_protein_id": None,
        "current_pdb_file": None,
        "current_pdb_id": None,
        "editor_smiles": "",
        "active_smiles": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
```

- [ ] **Step 4: Commit**

```bash
cd /mnt/c/Users/easam/Documents/ClaudeProjects/RDKitProjects
git add requirements.txt app.py
git commit -m "feat: add streamlit-ketcher dependency and editor session state"
```

---

### Task 2: Create Shared Editor Helper

**Files:**
- Create: `utils/editor_helpers.py`

- [ ] **Step 1: Create the helper module**

Create `utils/editor_helpers.py`:

```python
"""Shared helper for Structure Editor integration across modules."""

import streamlit as st


def edit_in_editor_button(smiles: str, key: str):
    """Render an 'Edit in Structure Editor' button that loads a molecule into the sidebar editor.

    Args:
        smiles: The SMILES string of the molecule to load.
        key: Unique Streamlit widget key to avoid duplicate IDs.
    """
    if smiles and st.button("\u270f\ufe0f Edit in Structure Editor", key=key):
        st.session_state["editor_smiles"] = smiles
        st.rerun()
```

- [ ] **Step 2: Commit**

```bash
cd /mnt/c/Users/easam/Documents/ClaudeProjects/RDKitProjects
git add utils/editor_helpers.py
git commit -m "feat: add shared editor helper for Edit in Structure Editor buttons"
```

---

### Task 3: Add Sidebar Editor to app.py

**Files:**
- Modify: `app.py:3-4` (imports), `app.py:92-99` (sidebar, after navigation divider)

- [ ] **Step 1: Add imports**

Add after the existing imports (after line 15 `from modules.drug_optimization import render_drug_optimization`):

```python
from streamlit_ketcher import st_ketcher
from utils.rdkit_utils import validate_smiles, mol_from_smiles
```

- [ ] **Step 2: Add the editor section in the sidebar**

In the `main()` function, inside the `with st.sidebar:` block, insert the editor section BETWEEN the navigation radio (ends around line 90) and the divider before "Database Stats" (line 92). Replace the section from `st.divider()` (line 92) through `st.text(f"Proteins: {len(proteins)}")` (line 99) with:

```python
        st.divider()

        # Structure Editor
        st.markdown("### Structure Editor")
        with st.expander("Draw Molecule", expanded=True):
            editor_result = st_ketcher(
                value=st.session_state.get("editor_smiles", ""),
                height=400,
                molecule_format="SMILES",
                key="ketcher_editor",
            )

            # Update editor_smiles from Ketcher output
            if editor_result is not None:
                st.session_state["editor_smiles"] = editor_result

            # Real-time SMILES preview
            current_smiles = st.session_state.get("editor_smiles", "")
            if current_smiles:
                is_valid, _ = validate_smiles(current_smiles)
                if is_valid:
                    mol = mol_from_smiles(current_smiles)
                    from rdkit.Chem import rdMolDescriptors
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    st.success(f"Molecule ready: {formula}")
                    st.code(current_smiles, language=None)

                    if st.button("Use This Molecule", key="use_molecule_btn", type="primary"):
                        st.session_state["active_smiles"] = current_smiles
                        st.rerun()
                else:
                    st.caption("Drawing incomplete...")
            else:
                st.caption("No structure drawn")

        st.divider()

        # Quick stats
        st.markdown("### Database Stats")
        mol_count = db.get_molecule_count(st.session_state.get("current_project_id"))
        proteins = db.get_proteins(st.session_state.get("current_project_id"))
        st.text(f"Molecules: {mol_count}")
        st.text(f"Proteins: {len(proteins)}")
```

- [ ] **Step 3: Verify the editor renders**

Run the app:

```bash
powershell.exe -Command "cd C:\Users\easam\Documents\ClaudeProjects\RDKitProjects; .\run.bat"
```

Open http://localhost:8501. Verify:
- Structure Editor section appears in the sidebar
- Ketcher canvas is interactive (can draw bonds, atoms)
- Drawing a molecule shows SMILES preview and "Molecule ready" status
- "Use This Molecule" button appears when structure is valid

- [ ] **Step 4: Commit**

```bash
cd /mnt/c/Users/easam/Documents/ClaudeProjects/RDKitProjects
git add app.py
git commit -m "feat: add Ketcher molecular editor to sidebar"
```

---

### Task 4: Integrate Editor with Molecular Input Module

**Files:**
- Modify: `modules/molecular_input.py:8-16` (imports), `modules/molecular_input.py:40-104` (_smiles_input_tab)

- [ ] **Step 1: Add import**

Add after the existing imports (after line 16 `from database.db_manager import DatabaseManager`):

```python
from utils.editor_helpers import edit_in_editor_button
```

- [ ] **Step 2: Modify _smiles_input_tab to read active_smiles**

Replace the SMILES text_input (lines 46-50) to use active_smiles as default:

```python
        # Use molecule from editor if available
        default_smiles = st.session_state.pop("active_smiles", "") or ""
        smiles = st.text_input(
            "SMILES string",
            value=default_smiles,
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
            help="Enter a valid SMILES string or use the Structure Editor in the sidebar",
        )
```

- [ ] **Step 3: Add "Edit in Structure Editor" button after structure display**

After the `st.image(svg, use_container_width=True)` call (line 62), add:

```python
                edit_in_editor_button(smiles, key="edit_smiles_input")
```

- [ ] **Step 4: Verify integration**

Run the app. Test:
1. Draw a molecule in the sidebar editor, click "Use This Molecule" — SMILES should appear in the Molecular Input text field
2. Type a SMILES manually, see the structure, click "Edit in Structure Editor" — molecule should load into the sidebar editor

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/easam/Documents/ClaudeProjects/RDKitProjects
git add modules/molecular_input.py
git commit -m "feat: integrate sidebar editor with molecular input module"
```

---

### Task 5: Add Edit Buttons to Property Calculation Module

**Files:**
- Modify: `modules/property_calc.py`

- [ ] **Step 1: Add import**

Add after the existing imports (after line 17 `from database.db_manager import DatabaseManager`):

```python
from utils.editor_helpers import edit_in_editor_button
```

- [ ] **Step 2: Add edit button to _get_molecule_input**

In `_get_molecule_input()`, after a valid molecule is obtained from SMILES input (after line 54 `mol = mol_from_smiles(smiles)`), add the edit button. Insert after the `if mol is None: st.error("Invalid SMILES")` block (after line 56):

```python
            if mol:
                edit_in_editor_button(smiles, key=f"{key_prefix}_edit_btn")
```

And for the database branch, after a molecule is selected from the database (the `else` branch around line 57-60), add a similar button. Find where `smiles` is set from the database selection and add:

```python
                edit_in_editor_button(smiles, key=f"{key_prefix}_db_edit_btn")
```

- [ ] **Step 3: Commit**

```bash
cd /mnt/c/Users/easam/Documents/ClaudeProjects/RDKitProjects
git add modules/property_calc.py
git commit -m "feat: add Edit in Structure Editor buttons to property calculation"
```

---

### Task 6: Add Edit Buttons to Drug Optimization Module

**Files:**
- Modify: `modules/drug_optimization.py`

- [ ] **Step 1: Add import**

Add after the existing imports (after line 21 `from database.db_manager import DatabaseManager`):

```python
from utils.editor_helpers import edit_in_editor_button
```

- [ ] **Step 2: Add edit button to _lead_profiling_tab**

In `_lead_profiling_tab()`, after the molecule SVG is displayed (after line 67 `st.image(svg, use_container_width=True)`), add:

```python
        edit_in_editor_button(smiles, key="lead_edit_btn")
```

- [ ] **Step 3: Add edit button to _analog_generation_tab**

Find `_analog_generation_tab()` — after analog molecules are displayed (there should be a loop displaying generated analogs with SVGs), add an edit button for each analog. Inside the display loop, add:

```python
                    edit_in_editor_button(analog_smiles, key=f"analog_edit_{i}")
```

Where `analog_smiles` is the SMILES of the generated analog and `i` is the loop index. Read the function first to find the exact variable names and insertion point.

- [ ] **Step 4: Commit**

```bash
cd /mnt/c/Users/easam/Documents/ClaudeProjects/RDKitProjects
git add modules/drug_optimization.py
git commit -m "feat: add Edit in Structure Editor buttons to drug optimization"
```

---

### Task 7: Add Edit Buttons to SAR Analysis Module

**Files:**
- Modify: `modules/sar_analysis.py`

- [ ] **Step 1: Add import**

Add after the existing imports (after line 20 `from database.db_manager import DatabaseManager`):

```python
from utils.editor_helpers import edit_in_editor_button
```

- [ ] **Step 2: Add edit button to molecule displays**

In `_similarity_tab()` or `_descriptors_tab()` — wherever individual molecules are displayed with their SMILES, add an edit button. Find the display loop and add:

```python
                    edit_in_editor_button(mol_smiles, key=f"sar_edit_{i}")
```

Where `mol_smiles` is the molecule's SMILES and `i` is the loop index. Read the specific functions first to find exact variable names and insertion points.

- [ ] **Step 3: Commit**

```bash
cd /mnt/c/Users/easam/Documents/ClaudeProjects/RDKitProjects
git add modules/sar_analysis.py
git commit -m "feat: add Edit in Structure Editor buttons to SAR analysis"
```

---

### Task 8: End-to-End Verification

**Files:** No changes — verification only.

- [ ] **Step 1: Test draw-to-input flow**

1. Open the app at http://localhost:8501
2. Draw a molecule (e.g., benzene ring) in the sidebar Ketcher editor
3. Verify SMILES preview appears below the editor
4. Click "Use This Molecule"
5. Navigate to "Molecular Input & Design" — verify SMILES appears in the text field
6. Verify structure renders and properties calculate

- [ ] **Step 2: Test edit-from-module flow**

1. Type a known SMILES (e.g., `CC(=O)OC1=CC=CC=C1C(=O)O` for aspirin) in Molecular Input
2. Click "Edit in Structure Editor"
3. Verify aspirin loads in the sidebar Ketcher editor
4. Modify the structure (add/remove an atom)
5. Click "Use This Molecule" to push the modified molecule back

- [ ] **Step 3: Test across modules**

1. Draw a molecule in the editor
2. Click "Use This Molecule"
3. Navigate to Property Calculation — verify the molecule can be used
4. Navigate to Drug Optimization — verify lead profiling accepts it
5. Check that "Edit in Structure Editor" buttons work in each module

- [ ] **Step 4: Final commit**

```bash
cd /mnt/c/Users/easam/Documents/ClaudeProjects/RDKitProjects
git add -A
git commit -m "feat: complete molecular editor integration across all modules"
```
