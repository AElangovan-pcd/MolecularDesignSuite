# Molecular Structure Editor Integration Design

**Date:** 2026-04-15
**App:** Molecular Design Suite (RDKitProjects)
**Goal:** Add a ChemDraw-style visual molecular editor in the sidebar using Ketcher, enabling draw-to-SMILES input and bi-directional molecule loading across all modules.

## Decision Log

| Decision | Choice |
|----------|--------|
| Editor library | Ketcher via `streamlit-ketcher` |
| Placement | Sidebar — always available in all modules |
| SMILES population | Both: real-time preview + "Use This Molecule" button |
| Bi-directional | Yes — draw new molecules AND load existing ones for editing |

## Architecture

### New Dependency

- `streamlit-ketcher` pip package, installed in `moldesign` conda environment
- Added to `requirements.txt`

### Sidebar Editor Component (in app.py)

**Location:** `st.sidebar`, below the existing project selector and database stats.

**UI layout:**
- Collapsible section: "Structure Editor" (expanded by default)
- Ketcher editor embedded via `streamlit-ketcher` component (~400px height)
- Below the editor: real-time SMILES preview text (read-only, updates as you draw)
- "Use This Molecule" button — pushes SMILES into `st.session_state["active_smiles"]`
- Status indicator: "Molecule ready" with molecular formula when valid, "No structure" when empty

**Bi-directional loading:**
- `st.session_state["editor_smiles"]` stores the current editor molecule
- Any module can set this value to load a molecule into the editor
- Ketcher accepts a SMILES string as initial value, so loading existing molecules works natively

### Session State Keys

| Key | Type | Purpose |
|-----|------|---------|
| `editor_smiles` | str | Current SMILES in Ketcher editor (bi-directional) |
| `active_smiles` | str | SMILES pushed from editor to modules via "Use This Molecule" |

### Integration with Existing Modules

**How modules receive the molecule:**
- "Use This Molecule" button sets `st.session_state["active_smiles"]`
- Each module's SMILES input field uses `st.session_state["active_smiles"]` as default value for `st.text_input()`
- User can still type SMILES manually — text field and editor are not mutually exclusive

**"Edit in Structure Editor" buttons added to:**
- `modules/molecular_input.py` — after SMILES validation and structure display
- `modules/property_calc.py` — next to the molecule being analyzed
- `modules/drug_optimization.py` — next to lead compound and generated analogs
- `modules/sar_analysis.py` — next to selected molecules in the dataset

Each button sets `st.session_state["editor_smiles"]` to that molecule's SMILES, which Ketcher picks up on the next render.

### Data Flow

```
Draw in Ketcher -> editor_smiles updates in real-time -> SMILES preview shown
    -> "Use This Molecule" -> active_smiles set -> module text_input picks it up

Existing molecule -> "Edit in Structure Editor" -> editor_smiles set -> Ketcher loads it
```

## Files Modified

| File | Change |
|------|--------|
| `requirements.txt` | Add `streamlit-ketcher` |
| `app.py` | Add sidebar editor section after existing sidebar content |
| `modules/molecular_input.py` | Read `active_smiles` as default for SMILES input; add "Edit in Structure Editor" button |
| `modules/property_calc.py` | Add "Edit in Structure Editor" button next to molecule display |
| `modules/drug_optimization.py` | Add "Edit in Structure Editor" button next to lead/analog displays |
| `modules/sar_analysis.py` | Add "Edit in Structure Editor" button next to selected molecules |

**New files:** None. All editor logic goes in `app.py` sidebar section.

## What Does NOT Change

- All existing input methods (SMILES text, file upload, PubChem, batch) remain untouched
- Validation pipeline — editor output goes through the same `validate_smiles()` flow
- Database save flow
- Property calculation
- All other module functionality

## Validation

- Editor SMILES output validated through existing `validate_smiles()` before any use
- Invalid/empty structures show "No structure" status, "Use This Molecule" button disabled
- Molecular formula displayed as confirmation when structure is valid
