"""Drug Design Optimization Module."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, DataStructs, rdFMCS

from utils.rdkit_utils import (
    mol_from_smiles, mol_to_smiles, calculate_basic_properties,
    all_drug_likeness_filters, admet_properties, morgan_fingerprint,
    tanimoto_similarity, get_murcko_scaffold, generate_3d_coords,
    synthetic_accessibility_score, check_all_alerts,
)
from utils.visualization import (
    mol_to_svg, mol_grid_image, multi_property_comparison,
    scatter_plot, property_radar_chart,
)
from database.db_manager import DatabaseManager
from utils.qsar import load_model_artifact, predict as qsar_predict
from utils.editor_helpers import edit_in_editor_button


def render_drug_optimization(db: DatabaseManager):
    """Render the Drug Optimization module."""
    st.header("Drug Design & Optimization")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Lead Profiling", "Analog Generation",
        "Scaffold Hopping", "Multi-Parameter Optimization",
        "Molecule Comparison"
    ])

    with tab1:
        _lead_profiling_tab(db)

    with tab2:
        _analog_generation_tab(db)

    with tab3:
        _scaffold_hopping_tab(db)

    with tab4:
        _mpo_tab(db)

    with tab5:
        _comparison_tab(db)


def _lead_profiling_tab(db: DatabaseManager):
    """Comprehensive profiling of a lead compound."""
    st.subheader("Lead Compound Profiling")

    smiles = st.text_input("Lead SMILES", placeholder="e.g., c1ccc2c(c1)cc1ccc3ccccc3c1n2",
                           key="lead_smiles")
    if not smiles:
        return

    mol = mol_from_smiles(smiles)
    if mol is None:
        st.error("Invalid SMILES")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        svg = mol_to_svg(mol, size=(350, 250))
        st.image(svg, use_container_width=True)
        edit_in_editor_button(smiles, key="lead_edit_btn")

    # Basic properties
    props = calculate_basic_properties(mol)
    with col2:
        st.markdown("#### Molecular Properties")
        metrics_row = st.columns(4)
        metric_items = [
            ("MW", f"{props['molecular_weight']:.1f}"),
            ("LogP", f"{props['logP']:.2f}"),
            ("TPSA", f"{props['tpsa']:.1f}"),
            ("QED", f"{props['qed']:.3f}"),
            ("HBD", str(props["hbd"])),
            ("HBA", str(props["hba"])),
            ("Rot. Bonds", str(props["rotatable_bonds"])),
            ("Fsp3", f"{props['fraction_csp3']:.2f}"),
        ]
        for i, (label, val) in enumerate(metric_items):
            with metrics_row[i % 4]:
                st.metric(label, val)

    # Drug-likeness
    st.markdown("#### Drug-Likeness Assessment")
    filters = all_drug_likeness_filters(mol)
    filter_cols = st.columns(5)
    for i, (name, result) in enumerate(filters.items()):
        with filter_cols[i]:
            status = "PASS" if result["passes"] else "FAIL"
            st.metric(name, status)

    # ADMET
    st.markdown("#### ADMET Estimates")
    admet = admet_properties(mol)
    admet_cols = st.columns(4)
    admet_items = list(admet.items())
    for i, (key, val) in enumerate(admet_items):
        with admet_cols[i % 4]:
            display_val = str(val) if isinstance(val, bool) else f"{val}"
            st.metric(key.replace("_", " ").title(), display_val)

    # Alerts
    alerts = check_all_alerts(mol)
    if alerts["has_pains"]:
        st.warning(f"PAINS Alerts: {', '.join(alerts['pains_alerts'])}")
    else:
        st.success("No structural alerts detected")

    # SA Score
    sa = synthetic_accessibility_score(mol)
    st.metric("Synthetic Accessibility", f"{sa}/10 ({'Easy' if sa < 4 else 'Moderate' if sa < 6 else 'Hard'})")


def _analog_generation_tab(db: DatabaseManager):
    """Generate analogs by enumeration of simple modifications."""
    st.subheader("Analog Generation")
    st.caption("Generate analogs by applying common medicinal chemistry transformations.")

    smiles = st.text_input("Starting SMILES", key="analog_smiles",
                           placeholder="e.g., c1ccccc1O")
    if not smiles:
        return

    mol = mol_from_smiles(smiles)
    if mol is None:
        st.error("Invalid SMILES")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        svg = mol_to_svg(mol, size=(250, 180))
        st.image(svg, use_container_width=True)
        edit_in_editor_button(smiles, key="analog_src_edit_btn")

    # Define simple transformations using SMIRKS patterns
    transformations = {
        "Methylation (OH -> OMe)": ("[OH:1]", "[O:1]C"),
        "Fluorination (H -> F)": ("[cH:1]", "[c:1]F"),
        "Chlorination (H -> Cl)": ("[cH:1]", "[c:1]Cl"),
        "Amine introduction": ("[cH:1]", "[c:1]N"),
        "Hydroxyl introduction": ("[cH:1]", "[c:1]O"),
        "Methyl introduction": ("[cH:1]", "[c:1]C"),
        "Nitrile introduction": ("[cH:1]", "[c:1]C#N"),
        "Trifluoromethyl": ("[cH:1]", "[c:1]C(F)(F)F"),
    }

    selected_transforms = st.multiselect(
        "Select transformations",
        list(transformations.keys()),
        default=list(transformations.keys())[:3],
    )

    if not st.button("Generate Analogs"):
        return

    analogs = set()
    analogs.add(Chem.MolToSmiles(mol))  # Include parent

    for transform_name in selected_transforms:
        reactant_smarts, product_smarts = transformations[transform_name]
        try:
            rxn_smarts = f"[{reactant_smarts[1:-1]}]>>[{product_smarts[1:-1]}]"
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            if rxn is None:
                continue
            products = rxn.RunReactants((mol,))
            for product_set in products:
                for p in product_set:
                    try:
                        Chem.SanitizeMol(p)
                        smi = Chem.MolToSmiles(p)
                        if smi and smi != Chem.MolToSmiles(mol):
                            analogs.add(smi)
                    except Exception:
                        continue
        except Exception:
            continue

    parent_smi = Chem.MolToSmiles(mol)
    analog_list = [s for s in analogs if s != parent_smi]

    if not analog_list:
        st.warning("No analogs generated. Try different transformations.")
        return

    st.success(f"Generated {len(analog_list)} unique analogs")

    # Calculate properties for all analogs
    rows = []
    valid_mols = []
    for smi in analog_list:
        m = mol_from_smiles(smi)
        if m:
            props = calculate_basic_properties(m)
            sim = tanimoto_similarity(mol, m)
            rows.append({
                "SMILES": smi,
                "MW": props["molecular_weight"],
                "LogP": props["logP"],
                "TPSA": props["tpsa"],
                "QED": props["qed"],
                "Similarity": sim,
            })
            valid_mols.append(m)

    df = pd.DataFrame(rows).sort_values("QED", ascending=False)
    st.dataframe(df, use_container_width=True)

    # Grid image
    if valid_mols:
        grid = mol_grid_image(valid_mols[:16],
                              [r["SMILES"][:20] for r in rows[:16]],
                              mols_per_row=4)
        st.image(grid, caption="Generated Analogs", use_container_width=True)


def _scaffold_hopping_tab(db: DatabaseManager):
    """Scaffold hopping / bioisosteric replacement suggestions."""
    st.subheader("Scaffold Hopping")
    st.caption("Find molecules with different scaffolds but similar properties.")

    smiles = st.text_input("Reference SMILES", key="scaffold_hop_smiles")
    if not smiles:
        return

    mol = mol_from_smiles(smiles)
    if mol is None:
        st.error("Invalid SMILES")
        return

    svg = mol_to_svg(mol, size=(300, 200))
    st.image(svg, use_container_width=True)

    # Common bioisosteric replacements
    bioisosteres = {
        "Carboxylic acid <-> Tetrazole": {
            "from": "C(=O)O",
            "to": "c1nnn[nH]1",
        },
        "Ester <-> Amide": {
            "from": "C(=O)O",
            "to": "C(=O)N",
        },
        "Phenyl <-> Pyridine": {
            "from": "c1ccccc1",
            "to": "c1ccncc1",
        },
        "Phenyl <-> Thiophene": {
            "from": "c1ccccc1",
            "to": "c1ccsc1",
        },
        "NH <-> O": {
            "from": "[NH]",
            "to": "O",
        },
        "CH2 <-> O": {
            "from": "CC",
            "to": "CO",
        },
        "F <-> Cl": {
            "from": "F",
            "to": "Cl",
        },
    }

    parent_smi = Chem.MolToSmiles(mol)
    results = []

    for name, replacement in bioisosteres.items():
        from_pat = Chem.MolFromSmarts(replacement["from"])
        if from_pat and mol.HasSubstructMatch(from_pat):
            # Try the replacement
            try:
                to_mol = Chem.MolFromSmiles(replacement["to"])
                if to_mol is None:
                    continue
                replaced = AllChem.ReplaceSubstructs(mol, from_pat, to_mol)
                for r in replaced:
                    try:
                        Chem.SanitizeMol(r)
                        new_smi = Chem.MolToSmiles(r)
                        if new_smi != parent_smi:
                            new_mol = mol_from_smiles(new_smi)
                            if new_mol:
                                props = calculate_basic_properties(new_mol)
                                sim = tanimoto_similarity(mol, new_mol)
                                results.append({
                                    "Replacement": name,
                                    "SMILES": new_smi,
                                    "MW": props["molecular_weight"],
                                    "LogP": props["logP"],
                                    "QED": props["qed"],
                                    "Similarity": sim,
                                    "mol": new_mol,
                                })
                    except Exception:
                        continue
            except Exception:
                continue

    if results:
        st.success(f"Found {len(results)} bioisosteric replacements")
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "mol"} for r in results])
        st.dataframe(df, use_container_width=True)

        mols_to_show = [r["mol"] for r in results[:12]]
        legends = [r["Replacement"][:25] for r in results[:12]]
        if mols_to_show:
            grid = mol_grid_image(mols_to_show, legends, mols_per_row=4)
            st.image(grid, caption="Scaffold Hop Results", use_container_width=True)
    else:
        st.info("No applicable bioisosteric replacements found for this molecule.")


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


def _comparison_tab(db: DatabaseManager):
    """Side-by-side molecule comparison."""
    st.subheader("Molecule Comparison")

    col1, col2 = st.columns(2)
    with col1:
        smiles1 = st.text_input("Molecule 1 SMILES", key="cmp_smi1",
                                 placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")
    with col2:
        smiles2 = st.text_input("Molecule 2 SMILES", key="cmp_smi2",
                                 placeholder="e.g., CC(=O)NC1=CC=C(O)C=C1")

    if not smiles1 or not smiles2:
        return

    mol1 = mol_from_smiles(smiles1)
    mol2 = mol_from_smiles(smiles2)

    if mol1 is None or mol2 is None:
        st.error("One or both SMILES are invalid")
        return

    # Structures
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Molecule 1**")
        svg1 = mol_to_svg(mol1, size=(350, 250))
        st.image(svg1, use_container_width=True)
    with col2:
        st.markdown("**Molecule 2**")
        svg2 = mol_to_svg(mol2, size=(350, 250))
        st.image(svg2, use_container_width=True)

    # Similarity
    sim = tanimoto_similarity(mol1, mol2)
    st.metric("Tanimoto Similarity", f"{sim:.4f}")

    # Properties comparison
    props1 = calculate_basic_properties(mol1)
    props2 = calculate_basic_properties(mol2)

    compare_props = ["molecular_weight", "logP", "tpsa", "hbd", "hba",
                     "rotatable_bonds", "aromatic_rings", "qed"]
    comp_df = pd.DataFrame({
        "Property": compare_props,
        "Molecule 1": [props1[p] for p in compare_props],
        "Molecule 2": [props2[p] for p in compare_props],
        "Difference": [round(props1[p] - props2[p], 3) for p in compare_props],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Drug-likeness comparison
    st.markdown("#### Drug-Likeness")
    filters1 = all_drug_likeness_filters(mol1)
    filters2 = all_drug_likeness_filters(mol2)
    filter_comp = []
    for name in filters1:
        filter_comp.append({
            "Filter": name,
            "Mol 1": "Pass" if filters1[name]["passes"] else "Fail",
            "Mol 2": "Pass" if filters2[name]["passes"] else "Fail",
        })
    st.dataframe(pd.DataFrame(filter_comp), hide_index=True, use_container_width=True)

    # Maximum Common Substructure
    try:
        mcs_result = rdFMCS.FindMCS([mol1, mol2], timeout=10)
        if mcs_result.smartsString:
            mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            st.markdown(f"**Maximum Common Substructure:** {mcs_result.numAtoms} atoms, {mcs_result.numBonds} bonds")
            if mcs_mol:
                svg = mol_to_svg(mcs_mol, size=(300, 200))
                st.image(svg, use_container_width=True)
    except Exception:
        pass
