"""Molecular Design Suite - Main Streamlit Application."""

import streamlit as st
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import DatabaseManager
from modules.molecular_input import render_molecular_input
from modules.property_calc import render_property_calculation
from modules.protein_analysis import render_protein_analysis
from modules.sar_analysis import render_sar_analysis
from modules.drug_optimization import render_drug_optimization
from streamlit_ketcher import st_ketcher
from utils.rdkit_utils import validate_smiles, mol_from_smiles


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


def main():
    st.set_page_config(
        page_title="Molecular Design Suite",
        page_icon="\U0001f9ec",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    db = DatabaseManager()

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.title("\U0001f9ec Molecular Design Suite")
        st.caption("Drug Discovery Research Platform")

        st.divider()

        # Project selector
        st.markdown("### Project")
        projects = db.get_projects()
        project_names = ["(All / No Project)"] + [p["name"] for p in projects]
        selected_project = st.selectbox("Active Project", project_names)

        if selected_project == "(All / No Project)":
            st.session_state["current_project_id"] = None
        else:
            for p in projects:
                if p["name"] == selected_project:
                    st.session_state["current_project_id"] = p["id"]
                    break

        # Create new project
        with st.expander("Create New Project"):
            new_name = st.text_input("Project name", key="new_proj_name")
            new_desc = st.text_input("Description", key="new_proj_desc")
            if st.button("Create Project") and new_name:
                try:
                    db.create_project(new_name, new_desc)
                    st.success(f"Created project: {new_name}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        st.divider()

        # Module navigation
        st.markdown("### Navigation")
        module = st.radio(
            "Select Module",
            [
                "Molecular Input & Design",
                "Property Calculation",
                "Protein Analysis",
                "SAR Analysis",
                "Drug Optimization",
                "Data Management",
            ],
            label_visibility="collapsed",
        )

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
                    from utils.rdkit_utils import canonical_smiles, calculate_basic_properties
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    st.success(f"Molecule ready: {formula}")
                    st.code(current_smiles, language=None)

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("Use This Molecule", key="use_molecule_btn", type="primary"):
                            st.session_state["active_smiles"] = current_smiles
                            st.rerun()
                    with btn_col2:
                        if st.button("Save to Database", key="save_editor_mol_btn"):
                            canon = canonical_smiles(current_smiles)
                            props = calculate_basic_properties(mol)
                            mol_id = db.add_molecule(
                                smiles=current_smiles,
                                canonical_smiles=canon,
                                name="",
                                formula=props["formula"],
                                mw=props["molecular_weight"],
                                logp=props["logP"],
                                tpsa=props["tpsa"],
                                hbd=props["hbd"],
                                hba=props["hba"],
                                rotatable_bonds=props["rotatable_bonds"],
                                aromatic_rings=props["aromatic_rings"],
                                project_id=st.session_state.get("current_project_id"),
                            )
                            st.success(f"Saved (ID: {mol_id})")
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

    # ── Main Content ──────────────────────────────────────────
    if module == "Molecular Input & Design":
        render_molecular_input(db)

    elif module == "Property Calculation":
        render_property_calculation(db)

    elif module == "Protein Analysis":
        render_protein_analysis(db)

    elif module == "SAR Analysis":
        render_sar_analysis(db)

    elif module == "Drug Optimization":
        render_drug_optimization(db)

    elif module == "Data Management":
        render_data_management(db)


def render_data_management(db: DatabaseManager):
    """Data management module for viewing and managing stored data."""
    st.header("Data Management")

    tab1, tab2, tab3, tab4 = st.tabs(["Molecules", "Proteins", "Experiments", "QSAR Models"])

    with tab1:
        st.subheader("Stored Molecules")
        project_id = st.session_state.get("current_project_id")
        molecules = db.get_molecules(project_id=project_id, limit=500)

        if molecules:
            import pandas as pd
            df = pd.DataFrame(molecules)
            display_cols = ["id", "name", "canonical_smiles", "molecular_weight",
                            "logp", "tpsa", "hbd", "hba", "created_date"]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_cols], use_container_width=True)

            # Delete molecule
            mol_ids = df["id"].tolist()
            del_id = st.selectbox("Select molecule ID to delete", mol_ids, key="del_mol")
            if st.button("Delete Selected Molecule", key="del_mol_btn"):
                db.delete_molecule(del_id)
                st.success(f"Deleted molecule ID {del_id}")
                st.rerun()

            # Export
            csv = df[available_cols].to_csv(index=False)
            st.download_button("Export All as CSV", csv, "molecules.csv", "text/csv")
        else:
            st.info("No molecules stored yet.")

    with tab2:
        st.subheader("Stored Proteins")
        proteins = db.get_proteins(project_id=st.session_state.get("current_project_id"))
        if proteins:
            import pandas as pd
            df = pd.DataFrame(proteins)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No proteins stored yet.")

    with tab3:
        st.subheader("Experiments")
        experiments = db.get_experiments(project_id=st.session_state.get("current_project_id"))
        if experiments:
            import pandas as pd
            df = pd.DataFrame(experiments)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No experiments recorded yet.")

        # Add experiment
        with st.expander("Record New Experiment"):
            molecules = db.get_molecules(limit=200)
            proteins = db.get_proteins()

            if molecules:
                mol_options = {f"{m['name'] or m['canonical_smiles'][:30]} (ID:{m['id']})": m["id"]
                               for m in molecules}
                sel_mol = st.selectbox("Molecule", list(mol_options.keys()), key="exp_mol")
                mol_id = mol_options[sel_mol]
            else:
                st.info("Add molecules first.")
                return

            protein_id = None
            if proteins:
                prot_options = {"None": None}
                prot_options.update({
                    f"{p['pdb_id'] or p['name']} (ID:{p['id']})": p["id"]
                    for p in proteins
                })
                sel_prot = st.selectbox("Protein target", list(prot_options.keys()), key="exp_prot")
                protein_id = prot_options[sel_prot]

            assay_type = st.text_input("Assay type", key="exp_assay",
                                        placeholder="e.g., IC50, EC50, Ki")
            activity_val = st.number_input("Activity value", key="exp_val")
            activity_unit = st.selectbox("Unit", ["nM", "uM", "mM", "pIC50", "%"],
                                          key="exp_unit")
            notes = st.text_area("Notes", key="exp_notes")

            if st.button("Save Experiment"):
                db.add_experiment(
                    molecule_id=mol_id,
                    protein_id=protein_id,
                    assay_type=assay_type,
                    activity_value=activity_val,
                    activity_unit=activity_unit,
                    notes=notes,
                    project_id=st.session_state.get("current_project_id"),
                )
                st.success("Experiment saved!")
                st.rerun()

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


if __name__ == "__main__":
    main()
