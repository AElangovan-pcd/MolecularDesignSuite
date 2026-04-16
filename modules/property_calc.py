"""Property Calculation Module for the Streamlit app."""

import streamlit as st
import pandas as pd
import numpy as np

from rdkit import Chem

from utils.rdkit_utils import (
    mol_from_smiles, calculate_basic_properties, all_drug_likeness_filters,
    admet_properties, check_all_alerts, calculate_descriptor_set,
)
from utils.visualization import (
    mol_to_svg, property_radar_chart, drug_likeness_summary_chart,
    property_distribution_plot, correlation_heatmap,
)
from database.db_manager import DatabaseManager
from utils.editor_helpers import edit_in_editor_button


def render_property_calculation(db: DatabaseManager):
    """Render the Property Calculation module."""
    st.header("Property Calculation")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Single Molecule", "Drug-Likeness Filters",
        "ADMET Predictions", "Batch Analysis"
    ])

    with tab1:
        _single_molecule_properties(db)

    with tab2:
        _drug_likeness_tab(db)

    with tab3:
        _admet_tab(db)

    with tab4:
        _batch_analysis_tab(db)


def _get_molecule_input(key_prefix: str = "prop") -> tuple:
    """Common molecule input widget. Returns (smiles, mol) or (None, None)."""
    input_method = st.radio(
        "Input method", ["SMILES", "From Database"],
        horizontal=True, key=f"{key_prefix}_input_method",
    )
    mol = None
    smiles = None
    if input_method == "SMILES":
        smiles = st.text_input("SMILES", key=f"{key_prefix}_smiles",
                               placeholder="e.g., c1ccccc1")
        if smiles:
            mol = mol_from_smiles(smiles)
            if mol is None:
                st.error("Invalid SMILES")
            else:
                edit_in_editor_button(smiles, key=f"{key_prefix}_edit_btn")
    else:
        db = DatabaseManager()
        molecules = db.get_molecules(limit=200)
        if molecules:
            options = {f"{m['name'] or m['canonical_smiles'][:30]} (ID:{m['id']})": m
                       for m in molecules}
            selected = st.selectbox("Select molecule", list(options.keys()),
                                    key=f"{key_prefix}_db_select")
            if selected:
                mol_data = options[selected]
                smiles = mol_data["smiles"]
                mol = mol_from_smiles(smiles)
                if mol:
                    edit_in_editor_button(smiles, key=f"{key_prefix}_db_edit_btn")
        else:
            st.info("No molecules in database. Add some in the Molecular Input module.")
    return smiles, mol


def _single_molecule_properties(db: DatabaseManager):
    """Calculate and display properties for a single molecule."""
    st.subheader("Single Molecule Properties")

    smiles, mol = _get_molecule_input("single_prop")
    if mol is None:
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        svg = mol_to_svg(mol, size=(350, 250))
        st.image(svg, use_container_width=True)

    props = calculate_basic_properties(mol)

    with col2:
        st.markdown("#### Calculated Properties")
        prop_df = pd.DataFrame([
            {"Property": k, "Value": v} for k, v in props.items()
        ])
        st.dataframe(prop_df, hide_index=True, use_container_width=True)

    # Radar chart
    st.plotly_chart(
        property_radar_chart(
            {"MW": props["molecular_weight"], "LogP": props["logP"],
             "TPSA": props["tpsa"], "HBD": props["hbd"], "HBA": props["hba"],
             "RotBonds": props["rotatable_bonds"], "QED": props["qed"]},
            title="Property Profile"
        ),
        use_container_width=True,
    )


def _drug_likeness_tab(db: DatabaseManager):
    """Drug-likeness filter evaluation."""
    st.subheader("Drug-Likeness Filters")

    smiles, mol = _get_molecule_input("druglike")
    if mol is None:
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        svg = mol_to_svg(mol, size=(250, 180))
        st.image(svg, use_container_width=True)

    filters = all_drug_likeness_filters(mol)

    with col2:
        st.plotly_chart(
            drug_likeness_summary_chart(filters),
            use_container_width=True,
        )

    # Detailed results
    for filter_name, results in filters.items():
        with st.expander(f"{filter_name} {'PASS' if results['passes'] else 'FAIL'}", expanded=False):
            for rule, value in results.items():
                if rule == "passes":
                    continue
                if rule == "violations":
                    st.text(f"  Total violations: {value}")
                else:
                    icon = "+" if value else "-"
                    st.text(f"  [{icon}] {rule}")

    # Structural alerts
    alerts = check_all_alerts(mol)
    with st.expander("Structural Alerts (PAINS)"):
        if alerts["has_pains"]:
            st.warning(f"PAINS alerts found: {', '.join(alerts['pains_alerts'])}")
        else:
            st.success("No PAINS alerts detected")


def _admet_tab(db: DatabaseManager):
    """ADMET property predictions."""
    st.subheader("ADMET Property Predictions")
    st.caption("Note: These are rule-based estimates, not ML predictions.")

    smiles, mol = _get_molecule_input("admet")
    if mol is None:
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        svg = mol_to_svg(mol, size=(300, 200))
        st.image(svg, use_container_width=True)

    admet = admet_properties(mol)

    with col2:
        st.markdown("#### ADMET Estimates")
        for prop, value in admet.items():
            if isinstance(value, bool):
                icon = "Yes" if value else "No"
                st.text(f"  {prop}: {icon}")
            else:
                st.text(f"  {prop}: {value}")


def _batch_analysis_tab(db: DatabaseManager):
    """Batch property calculation for multiple molecules."""
    st.subheader("Batch Property Analysis")

    source = st.radio("Molecule source", ["Database", "Paste SMILES"], horizontal=True)

    molecules_data = []
    if source == "Database":
        mols = db.get_molecules(limit=500)
        if not mols:
            st.info("No molecules in database.")
            return
        for m in mols:
            mol = mol_from_smiles(m["smiles"])
            if mol:
                molecules_data.append({"name": m["name"], "smiles": m["smiles"], "mol": mol})
    else:
        text = st.text_area("SMILES (one per line)", height=150)
        if text:
            for line in text.strip().split("\n"):
                parts = line.strip().split("\t")
                smi = parts[0].strip()
                name = parts[1].strip() if len(parts) > 1 else ""
                mol = mol_from_smiles(smi)
                if mol:
                    molecules_data.append({"name": name, "smiles": smi, "mol": mol})

    if not molecules_data:
        return

    st.info(f"Analyzing {len(molecules_data)} molecules...")

    # Calculate properties for all
    rows = []
    for md in molecules_data:
        props = calculate_basic_properties(md["mol"])
        row = {"Name": md["name"], "SMILES": md["smiles"]}
        row.update(props)
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Distribution plots
    st.subheader("Property Distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_prop = st.selectbox("Select property", numeric_cols)
    if selected_prop:
        st.plotly_chart(
            property_distribution_plot(df, selected_prop),
            use_container_width=True,
        )

    # Correlation heatmap
    if len(df) > 2:
        st.subheader("Property Correlations")
        st.plotly_chart(
            correlation_heatmap(df),
            use_container_width=True,
        )

    # Download
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "molecular_properties.csv", "text/csv")
