"""Molecular Input & Design Module for the Streamlit app."""

import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

from utils.rdkit_utils import (
    mol_from_smiles, canonical_smiles, validate_smiles,
    calculate_basic_properties, generate_3d_coords,
)
from utils.visualization import mol_to_svg, mol_to_png_bytes, mol_grid_image
from utils.file_handlers import (
    parse_sdf_file, parse_smiles_file, parse_csv_with_smiles,
)
from database.db_manager import DatabaseManager


def render_molecular_input(db: DatabaseManager):
    """Render the Molecular Input & Design module."""
    st.header("Molecular Input & Design")

    tab1, tab2, tab3, tab4 = st.tabs([
        "SMILES Input", "File Upload", "PubChem Lookup", "Batch Processing"
    ])

    with tab1:
        _smiles_input_tab(db)

    with tab2:
        _file_upload_tab(db)

    with tab3:
        _pubchem_lookup_tab(db)

    with tab4:
        _batch_processing_tab(db)


def _smiles_input_tab(db: DatabaseManager):
    """Single SMILES input with validation and visualization."""
    st.subheader("Enter SMILES")

    col1, col2 = st.columns([2, 1])
    with col1:
        smiles = st.text_input(
            "SMILES string",
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
            help="Enter a valid SMILES string for your molecule",
        )
        mol_name = st.text_input("Molecule name (optional)", placeholder="e.g., Aspirin")

    if smiles:
        is_valid, message = validate_smiles(smiles)
        if is_valid:
            mol = mol_from_smiles(smiles)
            canon = canonical_smiles(smiles)

            with col2:
                st.success(message)
                svg = mol_to_svg(mol, size=(350, 250))
                st.image(svg, use_container_width=True)

            props = calculate_basic_properties(mol)
            st.subheader("Basic Properties")
            prop_cols = st.columns(4)
            display_props = [
                ("MW", props["molecular_weight"], "g/mol"),
                ("LogP", props["logP"], ""),
                ("TPSA", props["tpsa"], "A\u00b2"),
                ("HBD", props["hbd"], ""),
                ("HBA", props["hba"], ""),
                ("Rot. Bonds", props["rotatable_bonds"], ""),
                ("Aromatic Rings", props["aromatic_rings"], ""),
                ("QED", props["qed"], ""),
            ]
            for i, (label, value, unit) in enumerate(display_props):
                with prop_cols[i % 4]:
                    st.metric(label, f"{value} {unit}".strip())

            st.text(f"Canonical SMILES: {canon}")
            st.text(f"Formula: {props['formula']}")

            # Save to database
            if st.button("Save Molecule to Database", key="save_single"):
                project_id = st.session_state.get("current_project_id")
                mol_id = db.add_molecule(
                    smiles=smiles,
                    canonical_smiles=canon,
                    name=mol_name,
                    formula=props["formula"],
                    mw=props["molecular_weight"],
                    logp=props["logP"],
                    tpsa=props["tpsa"],
                    hbd=props["hbd"],
                    hba=props["hba"],
                    rotatable_bonds=props["rotatable_bonds"],
                    aromatic_rings=props["aromatic_rings"],
                    project_id=project_id,
                )
                st.success(f"Saved molecule (ID: {mol_id})")
        else:
            with col2:
                st.error(message)


def _file_upload_tab(db: DatabaseManager):
    """File upload for SDF, MOL, SMILES, CSV files."""
    st.subheader("Upload Molecular File")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["sdf", "mol", "smi", "csv", "txt"],
        help="Supported formats: SDF, MOL, SMILES (.smi/.txt), CSV with SMILES column",
    )

    if uploaded_file is not None:
        filename = uploaded_file.name.lower()

        if filename.endswith(".sdf"):
            molecules = parse_sdf_file(uploaded_file.read())
            st.info(f"Loaded {len(molecules)} molecules from SDF file")
            _display_molecule_list(molecules, db)

        elif filename.endswith(".mol"):
            content = uploaded_file.read().decode("utf-8")
            mol = Chem.MolFromMolBlock(content)
            if mol:
                smiles = Chem.MolToSmiles(mol)
                st.success(f"Loaded molecule: {smiles}")
                svg = mol_to_svg(mol, size=(400, 300))
                st.image(svg, use_container_width=True)
            else:
                st.error("Failed to parse MOL file")

        elif filename.endswith((".smi", ".txt")):
            content = uploaded_file.read().decode("utf-8")
            delimiter = st.selectbox("Delimiter", ["\t", ",", " "], index=0)
            molecules = parse_smiles_file(content, delimiter=delimiter)
            st.info(f"Loaded {len(molecules)} molecules")
            _display_molecule_list(molecules, db)

        elif filename.endswith(".csv"):
            content = uploaded_file.read().decode("utf-8")
            try:
                df = parse_csv_with_smiles(content)
                valid_count = df["valid"].sum()
                st.info(f"Loaded {valid_count}/{len(df)} valid molecules from CSV")
                st.dataframe(df.drop(columns=["ROMol"]).head(20))
            except ValueError as e:
                st.error(str(e))


def _pubchem_lookup_tab(db: DatabaseManager):
    """Look up molecules from PubChem by name or CID."""
    st.subheader("PubChem Lookup")

    search_type = st.radio("Search by", ["Name", "CID"], horizontal=True)
    query = st.text_input(
        f"Enter molecule {search_type.lower()}",
        placeholder="e.g., Aspirin" if search_type == "Name" else "e.g., 2244",
    )

    if query and st.button("Search PubChem"):
        try:
            import pubchempy as pcp

            with st.spinner("Searching PubChem..."):
                if search_type == "Name":
                    compounds = pcp.get_compounds(query, "name")
                else:
                    compounds = pcp.get_compounds(int(query), "cid")

            if compounds:
                for comp in compounds[:5]:
                    st.markdown(f"**{comp.iupac_name or 'Unknown'}** (CID: {comp.cid})")
                    mol = mol_from_smiles(comp.isomeric_smiles)
                    if mol:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            svg = mol_to_svg(mol, size=(300, 200))
                            st.image(svg, use_container_width=True)
                        with col2:
                            st.text(f"SMILES: {comp.isomeric_smiles}")
                            st.text(f"Formula: {comp.molecular_formula}")
                            st.text(f"MW: {comp.molecular_weight}")
                            st.text(f"XLogP: {comp.xlogp}")

                        if st.button(f"Save CID {comp.cid}", key=f"save_pubchem_{comp.cid}"):
                            props = calculate_basic_properties(mol)
                            project_id = st.session_state.get("current_project_id")
                            db.add_molecule(
                                smiles=comp.isomeric_smiles,
                                canonical_smiles=comp.canonical_smiles,
                                name=comp.iupac_name or query,
                                formula=comp.molecular_formula,
                                mw=props["molecular_weight"],
                                logp=props["logP"],
                                tpsa=props["tpsa"],
                                hbd=props["hbd"],
                                hba=props["hba"],
                                rotatable_bonds=props["rotatable_bonds"],
                                aromatic_rings=props["aromatic_rings"],
                                project_id=project_id,
                            )
                            st.success("Saved to database!")
                    st.divider()
            else:
                st.warning("No compounds found")
        except ImportError:
            st.error("pubchempy is not installed. Run: pip install pubchempy")
        except Exception as e:
            st.error(f"PubChem search failed: {e}")


def _batch_processing_tab(db: DatabaseManager):
    """Batch input of multiple SMILES."""
    st.subheader("Batch SMILES Input")

    smiles_text = st.text_area(
        "Enter SMILES (one per line, optionally with name separated by tab/comma)",
        height=200,
        placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\tAspirin\nCC(=O)NC1=CC=C(O)C=C1\tAcetaminophen",
    )

    if smiles_text and st.button("Process Batch"):
        lines = [l.strip() for l in smiles_text.strip().split("\n") if l.strip()]
        results = []
        for line in lines:
            parts = line.replace(",", "\t").split("\t")
            smi = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else ""
            mol = mol_from_smiles(smi)
            if mol:
                props = calculate_basic_properties(mol)
                results.append({
                    "Name": name,
                    "SMILES": Chem.MolToSmiles(mol),
                    "MW": props["molecular_weight"],
                    "LogP": props["logP"],
                    "TPSA": props["tpsa"],
                    "HBD": props["hbd"],
                    "HBA": props["hba"],
                    "QED": props["qed"],
                    "_mol": mol,
                })
            else:
                results.append({"Name": name, "SMILES": smi, "MW": None, "LogP": None,
                                "TPSA": None, "HBD": None, "HBA": None, "QED": None,
                                "_mol": None})

        df = pd.DataFrame(results)
        valid = df[df["MW"].notna()]
        invalid = df[df["MW"].isna()]

        st.info(f"Valid: {len(valid)} | Invalid: {len(invalid)}")
        st.dataframe(df.drop(columns=["_mol"]))

        # Show grid image of valid molecules
        mols = [r["_mol"] for r in results if r["_mol"] is not None]
        legends = [r["Name"] or r["SMILES"][:20] for r in results if r["_mol"] is not None]
        if mols:
            grid_bytes = mol_grid_image(mols, legends, mols_per_row=4)
            st.image(grid_bytes, caption="Batch Molecules", use_container_width=True)

        # Save all valid to database
        if st.button("Save All Valid to Database", key="save_batch"):
            project_id = st.session_state.get("current_project_id")
            saved = 0
            for r in results:
                if r["_mol"] is not None:
                    canon = Chem.MolToSmiles(r["_mol"])
                    props = calculate_basic_properties(r["_mol"])
                    db.add_molecule(
                        smiles=r["SMILES"], canonical_smiles=canon,
                        name=r["Name"], formula=props["formula"],
                        mw=props["molecular_weight"], logp=props["logP"],
                        tpsa=props["tpsa"], hbd=props["hbd"], hba=props["hba"],
                        rotatable_bonds=props["rotatable_bonds"],
                        aromatic_rings=props["aromatic_rings"],
                        project_id=project_id,
                    )
                    saved += 1
            st.success(f"Saved {saved} molecules to database")


def _display_molecule_list(molecules: list[dict], db: DatabaseManager):
    """Display a list of parsed molecules with a grid image."""
    if not molecules:
        return
    mols = [m["mol"] for m in molecules if m.get("mol")]
    legends = [m.get("name", "") or m.get("smiles", "")[:20] for m in molecules]
    if mols:
        grid_bytes = mol_grid_image(mols[:20], legends[:20], mols_per_row=4)
        st.image(grid_bytes, caption=f"Showing first {min(20, len(mols))} molecules",
                 use_container_width=True)

    if st.button("Save All to Database", key="save_file_upload"):
        project_id = st.session_state.get("current_project_id")
        saved = 0
        for m in molecules:
            mol = m.get("mol")
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            props = calculate_basic_properties(mol)
            db.add_molecule(
                smiles=smi, canonical_smiles=smi,
                name=m.get("name", ""), formula=props["formula"],
                mw=props["molecular_weight"], logp=props["logP"],
                tpsa=props["tpsa"], hbd=props["hbd"], hba=props["hba"],
                rotatable_bonds=props["rotatable_bonds"],
                aromatic_rings=props["aromatic_rings"],
                project_id=project_id,
            )
            saved += 1
        st.success(f"Saved {saved} molecules")
