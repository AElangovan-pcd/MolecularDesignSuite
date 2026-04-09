"""Protein Structure Analysis Module for the Streamlit app."""

import streamlit as st
import pandas as pd
import io
import os
from typing import Optional

from database.db_manager import DatabaseManager


def render_protein_analysis(db: DatabaseManager):
    """Render the Protein Analysis module."""
    st.header("Protein Structure Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Load Protein", "Sequence Analysis",
        "3D Visualization", "Binding Site Analysis"
    ])

    with tab1:
        _load_protein_tab(db)

    with tab2:
        _sequence_analysis_tab(db)

    with tab3:
        _visualization_tab(db)

    with tab4:
        _binding_site_tab(db)


def _load_protein_tab(db: DatabaseManager):
    """Load protein structure from PDB or file."""
    st.subheader("Load Protein Structure")

    load_method = st.radio("Source", ["PDB ID", "Upload PDB File"], horizontal=True)

    if load_method == "PDB ID":
        pdb_id = st.text_input("PDB ID", placeholder="e.g., 1AKE").strip().upper()

        if pdb_id and st.button("Fetch from PDB"):
            try:
                from Bio.PDB import PDBList, PDBParser
                with st.spinner(f"Downloading {pdb_id}..."):
                    pdb_list = PDBList()
                    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "protein_structures")
                    os.makedirs(data_dir, exist_ok=True)
                    filename = pdb_list.retrieve_pdb_file(pdb_id, pdir=data_dir, file_format="pdb")

                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure(pdb_id, filename)

                    # Extract info
                    model = structure[0]
                    chains = list(model.get_chains())
                    residue_count = sum(1 for r in model.get_residues()
                                        if r.get_id()[0] == " ")
                    atom_count = sum(1 for _ in model.get_atoms())

                    st.success(f"Loaded {pdb_id}")
                    st.text(f"Chains: {len(chains)} ({', '.join(c.id for c in chains)})")
                    st.text(f"Residues: {residue_count}")
                    st.text(f"Atoms: {atom_count}")

                    # Extract sequence
                    from Bio.PDB.Polypeptide import PPBuilder
                    ppb = PPBuilder()
                    sequences = []
                    for pp in ppb.build_peptides(structure):
                        sequences.append(str(pp.get_sequence()))
                    full_seq = "".join(sequences)

                    # Save to database
                    protein_id = db.add_protein(
                        pdb_id=pdb_id,
                        name=pdb_id,
                        sequence=full_seq,
                        structure_file_path=filename,
                        project_id=st.session_state.get("current_project_id"),
                    )
                    st.session_state["current_protein_id"] = protein_id
                    st.session_state["current_pdb_file"] = filename
                    st.session_state["current_pdb_id"] = pdb_id

            except ImportError:
                st.error("BioPython is required. Run: pip install biopython")
            except Exception as e:
                st.error(f"Failed to load PDB: {e}")

    else:
        uploaded = st.file_uploader("Upload PDB file", type=["pdb", "ent"])
        if uploaded is not None:
            try:
                from Bio.PDB import PDBParser

                content = uploaded.read().decode("utf-8")
                # Save file locally
                data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "protein_structures")
                os.makedirs(data_dir, exist_ok=True)
                filepath = os.path.join(data_dir, uploaded.name)
                with open(filepath, "w") as f:
                    f.write(content)

                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("uploaded", filepath)
                model = structure[0]
                chains = list(model.get_chains())
                residue_count = sum(1 for r in model.get_residues() if r.get_id()[0] == " ")

                st.success(f"Loaded {uploaded.name}")
                st.text(f"Chains: {len(chains)}")
                st.text(f"Residues: {residue_count}")

                st.session_state["current_pdb_file"] = filepath

            except ImportError:
                st.error("BioPython is required.")
            except Exception as e:
                st.error(f"Failed to parse PDB file: {e}")

    # Show saved proteins
    st.divider()
    st.subheader("Saved Proteins")
    proteins = db.get_proteins(project_id=st.session_state.get("current_project_id"))
    if proteins:
        df = pd.DataFrame(proteins)
        display_cols = [c for c in ["id", "pdb_id", "name", "organism", "created_date"]
                        if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)
    else:
        st.info("No proteins loaded yet.")


def _sequence_analysis_tab(db: DatabaseManager):
    """Analyze protein sequence composition and properties."""
    st.subheader("Protein Sequence Analysis")

    input_method = st.radio("Input", ["Enter Sequence", "From Database"],
                            horizontal=True, key="seq_input")

    sequence = ""
    if input_method == "Enter Sequence":
        sequence = st.text_area("Protein Sequence (one-letter code)",
                                height=150, placeholder="MKTLLLTLVVVTIVCLDLGYA...")
    else:
        proteins = db.get_proteins()
        if proteins:
            options = {f"{p['pdb_id'] or p['name']} (ID:{p['id']})": p for p in proteins}
            selected = st.selectbox("Select protein", list(options.keys()))
            if selected:
                sequence = options[selected].get("sequence", "")
                if sequence:
                    st.text_area("Sequence", sequence, height=100, disabled=True)
        else:
            st.info("No proteins in database.")

    if not sequence:
        return

    # Clean sequence
    sequence = "".join(c for c in sequence.upper() if c.isalpha())

    st.markdown(f"**Length:** {len(sequence)} residues")

    # Amino acid composition
    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    composition = {aa: sequence.count(aa) for aa in aa_list}
    total = sum(composition.values())
    comp_df = pd.DataFrame([
        {"Amino Acid": aa, "Count": count,
         "Percentage": round(count / total * 100, 1) if total > 0 else 0}
        for aa, count in sorted(composition.items(), key=lambda x: -x[1])
        if count > 0
    ])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Property categories
    hydrophobic = sum(sequence.count(aa) for aa in "AILMFWV")
    polar = sum(sequence.count(aa) for aa in "STNQ")
    charged_pos = sum(sequence.count(aa) for aa in "RHK")
    charged_neg = sum(sequence.count(aa) for aa in "DE")
    aromatic = sum(sequence.count(aa) for aa in "FWY")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Hydrophobic", f"{hydrophobic/total*100:.1f}%" if total else "0%")
    col2.metric("Polar", f"{polar/total*100:.1f}%" if total else "0%")
    col3.metric("Positive", f"{charged_pos/total*100:.1f}%" if total else "0%")
    col4.metric("Negative", f"{charged_neg/total*100:.1f}%" if total else "0%")
    col5.metric("Aromatic", f"{aromatic/total*100:.1f}%" if total else "0%")

    # Estimated properties
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis as BioProtAnalysis
        analysis = BioProtAnalysis(sequence)
        st.markdown("#### Physicochemical Properties")
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Molecular Weight: {analysis.molecular_weight():.1f} Da")
            st.text(f"Isoelectric Point: {analysis.isoelectric_point():.2f}")
            st.text(f"Instability Index: {analysis.instability_index():.2f}")
        with col2:
            st.text(f"Aromaticity: {analysis.aromaticity():.4f}")
            gravy = analysis.gravy()
            st.text(f"GRAVY: {gravy:.4f}")
            st.text(f"{'Hydrophobic' if gravy > 0 else 'Hydrophilic'} overall")
    except ImportError:
        st.info("Install BioPython for full sequence analysis.")


def _visualization_tab(db: DatabaseManager):
    """3D protein visualization using py3Dmol."""
    st.subheader("3D Protein Visualization")

    pdb_file = st.session_state.get("current_pdb_file")
    pdb_id = st.session_state.get("current_pdb_id")

    if not pdb_file and not pdb_id:
        st.info("Load a protein structure first in the 'Load Protein' tab.")
        # Allow direct PDB ID input
        pdb_id = st.text_input("Or enter PDB ID for quick view", placeholder="e.g., 1AKE")
        if not pdb_id:
            return

    try:
        import py3Dmol
        from stmol import showmol

        style = st.selectbox("Visualization Style", [
            "cartoon", "stick", "sphere", "line", "cross"
        ])
        color_scheme = st.selectbox("Color Scheme", [
            "spectrum", "chain", "ssType", "residue"
        ])

        viewer = py3Dmol.view(width=700, height=500)

        if pdb_id:
            viewer.addModel(f"https://files.rcsb.org/download/{pdb_id}.pdb", "pdb")
        elif pdb_file and os.path.exists(pdb_file):
            with open(pdb_file) as f:
                viewer.addModel(f.read(), "pdb")

        if color_scheme == "spectrum":
            viewer.setStyle({style: {"color": "spectrum"}})
        elif color_scheme == "chain":
            viewer.setStyle({style: {"colorscheme": "chain"}})
        elif color_scheme == "ssType":
            viewer.setStyle({style: {"colorscheme": "ssType"}})
        else:
            viewer.setStyle({style: {"colorscheme": "amino"}})

        viewer.setBackgroundColor("white")
        viewer.zoomTo()
        showmol(viewer, height=500, width=700)

    except ImportError:
        st.error("py3Dmol and stmol are required. Run: pip install py3Dmol stmol")
    except Exception as e:
        st.error(f"Visualization error: {e}")


def _binding_site_tab(db: DatabaseManager):
    """Basic binding site analysis."""
    st.subheader("Binding Site Analysis")

    pdb_file = st.session_state.get("current_pdb_file")
    if not pdb_file:
        st.info("Load a protein structure first.")
        return

    try:
        from Bio.PDB import PDBParser, NeighborSearch

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        model = structure[0]

        # Find ligands (HETATM records that aren't water)
        ligands = []
        for residue in model.get_residues():
            het_flag = residue.get_id()[0]
            if het_flag not in (" ", "W"):
                resname = residue.get_resname()
                if resname not in ("HOH", "WAT", "DOD"):
                    ligands.append(residue)

        if not ligands:
            st.info("No ligands found in this structure.")
            st.markdown("Try analyzing the protein surface or uploading a complex.")
            return

        st.markdown(f"**Found {len(ligands)} ligand(s):**")

        for lig in ligands:
            lig_name = lig.get_resname()
            lig_chain = lig.get_parent().id
            lig_id = lig.get_id()[1]

            with st.expander(f"{lig_name} (Chain {lig_chain}, Res {lig_id})"):
                # Find nearby residues
                lig_atoms = list(lig.get_atoms())
                all_atoms = list(model.get_atoms())
                ns = NeighborSearch(all_atoms)

                contact_distance = st.slider(
                    f"Contact distance ({lig_name})", 3.0, 8.0, 4.5,
                    key=f"dist_{lig_name}_{lig_id}",
                )

                nearby_residues = set()
                for atom in lig_atoms:
                    neighbors = ns.search(atom.get_vector().get_array(), contact_distance)
                    for n in neighbors:
                        res = n.get_parent()
                        if res.get_id()[0] == " ":
                            nearby_residues.add(
                                (res.get_resname(), res.get_parent().id, res.get_id()[1])
                            )

                if nearby_residues:
                    res_df = pd.DataFrame(
                        sorted(nearby_residues, key=lambda x: x[2]),
                        columns=["Residue", "Chain", "Number"],
                    )
                    st.markdown(f"**{len(nearby_residues)} residues within {contact_distance} A:**")
                    st.dataframe(res_df, hide_index=True, use_container_width=True)

    except ImportError:
        st.error("BioPython is required for binding site analysis.")
    except Exception as e:
        st.error(f"Analysis error: {e}")
