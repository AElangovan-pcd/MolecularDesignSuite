"""Protein Structure Analysis Module for the Streamlit app."""

import streamlit as st
import pandas as pd
import io
import os
import json
import requests
import gzip
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


def _search_rcsb(query: str, max_results: int = 20) -> list[dict]:
    """Search RCSB PDB by keyword. Returns list of dicts with entry metadata."""
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query},
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    resp = requests.post(search_url, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    pdb_ids = [hit["identifier"] for hit in data.get("result_set", [])]
    if not pdb_ids:
        return []

    # Fetch metadata for each entry via GraphQL
    graphql_url = "https://data.rcsb.org/graphql"
    ids_str = ", ".join(f'"{pid}"' for pid in pdb_ids)
    gql_query = f"""{{
        entries(entry_ids: [{ids_str}]) {{
            rcsb_id
            struct {{ title }}
            rcsb_entry_info {{
                resolution_combined
                experimental_method
            }}
            polymer_entities {{
                rcsb_polymer_entity {{
                    pdbx_description
                }}
                entity_src_gen {{
                    pdbx_gene_src_scientific_name
                }}
            }}
        }}
    }}"""
    meta_resp = requests.post(graphql_url, json={"query": gql_query}, timeout=15)
    meta_resp.raise_for_status()
    entries = meta_resp.json().get("data", {}).get("entries", [])

    results = []
    for entry in entries:
        if entry is None:
            continue
        pdb_id = entry.get("rcsb_id", "")
        title = (entry.get("struct") or {}).get("title", "")
        info = entry.get("rcsb_entry_info") or {}
        resolution = info.get("resolution_combined")
        resolution_str = f"{resolution[0]:.2f}" if resolution else "N/A"
        method = info.get("experimental_method", "N/A")

        organism = "N/A"
        polymers = entry.get("polymer_entities") or []
        for poly in polymers:
            src_list = poly.get("entity_src_gen") or []
            for src in src_list:
                name = src.get("pdbx_gene_src_scientific_name")
                if name:
                    organism = name
                    break
            if organism != "N/A":
                break

        results.append({
            "PDB ID": pdb_id,
            "Title": title[:80],
            "Organism": organism,
            "Resolution (\u00c5)": resolution_str,
            "Method": method,
        })
    return results


def _download_rcsb_file(pdb_id: str, file_format: str, data_dir: str) -> str:
    """Download a structure file from RCSB. Returns the local file path."""
    os.makedirs(data_dir, exist_ok=True)
    pdb_id_lower = pdb_id.lower()

    if file_format == "PDB":
        url = f"https://files.rcsb.org/download/{pdb_id_lower}.pdb"
        local_path = os.path.join(data_dir, f"{pdb_id_lower}.pdb")
    elif file_format == "mmCIF":
        url = f"https://files.rcsb.org/download/{pdb_id_lower}.cif"
        local_path = os.path.join(data_dir, f"{pdb_id_lower}.cif")
    elif file_format == "FASTA":
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
        local_path = os.path.join(data_dir, f"{pdb_id_lower}.fasta")
    elif file_format == "Biological Assembly (PDB)":
        url = f"https://files.rcsb.org/download/{pdb_id_lower}.pdb1.gz"
        local_path = os.path.join(data_dir, f"{pdb_id_lower}_assembly1.pdb")
    else:
        raise ValueError(f"Unknown format: {file_format}")

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    if url.endswith(".gz"):
        content = gzip.decompress(resp.content).decode("utf-8")
    else:
        content = resp.text

    with open(local_path, "w") as f:
        f.write(content)
    return local_path


def _parse_and_save_structure(db: DatabaseManager, pdb_id: str, filepath: str):
    """Parse a PDB/CIF file, display info, and save to database."""
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Polypeptide import PPBuilder

    if filepath.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure(pdb_id, filepath)
    model = structure[0]
    chains = list(model.get_chains())
    residue_count = sum(1 for r in model.get_residues() if r.get_id()[0] == " ")
    atom_count = sum(1 for _ in model.get_atoms())

    st.success(f"Loaded {pdb_id}")
    st.text(f"Chains: {len(chains)} ({', '.join(c.id for c in chains)})")
    st.text(f"Residues: {residue_count}")
    st.text(f"Atoms: {atom_count}")

    # Extract sequence
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
        structure_file_path=filepath,
        project_id=st.session_state.get("current_project_id"),
    )
    st.session_state["current_protein_id"] = protein_id
    st.session_state["current_pdb_file"] = filepath
    st.session_state["current_pdb_id"] = pdb_id


def _load_protein_tab(db: DatabaseManager):
    """Load protein structure from RCSB search, PDB ID, or file upload."""
    st.subheader("Load Protein Structure")

    load_method = st.radio(
        "Source", ["Search RCSB", "PDB ID", "Upload PDB File"], horizontal=True
    )

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "protein_structures")

    if load_method == "Search RCSB":
        query = st.text_input(
            "Search RCSB PDB",
            placeholder="e.g., insulin receptor, EGFR kinase, hemoglobin",
            help="Search by protein name, function, organism, or any keyword",
        )
        max_results = st.slider("Max results", 5, 50, 20, key="rcsb_max")

        if query and st.button("Search", key="rcsb_search_btn"):
            with st.spinner(f"Searching RCSB for '{query}'..."):
                try:
                    results = _search_rcsb(query, max_results)
                    if results:
                        st.session_state["rcsb_search_results"] = results
                    else:
                        st.warning("No results found. Try a different search term.")
                        st.session_state.pop("rcsb_search_results", None)
                except Exception as e:
                    st.error(f"Search failed: {e}")

        # Display search results
        results = st.session_state.get("rcsb_search_results")
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

            pdb_options = [r["PDB ID"] for r in results]
            selected_pdb = st.selectbox("Select entry to download", pdb_options, key="rcsb_select")

            file_format = st.selectbox(
                "Download format",
                ["PDB", "mmCIF", "FASTA", "Biological Assembly (PDB)"],
                key="rcsb_format",
            )

            if st.button("Download & Load", key="rcsb_download_btn"):
                with st.spinner(f"Downloading {selected_pdb} ({file_format})..."):
                    try:
                        filepath = _download_rcsb_file(selected_pdb, file_format, data_dir)

                        if file_format == "FASTA":
                            with open(filepath) as f:
                                content = f.read()
                            st.success(f"Downloaded FASTA for {selected_pdb}")
                            st.code(content, language=None)
                            st.download_button(
                                "Save FASTA file",
                                content,
                                f"{selected_pdb}.fasta",
                                key="fasta_dl",
                            )
                        else:
                            _parse_and_save_structure(db, selected_pdb, filepath)
                    except Exception as e:
                        st.error(f"Download failed: {e}")

    elif load_method == "PDB ID":
        pdb_id = st.text_input("PDB ID", placeholder="e.g., 1AKE").strip().upper()

        file_format = st.selectbox(
            "Download format",
            ["PDB", "mmCIF", "Biological Assembly (PDB)"],
            key="pdbid_format",
        )

        if pdb_id and st.button("Fetch from RCSB"):
            try:
                with st.spinner(f"Downloading {pdb_id} ({file_format})..."):
                    filepath = _download_rcsb_file(pdb_id, file_format, data_dir)
                    _parse_and_save_structure(db, pdb_id, filepath)
            except requests.HTTPError as e:
                st.error(f"PDB ID not found or download failed: {e}")
            except ImportError:
                st.error("BioPython is required. Run: pip install biopython")
            except Exception as e:
                st.error(f"Failed to load PDB: {e}")

    else:
        uploaded = st.file_uploader("Upload structure file", type=["pdb", "ent", "cif"])
        if uploaded is not None:
            try:
                content = uploaded.read().decode("utf-8")
                os.makedirs(data_dir, exist_ok=True)
                filepath = os.path.join(data_dir, uploaded.name)
                with open(filepath, "w") as f:
                    f.write(content)

                pdb_id = uploaded.name.split(".")[0].upper()
                _parse_and_save_structure(db, pdb_id, filepath)

            except ImportError:
                st.error("BioPython is required.")
            except Exception as e:
                st.error(f"Failed to parse structure file: {e}")

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

        pdb_data = None
        fmt = "pdb"

        if pdb_file and os.path.exists(pdb_file):
            with open(pdb_file) as f:
                pdb_data = f.read()
            if pdb_file.endswith(".cif"):
                fmt = "cif"
        elif pdb_id:
            resp = requests.get(f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb", timeout=15)
            if resp.ok:
                pdb_data = resp.text

        if not pdb_data:
            st.warning("Could not load structure data. Make sure a protein is loaded or enter a valid PDB ID.")
            return

        viewer.addModel(pdb_data, fmt)

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

    except ImportError as e:
        st.error(f"Missing dependency: {e}. Run: pip install py3Dmol stmol ipython_genutils")
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
                    key=f"dist_{lig_name}_{lig_chain}_{lig_id}",
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
