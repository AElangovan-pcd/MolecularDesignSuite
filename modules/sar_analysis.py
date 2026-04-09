"""Structure-Activity Relationship (SAR) Analysis Module."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

from rdkit import Chem
from rdkit.Chem import DataStructs

from utils.rdkit_utils import (
    mol_from_smiles, calculate_descriptor_set, morgan_fingerprint,
    tanimoto_similarity, bulk_tanimoto, get_murcko_scaffold,
    get_generic_scaffold, mol_to_smiles, substructure_search,
)
from utils.visualization import (
    mol_to_svg, scatter_plot, correlation_heatmap,
    similarity_heatmap, chemical_space_plot, mol_grid_image,
)
from database.db_manager import DatabaseManager


def render_sar_analysis(db: DatabaseManager):
    """Render the SAR Analysis module."""
    st.header("Structure-Activity Relationship Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Molecular Descriptors", "Similarity Analysis",
        "Scaffold Analysis", "Chemical Space", "QSAR Modeling"
    ])

    with tab1:
        _descriptors_tab(db)

    with tab2:
        _similarity_tab(db)

    with tab3:
        _scaffold_tab(db)

    with tab4:
        _chemical_space_tab(db)

    with tab5:
        _qsar_tab(db)


def _load_molecules(db: DatabaseManager, key_prefix: str) -> list[dict]:
    """Load molecules from database or SMILES input."""
    source = st.radio("Source", ["Database", "Paste SMILES"],
                      horizontal=True, key=f"{key_prefix}_source")
    molecules = []
    if source == "Database":
        db_mols = db.get_molecules(limit=500)
        for m in db_mols:
            mol = mol_from_smiles(m["smiles"])
            if mol:
                molecules.append({"name": m["name"] or m["smiles"][:25],
                                  "smiles": m["smiles"], "mol": mol})
    else:
        text = st.text_area("SMILES (one per line, optionally tab-separated with name)",
                            height=150, key=f"{key_prefix}_text")
        if text:
            for line in text.strip().split("\n"):
                parts = line.strip().split("\t")
                smi = parts[0].strip()
                name = parts[1].strip() if len(parts) > 1 else smi[:25]
                mol = mol_from_smiles(smi)
                if mol:
                    molecules.append({"name": name, "smiles": smi, "mol": mol})
    return molecules


def _descriptors_tab(db: DatabaseManager):
    """Calculate and display molecular descriptors."""
    st.subheader("Molecular Descriptors")

    molecules = _load_molecules(db, "desc")
    if not molecules:
        st.info("Load or enter molecules to calculate descriptors.")
        return

    st.info(f"Calculating descriptors for {len(molecules)} molecules...")

    rows = []
    for md in molecules:
        desc = calculate_descriptor_set(md["mol"])
        row = {"Name": md["name"], "SMILES": md["smiles"]}
        row.update(desc)
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Correlations
    if len(df) > 2:
        st.subheader("Descriptor Correlations")
        numeric_df = df.select_dtypes(include=[np.number])
        st.plotly_chart(correlation_heatmap(numeric_df), use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("Download Descriptors CSV", csv, "descriptors.csv", "text/csv")


def _similarity_tab(db: DatabaseManager):
    """Molecular similarity analysis."""
    st.subheader("Similarity Analysis")

    molecules = _load_molecules(db, "sim")
    if len(molecules) < 2:
        st.info("Need at least 2 molecules for similarity analysis.")
        return

    # Compute pairwise similarity matrix
    n = len(molecules)
    sim_matrix = np.zeros((n, n))
    mols = [m["mol"] for m in molecules]
    names = [m["name"] for m in molecules]

    fps = [morgan_fingerprint(m) for m in mols]
    for i in range(n):
        for j in range(i, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    st.plotly_chart(
        similarity_heatmap(sim_matrix, labels=names),
        use_container_width=True,
    )

    # Most/least similar pairs
    st.subheader("Notable Pairs")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((names[i], names[j], round(sim_matrix[i][j], 4)))
    pairs.sort(key=lambda x: -x[2])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Most Similar**")
        for a, b, s in pairs[:5]:
            st.text(f"  {a} - {b}: {s}")
    with col2:
        st.markdown("**Least Similar**")
        for a, b, s in pairs[-5:]:
            st.text(f"  {a} - {b}: {s}")


def _scaffold_tab(db: DatabaseManager):
    """Scaffold analysis using Murcko decomposition."""
    st.subheader("Scaffold Analysis")

    molecules = _load_molecules(db, "scaffold")
    if not molecules:
        st.info("Load molecules to analyze scaffolds.")
        return

    scaffold_type = st.radio("Scaffold type", ["Murcko", "Generic (Framework)"],
                             horizontal=True)

    scaffolds = {}
    for md in molecules:
        try:
            if scaffold_type == "Murcko":
                scaffold = get_murcko_scaffold(md["mol"])
            else:
                scaffold = get_generic_scaffold(md["mol"])
            scaffold_smi = mol_to_smiles(scaffold)
            if scaffold_smi not in scaffolds:
                scaffolds[scaffold_smi] = {"scaffold_mol": scaffold, "members": []}
            scaffolds[scaffold_smi]["members"].append(md["name"])
        except Exception:
            continue

    st.markdown(f"**{len(scaffolds)} unique scaffolds** from {len(molecules)} molecules")

    # Sort by frequency
    sorted_scaffolds = sorted(scaffolds.items(), key=lambda x: -len(x[1]["members"]))

    for smi, data in sorted_scaffolds[:20]:
        with st.expander(f"{smi} ({len(data['members'])} molecules)"):
            col1, col2 = st.columns([1, 2])
            with col1:
                svg = mol_to_svg(data["scaffold_mol"], size=(250, 180))
                st.image(svg, use_container_width=True)
            with col2:
                st.markdown("**Member molecules:**")
                for name in data["members"]:
                    st.text(f"  - {name}")


def _chemical_space_tab(db: DatabaseManager):
    """Chemical space visualization using t-SNE/PCA."""
    st.subheader("Chemical Space Visualization")

    molecules = _load_molecules(db, "chemspace")
    if len(molecules) < 5:
        st.info("Need at least 5 molecules for chemical space analysis.")
        return

    method = st.radio("Dimensionality reduction", ["PCA", "t-SNE"], horizontal=True)
    color_by = st.selectbox("Color by", ["MW", "LogP", "TPSA", "QED", "HBD", "HBA"])

    mols = [m["mol"] for m in molecules]
    names = [m["name"] for m in molecules]

    # Generate fingerprints as feature vectors
    fps = []
    for mol in mols:
        fp = morgan_fingerprint(mol, n_bits=1024)
        arr = np.zeros(1024)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    X = np.array(fps)

    # Calculate color property
    from utils.rdkit_utils import calculate_basic_properties
    props = [calculate_basic_properties(m) for m in mols]
    prop_map = {
        "MW": "molecular_weight", "LogP": "logP", "TPSA": "tpsa",
        "QED": "qed", "HBD": "hbd", "HBA": "hba",
    }
    colors = [p[prop_map[color_by]] for p in props]

    if method == "PCA":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(X)
        explained = reducer.explained_variance_ratio_
        st.caption(f"Explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")
    else:
        from sklearn.manifold import TSNE
        perplexity = min(30, len(molecules) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = reducer.fit_transform(X)

    st.plotly_chart(
        chemical_space_plot(coords, labels=names, colors=colors,
                            title=f"Chemical Space ({method}, colored by {color_by})"),
        use_container_width=True,
    )


def _qsar_tab(db: DatabaseManager):
    """Simple QSAR model building."""
    st.subheader("QSAR Modeling")
    st.caption("Build a simple QSAR model from molecular descriptors and activity data.")

    st.markdown("#### Input Data")
    st.markdown("Upload a CSV with columns: `SMILES`, `Activity`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="qsar_csv")
    if uploaded is None:
        # Show example format
        st.code("SMILES,Activity\nCCO,3.5\nc1ccccc1,5.2\nCC(=O)O,2.1", language="csv")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    smiles_col = None
    activity_col = None
    for col in df.columns:
        if col.lower() in ("smiles", "smi", "molecule"):
            smiles_col = col
        if col.lower() in ("activity", "pic50", "ic50", "ec50", "potency", "value"):
            activity_col = col

    if smiles_col is None:
        smiles_col = st.selectbox("SMILES column", df.columns.tolist())
    if activity_col is None:
        activity_col = st.selectbox("Activity column",
                                     [c for c in df.columns if c != smiles_col])

    if not st.button("Build QSAR Model"):
        return

    # Calculate descriptors
    valid_rows = []
    for _, row in df.iterrows():
        mol = mol_from_smiles(str(row[smiles_col]))
        if mol is not None:
            desc = calculate_descriptor_set(mol)
            desc["Activity"] = float(row[activity_col])
            valid_rows.append(desc)

    if len(valid_rows) < 10:
        st.warning(f"Only {len(valid_rows)} valid molecules. Need at least 10 for modeling.")
        return

    desc_df = pd.DataFrame(valid_rows)
    feature_cols = [c for c in desc_df.columns if c != "Activity"]
    X = desc_df[feature_cols].values
    y = desc_df["Activity"].values

    # Train/test split and Random Forest
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

    st.markdown("#### Model Performance (5-fold CV)")
    st.metric("Mean R\u00b2", f"{scores.mean():.3f} \u00b1 {scores.std():.3f}")

    # Feature importance
    model.fit(X_scaled, y)
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    st.markdown("#### Top Feature Importances")
    st.dataframe(importances.head(10), hide_index=True, use_container_width=True)

    # Predicted vs actual
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(model, X_scaled, y, cv=5)
    pred_df = pd.DataFrame({"Actual": y, "Predicted": y_pred})
    st.plotly_chart(
        scatter_plot(pred_df, "Actual", "Predicted",
                     title="Predicted vs Actual Activity"),
        use_container_width=True,
    )
