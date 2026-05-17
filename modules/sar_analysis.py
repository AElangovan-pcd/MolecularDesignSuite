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
from utils.qsar import (
    train_qsar, save_model_artifact, load_model_artifact, predict, VALID_TRANSFORMS,
)
from utils.editor_helpers import edit_in_editor_button


def render_sar_analysis(db: DatabaseManager):
    """Render the SAR Analysis module."""
    st.header("Structure-Activity Relationship Analysis")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Molecular Descriptors", "Similarity Analysis",
        "Scaffold Analysis", "Chemical Space",
        "QSAR Modeling", "QSAR Predict",
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

    with tab6:
        _predict_tab(db)


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
                edit_in_editor_button(smi, key=f"scaffold_edit_{smi[:10]}")
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
    """Simple QSAR model building with persistence."""
    st.subheader("QSAR Modeling")
    st.caption("Build a QSAR model from molecular descriptors and activity data, then save it for later prediction.")

    st.markdown("#### Input Data")
    st.markdown("Upload a CSV with columns: `SMILES`, `Activity`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="qsar_csv")
    if uploaded is None:
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
        smiles_col = st.selectbox("SMILES column", df.columns.tolist(), key="qsar_smiles_col")
    if activity_col is None:
        activity_col = st.selectbox(
            "Activity column",
            [c for c in df.columns if c != smiles_col],
            key="qsar_activity_col",
        )

    transform = st.radio(
        "Activity transform (applied to the activity column before training)",
        VALID_TRANSFORMS,
        horizontal=True,
        key="qsar_transform",
        help="'pIC50' computes -log10(IC50 in molar units). 'log10' computes log10(value). 'none' uses values as-is.",
    )

    if st.button("Build QSAR Model", key="qsar_build_btn"):
        try:
            artifact, metrics = train_qsar(df, smiles_col, activity_col, activity_transform=transform)
        except ValueError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Training failed: {e}")
            return
        st.session_state["qsar_last_artifact"] = artifact
        st.session_state["qsar_last_metrics"] = metrics
        st.session_state["qsar_last_dataset_name"] = uploaded.name
        st.session_state["qsar_last_transform"] = transform

    # Render metrics + save section whenever a trained artifact lives in
    # session_state. Gating on the Build button alone makes the Save click
    # a no-op (the rerun triggered by Save would early-return).
    if "qsar_last_artifact" not in st.session_state:
        return

    artifact = st.session_state["qsar_last_artifact"]
    metrics = st.session_state["qsar_last_metrics"]

    st.markdown("#### Model Performance (5-fold CV)")
    st.metric("Mean R\u00b2", f"{metrics['cv_r2_mean']:.3f} \u00b1 {metrics['cv_r2_std']:.3f}")
    st.caption(f"Trained on {metrics['n_molecules']} valid molecules")

    importances = pd.DataFrame({
        "Feature": artifact.feature_columns,
        "Importance": artifact.model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    st.markdown("#### Top Feature Importances")
    st.dataframe(importances.head(10), hide_index=True, use_container_width=True)

    # Predicted vs actual comes from cross_val_predict run inside train_qsar.
    pred_df = pd.DataFrame({
        "Actual": metrics["cv_y_actual"],
        "Predicted": metrics["cv_y_predicted"],
    })
    st.plotly_chart(
        scatter_plot(pred_df, "Actual", "Predicted", title="Predicted vs Actual Activity"),
        use_container_width=True,
    )

    # \u2500\u2500 Save Model \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    st.divider()
    st.markdown("#### Save This Model")

    default_name = uploaded.name.rsplit(".", 1)[0] + "-RF"
    save_col1, save_col2 = st.columns(2)
    with save_col1:
        save_name = st.text_input("Model name", value=default_name, key="qsar_save_name")
        save_activity_label = st.text_input(
            "Activity label (required)",
            placeholder="e.g., pIC50, log(EC50) [nM], % inhibition",
            key="qsar_save_label",
        )
    with save_col2:
        higher_is_better = st.checkbox(
            "Higher activity = better potency?",
            value=True,
            key="qsar_save_higher_better",
            help="Used by Drug Optimization MPO to know whether to maximize or minimize predictions.",
        )

    if st.button("Save Model", key="qsar_save_btn", type="primary"):
        if not save_activity_label.strip():
            st.error("Activity label is required.")
            return
        if "qsar_last_artifact" not in st.session_state:
            st.error("Train a model first.")
            return
        meta = {
            "name": save_name.strip() or default_name,
            "dataset_name": st.session_state["qsar_last_dataset_name"],
            "activity_label": save_activity_label.strip(),
            "activity_transform": st.session_state["qsar_last_transform"],
            "higher_is_better": 1 if higher_is_better else 0,
            "cv_r2_mean": st.session_state["qsar_last_metrics"]["cv_r2_mean"],
            "cv_r2_std": st.session_state["qsar_last_metrics"]["cv_r2_std"],
            "project_id": st.session_state.get("current_project_id"),
        }
        try:
            model_id = save_model_artifact(
                st.session_state["qsar_last_artifact"], meta, db,
            )
            st.success(f"Saved as model #{model_id}: {meta['name']}")
        except Exception as e:
            st.error(f"Failed to save model: {e}")


def _predict_tab(db: DatabaseManager):
    """Predict activity using a saved QSAR model."""
    st.subheader("QSAR Predict")
    st.caption("Pick a saved model and score new SMILES.")

    project_id = st.session_state.get("current_project_id")
    models = db.get_qsar_models(project_id=project_id)
    if not models:
        st.info(
            "No saved QSAR models for this project. "
            "Train one in the 'QSAR Modeling' tab and click 'Save Model'."
        )
        return

    def _label(m: dict) -> str:
        r2 = m.get("cv_r2_mean")
        r2_str = f"R²={r2:.2f}" if r2 is not None else "R²=n/a"
        return f"#{m['id']} {m['name']} ({r2_str}, {m['created_date']})"

    options = {_label(m): m for m in models}
    chosen_label = st.selectbox("Saved model", list(options.keys()), key="predict_model_select")
    chosen = options[chosen_label]

    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        st.metric("Activity label", chosen["activity_label"])
    with meta_col2:
        st.metric("Higher is better", "Yes" if chosen["higher_is_better"] else "No")
    with meta_col3:
        st.metric("Training n", chosen["n_molecules"])
    if chosen.get("activity_transform") and chosen["activity_transform"] != "none":
        st.caption(f"Predictions are in *{chosen['activity_transform']}-transformed* space.")

    source = st.radio("Input", ["Paste SMILES", "From Database"], horizontal=True, key="predict_source")
    smiles_list: list[str] = []
    if source == "Paste SMILES":
        text = st.text_area(
            "SMILES (one per line)",
            height=150,
            key="predict_text",
            placeholder="CCO\nc1ccccc1\nCC(=O)O",
        )
        if text:
            smiles_list = [line.strip() for line in text.strip().split("\n") if line.strip()]
    else:
        db_mols = db.get_molecules(project_id=project_id, limit=500)
        if not db_mols:
            st.info("No molecules in the database for this project.")
            return
        labels = {f"{m['name'] or m['canonical_smiles'][:25]} (#{m['id']})": m["smiles"] for m in db_mols}
        chosen_dbs = st.multiselect("Molecules", list(labels.keys()), key="predict_db_select")
        smiles_list = [labels[c] for c in chosen_dbs]

    if not smiles_list:
        return

    if not st.button("Predict", key="predict_btn", type="primary"):
        return

    try:
        artifact = load_model_artifact(chosen["id"], db)
    except FileNotFoundError:
        st.error(
            "Model artifact missing on disk. "
            "Delete this row from Data Management → QSAR Models, then re-train."
        )
        return

    # Version skew warning, non-blocking
    import rdkit, sklearn
    if chosen.get("rdkit_version") and chosen["rdkit_version"] != rdkit.__version__:
        st.warning(
            f"Model was trained with RDKit {chosen['rdkit_version']}, "
            f"current is {rdkit.__version__}. Predictions may differ slightly."
        )
    if chosen.get("sklearn_version") and chosen["sklearn_version"] != sklearn.__version__:
        st.warning(
            f"Model was trained with scikit-learn {chosen['sklearn_version']}, "
            f"current is {sklearn.__version__}. Predictions may differ slightly."
        )

    results = predict(artifact, smiles_list)
    pred_col_label = f"Predicted {chosen['activity_label']}"
    results_df = pd.DataFrame([
        {
            "SMILES": r["smiles"],
            pred_col_label: r["predicted_value"],
            "in_training?": "yes ⚠" if r["in_training"] else "no",
            "error": r["error"] or "",
        }
        for r in results
    ])
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    csv = results_df.to_csv(index=False)
    st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
