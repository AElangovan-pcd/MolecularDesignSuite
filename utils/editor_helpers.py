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
