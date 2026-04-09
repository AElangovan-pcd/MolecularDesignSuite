"""Visualization utilities for molecules and data."""

import io
import base64
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdMolDescriptors


def mol_to_svg(mol: Chem.Mol, size: tuple[int, int] = (300, 200),
               highlight_atoms: Optional[list[int]] = None) -> str:
    """Render a molecule to SVG string."""
    if mol is None:
        return ""
    drawer = Draw.MolDraw2DSVG(size[0], size[1])
    if highlight_atoms:
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_to_png_bytes(mol: Chem.Mol, size: tuple[int, int] = (300, 200)) -> bytes:
    """Render a molecule to PNG bytes."""
    if mol is None:
        return b""
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mol_to_base64_png(mol: Chem.Mol, size: tuple[int, int] = (300, 200)) -> str:
    """Render a molecule to a base64-encoded PNG string."""
    png_bytes = mol_to_png_bytes(mol, size)
    return base64.b64encode(png_bytes).decode("utf-8")


def mol_grid_image(mols: list[Chem.Mol], legends: Optional[list[str]] = None,
                   mols_per_row: int = 4, sub_img_size: tuple[int, int] = (300, 200)) -> bytes:
    """Create a grid image of molecules as PNG bytes."""
    valid = [(m, l) for m, l in zip(mols, legends or [""] * len(mols)) if m is not None]
    if not valid:
        return b""
    ms, ls = zip(*valid)
    img = Draw.MolsToGridImage(list(ms), molsPerRow=mols_per_row,
                                subImgSize=sub_img_size, legends=list(ls))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Plotly Charts ─────────────────────────────────────────────

def property_radar_chart(properties: dict, title: str = "Molecular Properties") -> go.Figure:
    """Create a radar chart for molecular properties (normalized)."""
    # Define reference ranges for normalization
    ranges = {
        "MW": (0, 600), "LogP": (-3, 7), "TPSA": (0, 200),
        "HBD": (0, 8), "HBA": (0, 12), "RotBonds": (0, 15),
        "QED": (0, 1),
    }
    categories = []
    values = []
    for key, (lo, hi) in ranges.items():
        if key in properties:
            categories.append(key)
            norm = (properties[key] - lo) / (hi - lo) if hi != lo else 0
            values.append(max(0, min(1, norm)))

    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]] if values else [],
        theta=categories + [categories[0]] if categories else [],
        fill="toself",
        name=title,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
        showlegend=False,
        height=400,
    )
    return fig


def property_distribution_plot(df: pd.DataFrame, property_col: str,
                                title: str = "") -> go.Figure:
    """Create a histogram/distribution plot for a property."""
    fig = px.histogram(df, x=property_col, nbins=30,
                       title=title or f"Distribution of {property_col}",
                       marginal="box")
    fig.update_layout(height=400)
    return fig


def scatter_plot(df: pd.DataFrame, x: str, y: str,
                 color: Optional[str] = None, hover_data: Optional[list[str]] = None,
                 title: str = "") -> go.Figure:
    """Create an interactive scatter plot."""
    fig = px.scatter(df, x=x, y=y, color=color, hover_data=hover_data,
                     title=title or f"{y} vs {x}")
    fig.update_layout(height=500)
    return fig


def correlation_heatmap(df: pd.DataFrame, title: str = "Property Correlations") -> go.Figure:
    """Create a correlation heatmap for numerical columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", title=title,
                    zmin=-1, zmax=1)
    fig.update_layout(height=600, width=700)
    return fig


def drug_likeness_summary_chart(filter_results: dict) -> go.Figure:
    """Create a summary chart of drug-likeness filter results."""
    filters = []
    passes = []
    for name, result in filter_results.items():
        filters.append(name)
        passes.append(1 if result.get("passes", False) else 0)

    colors = ["#2ecc71" if p else "#e74c3c" for p in passes]
    labels = ["Pass" if p else "Fail" for p in passes]

    fig = go.Figure(data=go.Bar(
        x=filters, y=[1] * len(filters),
        marker_color=colors,
        text=labels,
        textposition="inside",
    ))
    fig.update_layout(
        title="Drug-Likeness Filter Results",
        yaxis=dict(visible=False),
        height=300,
    )
    return fig


def similarity_heatmap(similarity_matrix: np.ndarray,
                       labels: Optional[list[str]] = None,
                       title: str = "Molecular Similarity") -> go.Figure:
    """Create a heatmap from a similarity matrix."""
    fig = px.imshow(
        similarity_matrix,
        x=labels, y=labels,
        text_auto=".2f",
        color_continuous_scale="Viridis",
        title=title,
        zmin=0, zmax=1,
    )
    fig.update_layout(height=600, width=700)
    return fig


def multi_property_comparison(molecules_data: list[dict],
                               properties: list[str]) -> go.Figure:
    """Compare multiple molecules across multiple properties."""
    fig = make_subplots(rows=1, cols=len(properties),
                        subplot_titles=properties)
    names = [d.get("name", f"Mol {i+1}") for i, d in enumerate(molecules_data)]
    for j, prop in enumerate(properties):
        values = [d.get(prop, 0) for d in molecules_data]
        fig.add_trace(
            go.Bar(x=names, y=values, name=prop, showlegend=False),
            row=1, col=j + 1,
        )
    fig.update_layout(height=400, title="Multi-Property Comparison")
    return fig


def chemical_space_plot(coords_2d: np.ndarray,
                        labels: Optional[list[str]] = None,
                        colors: Optional[list] = None,
                        title: str = "Chemical Space") -> go.Figure:
    """Plot 2D chemical space (from t-SNE/UMAP)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        mode="markers",
        text=labels,
        marker=dict(
            size=8,
            color=colors if colors is not None else "steelblue",
            colorscale="Viridis" if colors is not None else None,
            showscale=colors is not None,
            opacity=0.7,
        ),
        hovertemplate="%{text}<extra></extra>" if labels else None,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        height=600,
    )
    return fig
