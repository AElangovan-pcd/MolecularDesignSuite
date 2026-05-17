"""QSAR model training, persistence, and prediction.

Single owner of joblib and sklearn-persistence imports in this codebase.
Consumed by modules/sar_analysis.py and modules/drug_optimization.py.
"""

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

import rdkit
from rdkit import Chem

from utils.rdkit_utils import mol_from_smiles, calculate_descriptor_set, canonical_smiles


VALID_TRANSFORMS = ("none", "log10", "pIC50")


@dataclass
class ModelArtifact:
    """In-memory bundle of everything needed to predict from a trained QSAR model."""
    model: Any
    scaler: Any
    feature_columns: list[str]
    training_smiles_hashes: frozenset[str]
    training_y_min: float
    training_y_max: float
    rdkit_version: str
    sklearn_version: str


def _hash_smiles(smiles: str) -> str:
    """Canonical-SMILES md5; used as a stable training-set membership marker."""
    canon = canonical_smiles(smiles) or smiles
    return hashlib.md5(canon.encode("utf-8")).hexdigest()


def _apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
    """Apply the activity transform to a numeric array."""
    if transform == "none":
        return values
    if transform == "log10":
        if (values <= 0).any():
            raise ValueError("log10 transform requires positive activity values")
        return np.log10(values)
    if transform == "pIC50":
        if (values <= 0).any():
            raise ValueError("pIC50 transform requires positive IC50 values (molar units)")
        return -np.log10(values)
    raise ValueError(f"Unknown activity_transform: {transform!r} (must be one of {VALID_TRANSFORMS})")


def train_qsar(
    df: pd.DataFrame,
    smiles_col: str,
    activity_col: str,
    activity_transform: str = "none",
) -> tuple[ModelArtifact, dict]:
    """Train a RandomForestRegressor on RDKit descriptors of SMILES + activity.

    Returns (artifact, metrics) where metrics is {cv_r2_mean, cv_r2_std, n_molecules,
    cv_y_actual, cv_y_predicted}.

    Raises ValueError if no valid molecules survive parsing, or if the transform
    encounters non-positive values.
    """
    if activity_transform not in VALID_TRANSFORMS:
        raise ValueError(f"Unknown activity_transform: {activity_transform!r}")

    # Parse SMILES, compute descriptors, drop invalid rows
    descriptor_rows = []
    activities = []
    smiles_kept = []
    for _, row in df.iterrows():
        smi = str(row[smiles_col])
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        desc = calculate_descriptor_set(mol)
        descriptor_rows.append(desc)
        activities.append(float(row[activity_col]))
        smiles_kept.append(smi)

    if not descriptor_rows:
        raise ValueError("No valid molecules in training set")

    desc_df = pd.DataFrame(descriptor_rows)
    feature_columns = list(desc_df.columns)
    X = desc_df[feature_columns].values
    y_raw = np.asarray(activities, dtype=float)
    y = _apply_transform(y_raw, activity_transform)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # 5-fold CV (or fewer folds if dataset is tiny)
    cv = min(5, len(y))
    if cv < 2:
        cv_mean = float("nan")
        cv_std = float("nan")
        cv_predictions = np.full_like(y, np.nan, dtype=float)
    else:
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")
        cv_mean = float(scores.mean())
        cv_std = float(scores.std())
        cv_predictions = cross_val_predict(model, X_scaled, y, cv=cv)

    model.fit(X_scaled, y)

    artifact = ModelArtifact(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        training_smiles_hashes=frozenset(_hash_smiles(s) for s in smiles_kept),
        training_y_min=float(y.min()),
        training_y_max=float(y.max()),
        rdkit_version=rdkit.__version__,
        sklearn_version=sklearn.__version__,
    )
    metrics = {
        "cv_r2_mean": cv_mean,
        "cv_r2_std": cv_std,
        "n_molecules": len(smiles_kept),
        "cv_y_actual": y,
        "cv_y_predicted": cv_predictions,
    }
    return artifact, metrics
