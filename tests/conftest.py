"""Shared pytest fixtures for the Molecular Design Suite test suite."""

import pandas as pd
import pytest

from database.db_manager import DatabaseManager


@pytest.fixture
def tmp_db(tmp_path):
    """Fresh DatabaseManager backed by an isolated temp SQLite file."""
    return DatabaseManager(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def sample_dataset_df():
    """13-row dataset with valid SMILES and activity correlated to MW
    so a RandomForestRegressor can actually learn a signal.
    """
    smiles = [
        "CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
        "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "CCCCc1ccccc1",
        "CC(=O)O", "CCC(=O)O", "CCCC(=O)O",
    ]
    activity = [3.0 + 0.5 * i for i in range(len(smiles))]
    return pd.DataFrame({"SMILES": smiles, "Activity": activity})


@pytest.fixture
def pic50_dataset_df():
    """Same SMILES as sample_dataset_df with positive IC50 values in molar units,
    used to exercise the pIC50 transform path.
    """
    smiles = [
        "CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
        "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "CCCCc1ccccc1",
        "CC(=O)O", "CCC(=O)O", "CCCC(=O)O",
    ]
    # IC50 in molar units, ranging 1e-9 to 1e-5
    ic50 = [10.0 ** (-9 + 0.3 * i) for i in range(len(smiles))]
    return pd.DataFrame({"SMILES": smiles, "Activity": ic50})
