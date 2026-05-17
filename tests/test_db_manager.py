"""Tests for database/db_manager.py."""

import pytest


def test_add_molecule_duplicate_returns_existing_id(tmp_db):
    """The INSERT OR IGNORE contract: adding the same SMILES twice returns
    the original id, never raises and never creates a duplicate row.
    """
    id1 = tmp_db.add_molecule(
        smiles="CCO", canonical_smiles="CCO", name="ethanol",
        formula="C2H6O", mw=46.07,
    )
    id2 = tmp_db.add_molecule(
        smiles="CCO", canonical_smiles="CCO", name="ethanol-again",
        formula="C2H6O", mw=46.07,
    )
    assert id1 == id2
    assert tmp_db.get_molecule_count() == 1


def test_get_molecule_by_smiles_returns_dict(tmp_db):
    mol_id = tmp_db.add_molecule(
        smiles="CCO", canonical_smiles="CCO", name="ethanol",
    )
    row = tmp_db.get_molecule_by_smiles("CCO")
    assert row is not None
    assert row["id"] == mol_id
    assert row["name"] == "ethanol"


def test_get_molecule_by_smiles_missing_returns_none(tmp_db):
    assert tmp_db.get_molecule_by_smiles("NEVER_INSERTED") is None
