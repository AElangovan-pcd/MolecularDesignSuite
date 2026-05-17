"""Backfill tests for the existing utils/rdkit_utils.py surface."""

import pytest
from rdkit import Chem

from utils.rdkit_utils import (
    mol_from_smiles, canonical_smiles, validate_smiles,
    calculate_basic_properties,
    lipinski_rule_of_five, veber_rules, ghose_filter,
    egan_rules, muegge_rules, all_drug_likeness_filters,
    tanimoto_similarity, check_pains,
)


ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
LONG_ALKANE = "C" * 60  # deliberate Lipinski violator (MW > 500)


def test_mol_from_smiles_valid():
    assert mol_from_smiles(ASPIRIN) is not None


def test_mol_from_smiles_invalid_returns_none():
    assert mol_from_smiles("not a smiles") is None
    assert mol_from_smiles("") is None
    assert mol_from_smiles("   ") is None


def test_canonical_smiles_idempotent():
    canon = canonical_smiles(ASPIRIN)
    assert canon is not None
    assert canonical_smiles(canon) == canon


def test_validate_smiles_returns_tuple():
    ok, msg = validate_smiles(ASPIRIN)
    assert ok is True
    assert isinstance(msg, str)

    bad, msg = validate_smiles("xyz123")
    assert bad is False
    assert isinstance(msg, str)


def test_calculate_basic_properties_aspirin():
    props = calculate_basic_properties(mol_from_smiles(ASPIRIN))
    expected_keys = {
        "molecular_weight", "exact_mass", "formula", "logP", "tpsa",
        "hbd", "hba", "rotatable_bonds", "aromatic_rings", "rings",
        "heavy_atoms", "fraction_csp3", "num_heteroatoms",
        "molar_refractivity", "qed",
    }
    assert expected_keys.issubset(props.keys())
    # Aspirin MW = 180.16 g/mol
    assert abs(props["molecular_weight"] - 180.16) < 1.0
    assert props["formula"] == "C9H8O4"


def test_lipinski_pass_and_fail():
    aspirin = lipinski_rule_of_five(mol_from_smiles(ASPIRIN))
    assert aspirin["passes"] is True
    assert aspirin["violations"] == 0

    fail = lipinski_rule_of_five(mol_from_smiles(LONG_ALKANE))
    assert fail["passes"] is False


def test_veber_rules_aspirin_passes():
    result = veber_rules(mol_from_smiles(ASPIRIN))
    assert result["passes"] is True


def test_ghose_filter_aspirin():
    # Aspirin MW is below Ghose minimum (160), so this should fail on MW
    result = ghose_filter(mol_from_smiles(ASPIRIN))
    assert result["160 <= MW <= 480"] is True  # 180 is in range
    assert "passes" in result


def test_egan_rules_aspirin_passes():
    result = egan_rules(mol_from_smiles(ASPIRIN))
    assert result["passes"] is True


def test_muegge_rules_aspirin():
    result = muegge_rules(mol_from_smiles(ASPIRIN))
    assert "passes" in result
    assert isinstance(result["passes"], bool)


def test_all_drug_likeness_filters_returns_all_five():
    result = all_drug_likeness_filters(mol_from_smiles(ASPIRIN))
    assert set(result.keys()) == {"Lipinski", "Veber", "Ghose", "Egan", "Muegge"}
    for name, checks in result.items():
        assert "passes" in checks


def test_tanimoto_self_similarity_is_one():
    mol = mol_from_smiles(ASPIRIN)
    assert tanimoto_similarity(mol, mol) == 1.0


def test_tanimoto_different_molecules_below_one():
    mol1 = mol_from_smiles(ASPIRIN)
    mol2 = mol_from_smiles("c1ccccc1")  # benzene
    sim = tanimoto_similarity(mol1, mol2)
    assert 0.0 <= sim < 1.0


def test_check_pains_aspirin_clean():
    # Aspirin is not a PAINS hit
    assert check_pains(mol_from_smiles(ASPIRIN)) == []


def test_check_pains_known_alert():
    # Quinone / catechol style structures often hit PAINS catalogs.
    # 1,2-dihydroxybenzene (catechol) hits the catechol_A PAINS pattern.
    catechol = mol_from_smiles("Oc1ccccc1O")
    matches = check_pains(catechol)
    assert len(matches) >= 1
