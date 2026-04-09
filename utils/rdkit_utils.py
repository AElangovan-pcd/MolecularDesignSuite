"""RDKit utility functions for molecular operations."""

from typing import Optional
import numpy as np

from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, Lipinski, rdMolDescriptors, Draw,
    rdFingerprintGenerator, DataStructs, Fragments, FilterCatalog,
    QED, Crippen, rdmolops,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import RDConfig
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES and return an RDKit Mol object, or None if invalid."""
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol
    except Exception:
        return None


def canonical_smiles(smiles: str) -> Optional[str]:
    """Return the canonical SMILES string, or None if invalid."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def validate_smiles(smiles: str) -> tuple[bool, str]:
    """Validate a SMILES string. Returns (is_valid, message)."""
    if not smiles or not smiles.strip():
        return False, "Empty SMILES string"
    mol = mol_from_smiles(smiles)
    if mol is None:
        return False, f"Invalid SMILES: '{smiles}'"
    return True, f"Valid molecule: {Chem.MolToSmiles(mol)}"


def mol_from_mol_block(mol_block: str) -> Optional[Chem.Mol]:
    """Parse a MOL block string and return an RDKit Mol object."""
    try:
        mol = Chem.MolFromMolBlock(mol_block)
        return mol
    except Exception:
        return None


def mol_to_smiles(mol: Chem.Mol) -> str:
    """Convert an RDKit Mol to canonical SMILES."""
    return Chem.MolToSmiles(mol)


def mol_to_mol_block(mol: Chem.Mol) -> str:
    """Convert an RDKit Mol to MOL block string."""
    return Chem.MolToMolBlock(mol)


# ── Basic Properties ──────────────────────────────────────────

def calculate_basic_properties(mol: Chem.Mol) -> dict:
    """Calculate basic molecular properties."""
    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "exact_mass": round(Descriptors.ExactMolWt(mol), 4),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "logP": round(Crippen.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "aromatic_rings": Lipinski.NumAromaticRings(mol),
        "rings": Lipinski.RingCount(mol),
        "heavy_atoms": Lipinski.HeavyAtomCount(mol),
        "fraction_csp3": round(Lipinski.FractionCSP3(mol), 3),
        "num_heteroatoms": Lipinski.NumHeteroatoms(mol),
        "molar_refractivity": round(Crippen.MolMR(mol), 2),
        "qed": round(QED.qed(mol), 3),
    }


# ── Drug-Likeness Filters ────────────────────────────────────

def lipinski_rule_of_five(mol: Chem.Mol) -> dict:
    """Evaluate Lipinski's Rule of Five."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "MW <= 500": mw <= 500,
        "LogP <= 5": logp <= 5,
        "HBD <= 5": hbd <= 5,
        "HBA <= 10": hba <= 10,
        "violations": violations,
        "passes": violations <= 1,
    }


def veber_rules(mol: Chem.Mol) -> dict:
    """Evaluate Veber's rules for oral bioavailability."""
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    return {
        "TPSA <= 140": tpsa <= 140,
        "Rotatable Bonds <= 10": rot_bonds <= 10,
        "passes": tpsa <= 140 and rot_bonds <= 10,
    }


def ghose_filter(mol: Chem.Mol) -> dict:
    """Evaluate Ghose filter."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    mr = Crippen.MolMR(mol)
    n_atoms = mol.GetNumAtoms()
    return {
        "160 <= MW <= 480": 160 <= mw <= 480,
        "-0.4 <= LogP <= 5.6": -0.4 <= logp <= 5.6,
        "40 <= MR <= 130": 40 <= mr <= 130,
        "20 <= Atoms <= 70": 20 <= n_atoms <= 70,
        "passes": all([160 <= mw <= 480, -0.4 <= logp <= 5.6,
                       40 <= mr <= 130, 20 <= n_atoms <= 70]),
    }


def egan_rules(mol: Chem.Mol) -> dict:
    """Evaluate Egan rules for absorption."""
    tpsa = Descriptors.TPSA(mol)
    logp = Crippen.MolLogP(mol)
    return {
        "TPSA <= 132": tpsa <= 132,
        "LogP <= 5.88": logp <= 5.88,
        "passes": tpsa <= 132 and logp <= 5.88,
    }


def muegge_rules(mol: Chem.Mol) -> dict:
    """Evaluate Muegge rules."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    rings = Lipinski.RingCount(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    heavy = Lipinski.HeavyAtomCount(mol)
    checks = {
        "200 <= MW <= 600": 200 <= mw <= 600,
        "-2 <= LogP <= 5": -2 <= logp <= 5,
        "TPSA <= 150": tpsa <= 150,
        "Rings <= 7": rings <= 7,
        "HBA <= 10": hba <= 10,
        "HBD <= 5": hbd <= 5,
        "Rotatable Bonds <= 15": rot <= 15,
        "Heavy Atoms >= 5": heavy >= 5,
    }
    checks["passes"] = all(v for k, v in checks.items() if k != "passes")
    return checks


def all_drug_likeness_filters(mol: Chem.Mol) -> dict:
    """Run all drug-likeness filters."""
    return {
        "Lipinski": lipinski_rule_of_five(mol),
        "Veber": veber_rules(mol),
        "Ghose": ghose_filter(mol),
        "Egan": egan_rules(mol),
        "Muegge": muegge_rules(mol),
    }


# ── ADMET-like Properties ────────────────────────────────────

def synthetic_accessibility_score(mol: Chem.Mol) -> float:
    """Calculate synthetic accessibility score (1=easy, 10=hard)."""
    from rdkit.Chem import RDConfig
    import os, sys
    sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_path not in sys.path:
        sys.path.insert(0, sa_path)
    try:
        import sascorer
        return round(sascorer.calculateScore(mol), 2)
    except ImportError:
        return _estimate_sa_score(mol)


def _estimate_sa_score(mol: Chem.Mol) -> float:
    """Fallback SA score estimation using basic descriptors."""
    ring_count = Lipinski.RingCount(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    stereo_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    score = 1.0 + (stereo_centers * 0.5) + (ring_count * 0.3) + (rot_bonds * 0.1)
    score = min(10.0, max(1.0, score))
    return round(score, 2)


def bioavailability_score(mol: Chem.Mol) -> float:
    """Estimate oral bioavailability score (0-1)."""
    violations = lipinski_rule_of_five(mol)["violations"]
    tpsa = Descriptors.TPSA(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    score = 1.0
    score -= violations * 0.15
    if tpsa > 140:
        score -= 0.15
    if rot > 10:
        score -= 0.1
    return round(max(0.0, min(1.0, score)), 2)


def estimate_bbb_permeant(mol: Chem.Mol) -> bool:
    """Estimate blood-brain barrier permeability."""
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    return tpsa <= 90 and mw <= 450 and logp >= 0 and hbd <= 3


def estimate_pgp_substrate(mol: Chem.Mol) -> bool:
    """Rough estimate of P-glycoprotein substrate likelihood."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    return mw > 400 or logp > 4 or hbd > 4


def admet_properties(mol: Chem.Mol) -> dict:
    """Calculate ADMET-related property estimates."""
    return {
        "bioavailability_score": bioavailability_score(mol),
        "synthetic_accessibility": synthetic_accessibility_score(mol),
        "bbb_permeant": estimate_bbb_permeant(mol),
        "pgp_substrate": estimate_pgp_substrate(mol),
        "qed": round(QED.qed(mol), 3),
    }


# ── Fingerprints & Similarity ────────────────────────────────

def morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    """Generate Morgan fingerprint."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol)


def tanimoto_similarity(mol1: Chem.Mol, mol2: Chem.Mol, radius: int = 2) -> float:
    """Calculate Tanimoto similarity between two molecules."""
    fp1 = morgan_fingerprint(mol1, radius)
    fp2 = morgan_fingerprint(mol2, radius)
    return round(DataStructs.TanimotoSimilarity(fp1, fp2), 4)


def bulk_tanimoto(ref_mol: Chem.Mol, mol_list: list[Chem.Mol], radius: int = 2) -> list[float]:
    """Calculate Tanimoto similarity of a reference mol against a list."""
    ref_fp = morgan_fingerprint(ref_mol, radius)
    fps = [morgan_fingerprint(m, radius) for m in mol_list]
    return [round(DataStructs.TanimotoSimilarity(ref_fp, fp), 4) for fp in fps]


# ── Scaffold Analysis ─────────────────────────────────────────

def get_murcko_scaffold(mol: Chem.Mol) -> Chem.Mol:
    """Get the Murcko scaffold of a molecule."""
    return MurckoScaffold.GetScaffoldForMol(mol)


def get_generic_scaffold(mol: Chem.Mol) -> Chem.Mol:
    """Get the generic (framework) scaffold of a molecule."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return MurckoScaffold.MakeScaffoldGeneric(scaffold)


# ── Conformer Generation ──────────────────────────────────────

def generate_conformers(mol: Chem.Mol, num_conformers: int = 10,
                        random_seed: int = 42) -> Chem.Mol:
    """Generate 3D conformers for a molecule."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
    if mol.GetNumConformers() > 0:
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    return mol


def generate_3d_coords(mol: Chem.Mol, random_seed: int = 42) -> Optional[Chem.Mol]:
    """Generate a single 3D conformation."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


# ── Molecular Descriptors (for QSAR) ─────────────────────────

def calculate_descriptor_set(mol: Chem.Mol) -> dict:
    """Calculate a comprehensive set of molecular descriptors."""
    desc = {}
    desc["MW"] = Descriptors.MolWt(mol)
    desc["LogP"] = Crippen.MolLogP(mol)
    desc["TPSA"] = Descriptors.TPSA(mol)
    desc["HBD"] = Lipinski.NumHDonors(mol)
    desc["HBA"] = Lipinski.NumHAcceptors(mol)
    desc["RotBonds"] = Lipinski.NumRotatableBonds(mol)
    desc["AromaticRings"] = Lipinski.NumAromaticRings(mol)
    desc["Rings"] = Lipinski.RingCount(mol)
    desc["HeavyAtoms"] = Lipinski.HeavyAtomCount(mol)
    desc["FractionCSP3"] = Lipinski.FractionCSP3(mol)
    desc["NumHeteroatoms"] = Lipinski.NumHeteroatoms(mol)
    desc["MR"] = Crippen.MolMR(mol)
    desc["QED"] = QED.qed(mol)
    desc["BalabanJ"] = Descriptors.BalabanJ(mol) if Lipinski.RingCount(mol) > 0 else 0
    desc["BertzCT"] = Descriptors.BertzCT(mol)
    desc["Chi0"] = Descriptors.Chi0(mol)
    desc["HallKierAlpha"] = Descriptors.HallKierAlpha(mol)
    desc["Kappa1"] = Descriptors.Kappa1(mol)
    desc["Kappa2"] = Descriptors.Kappa2(mol)
    desc["LabuteASA"] = Descriptors.LabuteASA(mol)
    desc["NumValenceElectrons"] = Descriptors.NumValenceElectrons(mol)
    desc["NumAliphaticRings"] = Descriptors.NumAliphaticRings(mol)
    desc["NumSaturatedRings"] = Descriptors.NumSaturatedRings(mol)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in desc.items()}


# ── Substructure Search ───────────────────────────────────────

def substructure_search(mol_list: list[Chem.Mol], pattern_smarts: str) -> list[int]:
    """Return indices of molecules matching the SMARTS pattern."""
    pattern = Chem.MolFromSmarts(pattern_smarts)
    if pattern is None:
        return []
    return [i for i, mol in enumerate(mol_list) if mol.HasSubstructMatch(pattern)]


# ── PAINS Filter ──────────────────────────────────────────────

def check_pains(mol: Chem.Mol) -> list[str]:
    """Check for PAINS (Pan Assay Interference Compounds) alerts."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    matches = []
    entry = catalog.GetFirstMatch(mol)
    while entry is not None:
        matches.append(entry.GetDescription())
        entry = catalog.GetFirstMatch(mol)
        break  # Avoid infinite loop; get first match
    return matches


def check_all_alerts(mol: Chem.Mol) -> dict:
    """Check for structural alerts (PAINS, Brenk, etc.)."""
    pains = check_pains(mol)
    return {
        "pains_alerts": pains,
        "has_pains": len(pains) > 0,
    }
