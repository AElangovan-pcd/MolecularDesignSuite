"""File handling utilities for molecular data import/export."""

import io
import csv
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools


def parse_sdf_file(file_content: bytes) -> list[dict]:
    """Parse an SDF file and return a list of molecule dicts."""
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(file_content))
    molecules = []
    for mol in supplier:
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(mol)
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        props = {}
        for prop_name in mol.GetPropsAsDict():
            if not prop_name.startswith("_"):
                props[prop_name] = mol.GetPropsAsDict()[prop_name]
        molecules.append({
            "smiles": smiles,
            "name": name,
            "mol": mol,
            "properties": props,
        })
    return molecules


def parse_mol_file(file_content: str) -> Optional[Chem.Mol]:
    """Parse a MOL file string and return an RDKit Mol object."""
    try:
        mol = Chem.MolFromMolBlock(file_content)
        return mol
    except Exception:
        return None


def parse_smiles_file(file_content: str, delimiter: str = "\t",
                      smiles_col: int = 0, name_col: int = 1) -> list[dict]:
    """Parse a SMILES file (tab/comma delimited)."""
    molecules = []
    reader = csv.reader(io.StringIO(file_content), delimiter=delimiter)
    for row in reader:
        if not row:
            continue
        smiles = row[smiles_col].strip() if len(row) > smiles_col else ""
        name = row[name_col].strip() if len(row) > name_col else ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecules.append({
                "smiles": Chem.MolToSmiles(mol),
                "name": name,
                "mol": mol,
            })
    return molecules


def parse_csv_with_smiles(file_content: str, smiles_column: str = "SMILES",
                          name_column: Optional[str] = None) -> pd.DataFrame:
    """Parse a CSV file containing a SMILES column."""
    df = pd.read_csv(io.StringIO(file_content))
    if smiles_column not in df.columns:
        # Try case-insensitive match
        for col in df.columns:
            if col.lower() == smiles_column.lower():
                smiles_column = col
                break
        else:
            raise ValueError(f"Column '{smiles_column}' not found. Available: {list(df.columns)}")
    df["ROMol"] = df[smiles_column].apply(lambda s: Chem.MolFromSmiles(str(s)))
    df["valid"] = df["ROMol"].apply(lambda m: m is not None)
    return df


def export_to_sdf(molecules: list[dict], include_properties: bool = True) -> str:
    """Export molecules to SDF format string."""
    output = io.StringIO()
    writer = Chem.SDWriter(output)
    for mol_data in molecules:
        mol = mol_data.get("mol")
        if mol is None:
            smiles = mol_data.get("smiles", "")
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
        if mol_data.get("name"):
            mol.SetProp("_Name", mol_data["name"])
        if include_properties:
            for key, val in mol_data.get("properties", {}).items():
                mol.SetProp(str(key), str(val))
        writer.write(mol)
    writer.close()
    return output.getvalue()


def export_to_csv(molecules_df: pd.DataFrame, include_mol: bool = False) -> str:
    """Export a DataFrame to CSV, optionally excluding RDKit mol objects."""
    df = molecules_df.copy()
    if not include_mol and "ROMol" in df.columns:
        df = df.drop(columns=["ROMol"])
    return df.to_csv(index=False)


def export_molecules_to_smiles(molecules: list[dict]) -> str:
    """Export molecules as a SMILES file (SMILES\\tName per line)."""
    lines = []
    for mol_data in molecules:
        smiles = mol_data.get("smiles", "")
        name = mol_data.get("name", "")
        lines.append(f"{smiles}\t{name}")
    return "\n".join(lines)
