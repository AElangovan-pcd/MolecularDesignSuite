"""Database manager for the Molecular Design Suite."""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


DB_DIR = Path(__file__).parent
SCHEMA_PATH = DB_DIR / "schema.sql"
DEFAULT_DB_PATH = DB_DIR / "molecular_design.db"


class DatabaseManager:
    """Manages SQLite database connections and operations."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(DEFAULT_DB_PATH)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with open(SCHEMA_PATH, "r") as f:
            schema_sql = f.read()
        conn = self._get_connection()
        try:
            conn.executescript(schema_sql)
            conn.commit()
        finally:
            conn.close()

    # ── Projects ──────────────────────────────────────────────

    def create_project(self, name: str, description: str = "") -> int:
        conn = self._get_connection()
        try:
            cur = conn.execute(
                "INSERT INTO projects (name, description) VALUES (?, ?)",
                (name, description),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_projects(self) -> list[dict]:
        conn = self._get_connection()
        try:
            rows = conn.execute("SELECT * FROM projects ORDER BY updated_date DESC").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_project(self, project_id: int):
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()
        finally:
            conn.close()

    # ── Molecules ─────────────────────────────────────────────

    def add_molecule(self, smiles: str, canonical_smiles: str, name: str = "",
                     formula: str = "", mw: float = 0, logp: float = 0,
                     tpsa: float = 0, hbd: int = 0, hba: int = 0,
                     rotatable_bonds: int = 0, aromatic_rings: int = 0,
                     project_id: Optional[int] = None) -> int:
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """INSERT OR IGNORE INTO molecules
                   (smiles, canonical_smiles, name, formula, molecular_weight,
                    logp, tpsa, hbd, hba, rotatable_bonds, aromatic_rings, project_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (smiles, canonical_smiles, name, formula, mw, logp, tpsa,
                 hbd, hba, rotatable_bonds, aromatic_rings, project_id),
            )
            conn.commit()
            if cur.lastrowid == 0:
                row = conn.execute(
                    "SELECT id FROM molecules WHERE smiles = ?", (smiles,)
                ).fetchone()
                return row["id"]
            return cur.lastrowid
        finally:
            conn.close()

    def get_molecules(self, project_id: Optional[int] = None,
                      limit: int = 100, offset: int = 0) -> list[dict]:
        conn = self._get_connection()
        try:
            if project_id is not None:
                rows = conn.execute(
                    "SELECT * FROM molecules WHERE project_id = ? ORDER BY created_date DESC LIMIT ? OFFSET ?",
                    (project_id, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM molecules ORDER BY created_date DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_molecule_by_id(self, mol_id: int) -> Optional[dict]:
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT * FROM molecules WHERE id = ?", (mol_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_molecule_by_smiles(self, smiles: str) -> Optional[dict]:
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT * FROM molecules WHERE smiles = ?", (smiles,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def delete_molecule(self, mol_id: int):
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM molecules WHERE id = ?", (mol_id,))
            conn.commit()
        finally:
            conn.close()

    def get_molecule_count(self, project_id: Optional[int] = None) -> int:
        conn = self._get_connection()
        try:
            if project_id is not None:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM molecules WHERE project_id = ?",
                    (project_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as cnt FROM molecules").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    # ── Molecular Properties ──────────────────────────────────

    def add_property(self, molecule_id: int, property_name: str,
                     property_value: Optional[float] = None,
                     property_text: Optional[str] = None,
                     calculation_method: str = "RDKit"):
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO molecular_properties
                   (molecule_id, property_name, property_value, property_text, calculation_method)
                   VALUES (?, ?, ?, ?, ?)""",
                (molecule_id, property_name, property_value, property_text, calculation_method),
            )
            conn.commit()
        finally:
            conn.close()

    def get_properties(self, molecule_id: int) -> list[dict]:
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM molecular_properties WHERE molecule_id = ? ORDER BY property_name",
                (molecule_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ── Proteins ──────────────────────────────────────────────

    def add_protein(self, pdb_id: str = "", name: str = "", organism: str = "",
                    sequence: str = "", structure_file_path: str = "",
                    resolution: float = 0, project_id: Optional[int] = None) -> int:
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """INSERT INTO proteins
                   (pdb_id, name, organism, sequence, structure_file_path, resolution, project_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (pdb_id, name, organism, sequence, structure_file_path, resolution, project_id),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_proteins(self, project_id: Optional[int] = None) -> list[dict]:
        conn = self._get_connection()
        try:
            if project_id is not None:
                rows = conn.execute(
                    "SELECT * FROM proteins WHERE project_id = ? ORDER BY created_date DESC",
                    (project_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM proteins ORDER BY created_date DESC"
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ── Experiments ────────────────────────────────────────────

    def add_experiment(self, molecule_id: int, protein_id: Optional[int] = None,
                       assay_type: str = "", activity_value: float = 0,
                       activity_unit: str = "", activity_relation: str = "=",
                       notes: str = "", project_id: Optional[int] = None) -> int:
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """INSERT INTO experiments
                   (molecule_id, protein_id, assay_type, activity_value,
                    activity_unit, activity_relation, notes, project_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (molecule_id, protein_id, assay_type, activity_value,
                 activity_unit, activity_relation, notes, project_id),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_experiments(self, molecule_id: Optional[int] = None,
                        protein_id: Optional[int] = None,
                        project_id: Optional[int] = None) -> list[dict]:
        conn = self._get_connection()
        try:
            query = "SELECT * FROM experiments WHERE 1=1"
            params = []
            if molecule_id is not None:
                query += " AND molecule_id = ?"
                params.append(molecule_id)
            if protein_id is not None:
                query += " AND protein_id = ?"
                params.append(protein_id)
            if project_id is not None:
                query += " AND project_id = ?"
                params.append(project_id)
            query += " ORDER BY experiment_date DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ── SAR Datasets ──────────────────────────────────────────

    def create_sar_dataset(self, name: str, description: str = "",
                           project_id: Optional[int] = None) -> int:
        conn = self._get_connection()
        try:
            cur = conn.execute(
                "INSERT INTO sar_datasets (name, description, project_id) VALUES (?, ?, ?)",
                (name, description, project_id),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def add_molecule_to_sar_dataset(self, dataset_id: int, molecule_id: int,
                                     activity_value: Optional[float] = None,
                                     activity_label: str = ""):
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO sar_dataset_molecules
                   (dataset_id, molecule_id, activity_value, activity_label)
                   VALUES (?, ?, ?, ?)""",
                (dataset_id, molecule_id, activity_value, activity_label),
            )
            conn.commit()
        finally:
            conn.close()

    def get_sar_datasets(self, project_id: Optional[int] = None) -> list[dict]:
        conn = self._get_connection()
        try:
            if project_id is not None:
                rows = conn.execute(
                    "SELECT * FROM sar_datasets WHERE project_id = ? ORDER BY created_date DESC",
                    (project_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sar_datasets ORDER BY created_date DESC"
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_sar_dataset_molecules(self, dataset_id: int) -> list[dict]:
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """SELECT m.*, sdm.activity_value as sar_activity, sdm.activity_label
                   FROM molecules m
                   JOIN sar_dataset_molecules sdm ON m.id = sdm.molecule_id
                   WHERE sdm.dataset_id = ?
                   ORDER BY sdm.activity_value DESC""",
                (dataset_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ── QSAR Models ───────────────────────────────────────────

    def add_qsar_model(self, meta: dict) -> int:
        """Insert a qsar_models row from a metadata dict. Returns new id.

        Expected keys: name, dataset_name, n_molecules, activity_label,
        activity_transform, higher_is_better, cv_r2_mean, cv_r2_std,
        model_type, artifact_path, rdkit_version, sklearn_version, project_id.
        """
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """INSERT INTO qsar_models
                   (name, dataset_name, n_molecules, activity_label,
                    activity_transform, higher_is_better, cv_r2_mean, cv_r2_std,
                    model_type, artifact_path, rdkit_version, sklearn_version,
                    project_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    meta["name"], meta.get("dataset_name"), meta["n_molecules"],
                    meta["activity_label"], meta.get("activity_transform", "none"),
                    int(meta["higher_is_better"]),
                    meta.get("cv_r2_mean"), meta.get("cv_r2_std"),
                    meta.get("model_type", "RandomForestRegressor"),
                    meta["artifact_path"],
                    meta.get("rdkit_version"), meta.get("sklearn_version"),
                    meta.get("project_id"),
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_qsar_models(self, project_id: Optional[int] = None) -> list[dict]:
        conn = self._get_connection()
        try:
            if project_id is not None:
                rows = conn.execute(
                    "SELECT * FROM qsar_models WHERE project_id = ? ORDER BY created_date DESC",
                    (project_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM qsar_models ORDER BY created_date DESC"
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_qsar_model(self, model_id: int) -> None:
        """Delete the qsar_models row and unlink the artifact file.
        Tolerant of a missing file so it's safe as a rollback path.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT artifact_path FROM qsar_models WHERE id = ?", (model_id,)
            ).fetchone()
            conn.execute("DELETE FROM qsar_models WHERE id = ?", (model_id,))
            conn.commit()
        finally:
            conn.close()
        if row is not None and row["artifact_path"]:
            Path(row["artifact_path"]).unlink(missing_ok=True)
