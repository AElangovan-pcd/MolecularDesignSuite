-- Molecular Design Suite Database Schema

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS molecules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    smiles TEXT NOT NULL UNIQUE,
    canonical_smiles TEXT,
    name TEXT,
    formula TEXT,
    molecular_weight REAL,
    logp REAL,
    tpsa REAL,
    hbd INTEGER,
    hba INTEGER,
    rotatable_bonds INTEGER,
    aromatic_rings INTEGER,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    project_id INTEGER,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS molecular_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    molecule_id INTEGER NOT NULL,
    property_name TEXT NOT NULL,
    property_value REAL,
    property_text TEXT,
    calculation_method TEXT,
    calculated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (molecule_id) REFERENCES molecules(id) ON DELETE CASCADE,
    UNIQUE(molecule_id, property_name, calculation_method)
);

CREATE TABLE IF NOT EXISTS proteins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pdb_id TEXT,
    name TEXT,
    organism TEXT,
    sequence TEXT,
    structure_file_path TEXT,
    resolution REAL,
    binding_sites TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    project_id INTEGER,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    molecule_id INTEGER,
    protein_id INTEGER,
    assay_type TEXT,
    activity_value REAL,
    activity_unit TEXT,
    activity_relation TEXT DEFAULT '=',
    experiment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    project_id INTEGER,
    FOREIGN KEY (molecule_id) REFERENCES molecules(id) ON DELETE CASCADE,
    FOREIGN KEY (protein_id) REFERENCES proteins(id) ON DELETE SET NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS sar_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    project_id INTEGER,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS sar_dataset_molecules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    molecule_id INTEGER NOT NULL,
    activity_value REAL,
    activity_label TEXT,
    FOREIGN KEY (dataset_id) REFERENCES sar_datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (molecule_id) REFERENCES molecules(id) ON DELETE CASCADE,
    UNIQUE(dataset_id, molecule_id)
);

CREATE TABLE IF NOT EXISTS qsar_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    dataset_name TEXT,
    n_molecules INTEGER NOT NULL,
    activity_label TEXT NOT NULL,
    activity_transform TEXT DEFAULT 'none',
    higher_is_better INTEGER NOT NULL,
    cv_r2_mean REAL,
    cv_r2_std REAL,
    model_type TEXT DEFAULT 'RandomForestRegressor',
    artifact_path TEXT NOT NULL,
    rdkit_version TEXT,
    sklearn_version TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    project_id INTEGER,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_molecules_project ON molecules(project_id);
CREATE INDEX IF NOT EXISTS idx_molecules_smiles ON molecules(canonical_smiles);
CREATE INDEX IF NOT EXISTS idx_properties_molecule ON molecular_properties(molecule_id);
CREATE INDEX IF NOT EXISTS idx_experiments_molecule ON experiments(molecule_id);
CREATE INDEX IF NOT EXISTS idx_experiments_protein ON experiments(protein_id);
CREATE INDEX IF NOT EXISTS idx_qsar_models_project ON qsar_models(project_id);
