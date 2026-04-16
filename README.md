# Molecular Design Suite

A comprehensive molecular design application for drug discovery research and development. Built with RDKit and Streamlit, this platform provides tools for designing, analyzing, and optimizing molecules for therapeutic applications.

## Features

### Structure Editor (Sidebar)
- Built-in Ketcher molecular editor always available in the sidebar
- Draw molecules visually with ChemDraw-style tools (atoms, bonds, rings, templates)
- Real-time SMILES generation and molecular formula preview
- "Use This Molecule" button pushes drawn structures into any active module
- Bi-directional: load existing molecules back into the editor for modification via "Edit in Structure Editor" buttons throughout the app

### Molecular Input & Design
- Enter molecules via SMILES strings with real-time validation
- Draw molecules in the sidebar Structure Editor and push to input
- Upload molecular files (SDF, MOL, SMILES, CSV formats)
- Look up compounds from PubChem by name or CID
- Batch processing for multiple molecules at once

### Property Calculation
- Basic molecular properties (MW, LogP, TPSA, HBD, HBA, QED, etc.)
- Drug-likeness filters: Lipinski Rule of Five, Veber, Ghose, Egan, and Muegge rules
- ADMET property estimates (bioavailability, synthetic accessibility, BBB permeability)
- Structural alerts detection (PAINS)
- Batch analysis with distribution plots and correlation heatmaps

### Protein Structure Analysis
- **Search RCSB PDB** by keyword (protein name, organism, function) with results table showing title, organism, resolution, and method
- Download structures in multiple formats: PDB, mmCIF, FASTA, Biological Assembly
- Load protein structures by PDB ID or upload local PDB/CIF files
- Protein sequence composition and physicochemical analysis
- Interactive 3D visualization with customizable styles and color schemes
- Binding site identification with residue contact analysis

### SAR Analysis (Structure-Activity Relationships)
- Comprehensive molecular descriptor calculation
- Pairwise molecular similarity analysis with Tanimoto coefficients
- Murcko scaffold decomposition and analysis
- Chemical space visualization using PCA or t-SNE
- QSAR model building with Random Forest and cross-validation

### Drug Optimization
- Lead compound profiling with full property and filter assessment
- Analog generation using common medicinal chemistry transformations
- Scaffold hopping with bioisosteric replacements
- Multi-parameter optimization (MPO) with customizable desirability functions
- Side-by-side molecule comparison with maximum common substructure detection

### Data Management
- SQLite database for storing molecules, proteins, and experiments
- Project-based organization
- CSV export for all data
- Experiment recording with activity data

## Installation

### Prerequisites

You need **Miniconda** (or Anaconda) installed on your system.

- Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Follow the installer instructions for your operating system

### Step 1: Clone the Repository

```bash
git clone https://github.com/AElangovan-pcd/MolecularDesignSuite.git
cd MolecularDesignSuite
```

### Step 2: Create the Conda Environment

This installs Python 3.11 and RDKit:

```bash
conda create -n moldesign python=3.11 rdkit -c conda-forge -y
```

If prompted to accept Terms of Service, run the commands shown and retry.

### Step 3: Activate the Environment

```bash
conda activate moldesign
```

### Step 4: Install Additional Dependencies

```bash
pip install streamlit plotly seaborn biopython py3Dmol stmol pubchempy chembl-webresource-client scikit-learn pillow streamlit-ketcher
```

### Step 5: Run the Application

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

Then open your browser to **http://localhost:8501**.

## Windows Quick Start

After completing the installation steps above, you can use the included `run.bat` file:

1. Open **Command Prompt** (not PowerShell)
2. Navigate to the project folder:
   ```
   cd /d C:\path\to\MolecularDesignSuite
   ```
3. Run:
   ```
   run.bat
   ```

> **Important:** If you see an error about missing modules or the app crashes silently, make sure the `run.bat` file points to the correct conda environment path. Open `run.bat` in a text editor and update the `CONDA_ENV` path to match your system.

## Troubleshooting

### "streamlit is not recognized"
You need to activate the conda environment first:
```bash
conda activate moldesign
```
Or use the full path to Python:
```bash
C:\Users\<username>\.conda\envs\moldesign\python.exe -m streamlit run app.py
```

### RDKit import crashes silently on Windows
This happens when RDKit's DLL files are not on the system PATH. The `run.bat` file handles this by adding the correct paths. If running manually, add these to your PATH before launching:
```
set PATH=C:\Users\<username>\.conda\envs\moldesign;C:\Users\<username>\.conda\envs\moldesign\Library\bin;%PATH%
```

### "conda is not recognized"
- On Windows, use **Anaconda Prompt** instead of PowerShell or Command Prompt
- Or add Miniconda to your system PATH during installation

### PubChem lookup not working
Ensure you have internet access and `pubchempy` is installed:
```bash
pip install pubchempy
```

### 3D visualization not showing
Install the required packages:
```bash
pip install py3Dmol stmol
```

## Project Structure

```
MolecularDesignSuite/
├── app.py                          # Main Streamlit application
├── run.bat                         # Windows launch script
├── requirements.txt                # Python dependencies
├── modules/
│   ├── molecular_input.py          # Molecule input and design
│   ├── property_calc.py            # Property calculation engine
│   ├── protein_analysis.py         # Protein structure analysis
│   ├── sar_analysis.py             # SAR and QSAR analysis
│   └── drug_optimization.py        # Drug optimization tools
├── database/
│   ├── db_manager.py               # SQLite database operations
│   └── schema.sql                  # Database schema
├── utils/
│   ├── rdkit_utils.py              # RDKit utility functions
│   ├── visualization.py            # Plotly chart utilities
│   ├── file_handlers.py            # File import/export handlers
│   └── editor_helpers.py           # Structure Editor integration helpers
└── data/
    ├── reference_compounds/        # Reference molecule files
    └── protein_structures/         # Downloaded PDB files
```

## Usage Guide

### Getting Started

1. Launch the app and select a module from the sidebar
2. Optionally create a **Project** to organize your work
3. Use the **Structure Editor** in the sidebar to draw molecules visually, or enter SMILES directly in the **Molecular Input & Design** module

### Example: Drawing and Analyzing a Molecule

1. Use the **Structure Editor** in the sidebar to draw a molecule (or type a SMILES)
2. Click **Use This Molecule** to push it into the active module
3. In **Molecular Input & Design**, view the 2D structure and basic properties
4. Save it to the database
5. Switch to **Property Calculation** to see drug-likeness filters and ADMET estimates
6. Use **Drug Optimization** > **Lead Profiling** for a comprehensive assessment
7. Click **Edit in Structure Editor** on any molecule to modify it visually

### Example: Searching for a Protein Structure

1. Go to **Protein Analysis** > **Load Protein**
2. Select **Search RCSB** and enter a keyword (e.g., "insulin receptor")
3. Browse results and select an entry
4. Choose a download format (PDB, mmCIF, FASTA, or Biological Assembly)
5. Click **Download & Load** to fetch and parse the structure

### Example: Analyzing a Drug Molecule

1. Go to **Molecular Input & Design**
2. Enter a SMILES string, e.g., `CC(=O)OC1=CC=CC=C1C(=O)O` (Aspirin)
3. View the 2D structure and basic properties
4. Save it to the database
5. Switch to **Property Calculation** to see drug-likeness filters and ADMET estimates
6. Use **Drug Optimization** > **Lead Profiling** for a comprehensive assessment

### Example: Comparing Molecules

1. Go to **Drug Optimization** > **Molecule Comparison**
2. Enter two SMILES strings
3. View side-by-side structures, property comparison, drug-likeness filters, and the maximum common substructure

### Example: Building a QSAR Model

1. Prepare a CSV file with `SMILES` and `Activity` columns
2. Go to **SAR Analysis** > **QSAR Modeling**
3. Upload your CSV file
4. Select the SMILES and activity columns
5. Click **Build QSAR Model** to train a Random Forest model with cross-validation

## Technologies

- **RDKit** - Cheminformatics toolkit
- **Streamlit** - Web application framework
- **Ketcher** (via streamlit-ketcher) - Visual molecular structure editor
- **Plotly** - Interactive visualizations
- **py3Dmol** - 3D molecular visualization
- **BioPython** - Protein sequence and structure analysis
- **scikit-learn** - Machine learning for QSAR
- **PubChemPy** - PubChem API integration
- **RCSB PDB API** - Protein structure search and download
- **SQLite** - Local database storage

## License

This project is for educational and research purposes.
