# SDF Viewer Streamlit App

This repository contains a Streamlit application for viewing and filtering molecules stored in SDF or CSV files. Molecules are displayed alongside their properties using `st_aggrid` for an interactive table.

## Prerequisites

- Python 3.8 or later
- The Python packages listed in [`requirements.txt`](requirements.txt)

RDKit can be installed via the `rdkit-pypi` package or from conda using `rdkit`.

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Run the Streamlit application with:

```bash
streamlit run SDF-viewer_app001.py
```

The web interface will open in your browser. Upload an `.sdf` or `.csv` file containing a `SMILES` column to explore and filter your molecules.
