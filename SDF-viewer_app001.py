import streamlit as st
import pandas as pd
from io import BytesIO

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import PandasTools
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Central CSS block used for AgGrid styling
AGGRID_CSS = """
<style>
/* Target AgGrid header cells specifically */
.ag-header-cell-text {
    white-space: normal !important;
    text-align: right !important;
    justify-content: flex-end !important;
    line-height: 1.2 !important;
    font-size: 14px !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
}

.ag-header-cell {
    padding: 4px !important;
}

/* Custom header class - backup */
.custom-header {
    white-space: normal !important;
    text-align: right !important;
    justify-content: flex-end !important;
    line-height: 1.2 !important;
    font-size: 14px;
    padding: 4px;
    word-wrap: break-word !important;
}

/* Right align ALL cell content */
.ag-cell {
    text-align: right !important;
    justify-content: flex-end !important;
}

/* Right align cell content text specifically */
.ag-cell-value {
    text-align: right !important;
}

/* Ensure numeric columns stay right-aligned */
.ag-cell-numeric {
    text-align: right !important;
}

/* Ensure all cell wrappers are right aligned */
.ag-cell-wrapper {
    justify-content: flex-end !important;
}

/* Right align text columns too */
.ag-cell-text {
    text-align: right !important;
}
</style>
"""

def set_custom_aggrid_css() -> None:
    """Inject custom CSS for AgGrid header text wrapping and right alignment."""
    st.markdown(AGGRID_CSS, unsafe_allow_html=True)

@st.cache_data
def mol_to_svg_str(smiles: str) -> str:
    """Convert a SMILES string to an HTML-embeddable SVG.

    Returns an empty string if the SMILES cannot be parsed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return ""
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(120, 100)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("\n", "")
    return f"<div>{svg}</div>"

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Allow filtering of DataFrame using query input."""
    st.subheader("Filter Table")
    query = st.text_input("Filter Query (e.g. MW > 300 & LogP < 5):")
    try:
        if query:
            df = df.query(query)
    except Exception as e:
        st.error(f"Invalid filter expression: {e}")
    return df

def prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add index and structure columns and reorder for display.
    Returns a new DataFrame with columns: Idx, Structure, <original columns>.
    """
    df = raw_df.copy()
    df.insert(0, "Idx", range(1, len(df) + 1))
    df.insert(1, "Structure", df['SMILES'].apply(lambda smi: mol_to_svg_str(smi) if pd.notna(smi) else ""))
    return df

def get_svg_cellrenderer() -> JsCode:
    """
    JavaScript cell renderer to correctly render HTML/SVG in AgGrid cells.
    """
    return JsCode("""
        class HtmlCellRenderer {
            init(params) {
                this.eGui = document.createElement('div');
                this.eGui.innerHTML = params.value;
            }
            getGui() {
                return this.eGui;
            }
        }
    """)


def _configure_index_column(gb: GridOptionsBuilder) -> None:
    """Configure the AgGrid index column."""
    gb.configure_column(
        "Idx",
        header_name="Idx",
        width=60,
        minWidth=60,
        maxWidth=60,
        pinned="left",
        resizable=False,
        headerClass="custom-header",
        wrapHeaderText=True,
        cellStyle={"textAlign": "right"},
    )


def _configure_structure_column(gb: GridOptionsBuilder, renderer: JsCode) -> None:
    """Configure the structure column for SVG rendering."""
    gb.configure_column(
        "Structure",
        header_name="Structure",
        width=130,
        minWidth=130,
        maxWidth=130,
        resizable=False,
        cellRenderer=renderer,
        headerClass="custom-header",
        wrapHeaderText=True,
        cellStyle={"textAlign": "center"},
    )


def _configure_other_columns(gb: GridOptionsBuilder, df: pd.DataFrame) -> None:
    """Configure all remaining DataFrame columns."""
    for col in df.columns:
        if col in ["Idx", "Structure"]:
            continue
        gb.configure_column(
            col,
            width=50,
            minWidth=50,
            maxWidth=200,
            flex=0,
            resizable=True,
            headerClass="custom-header",
            wrapHeaderText=True,
            suppressSizeToFit=True,
            suppressAutoSize=True,
            cellStyle={"textAlign": "right"},
            type="numericColumn",
        )


def _apply_common_grid_options(grid_options: dict) -> None:
    """Apply common grid options for consistent layout."""
    grid_options["getRowHeight"] = JsCode("function(params) { return 100; }")
    grid_options["headerHeight"] = 80
    grid_options["suppressColumnVirtualisation"] = True
    grid_options["suppressAutoSize"] = True
    grid_options["suppressSizeToFit"] = True
    grid_options["skipHeaderOnAutoSize"] = True
    grid_options["onGridReady"] = JsCode(
        """
        function(params) {
            const columnDefs = params.api.getColumnDefs();
            columnDefs.forEach(function(colDef) {
                if (colDef.field !== 'Idx' && colDef.field !== 'Structure') {
                    colDef.width = 50;
                    colDef.minWidth = 50;
                    colDef.cellStyle = {'text-align': 'right'};
                } else if (colDef.field === 'Idx') {
                    colDef.cellStyle = {'text-align': 'right'};
                } else if (colDef.field === 'Structure') {
                    colDef.cellStyle = {'text-align': 'center'};
                }
                colDef.headerClass = 'custom-header';
            });
            params.api.setColumnDefs(columnDefs);

            const allCells = document.querySelectorAll('.ag-cell');
            allCells.forEach(function(cell) {
                const colId = cell.getAttribute('col-id');
                if (colId === 'Structure') {
                    cell.style.textAlign = 'center';
                    cell.style.justifyContent = 'center';
                } else {
                    cell.style.textAlign = 'right';
                    cell.style.justifyContent = 'flex-end';
                }
            });
        }
        """
    )

def build_aggrid_options(df: pd.DataFrame) -> dict:
    """Build AgGrid options with explicit header wrapping and column sizing."""
    gb = GridOptionsBuilder.from_dataframe(df)
    svg_cellrenderer = get_svg_cellrenderer()

    _configure_index_column(gb)
    _configure_structure_column(gb, svg_cellrenderer)
    _configure_other_columns(gb, df)

    gb.configure_default_column(
        wrapText=True,
        suppressSizeToFit=True,
        suppressAutoSize=True,
        flex=0,
        cellStyle={"textAlign": "right"},
    )

    grid_options = gb.build()
    _apply_common_grid_options(grid_options)
    return grid_options

def display_aggrid_table(df: pd.DataFrame, grid_options: dict) -> None:
    """
    Render the DataFrame in AgGrid, left-aligned with custom header styling.
    """
    st.markdown("<div style='display: flex; justify-content: flex-start;'>", unsafe_allow_html=True)
    AgGrid(
        df,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        height=600,
        fit_columns_on_grid_load=False,  # CRITICAL: Keep this False
        reload_data=False,  # Change to False to prevent reloading
        theme='streamlit',
        update_mode='MANUAL',  # Use MANUAL instead of MODEL_CHANGED
        columns_auto_size_mode=None  # Disable auto-sizing completely
    )
    st.markdown("</div>", unsafe_allow_html=True)

@st.cache_data
def load_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Load SDF or CSV data from the uploaded file bytes."""
    file_type = file_name.split('.')[-1].lower()
    try:
        if file_type == 'sdf':
            df = PandasTools.LoadSDF(
                BytesIO(file_bytes), smilesName='SMILES', includeFingerprints=False
            )
        elif file_type == 'csv':
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file type.")
    except Exception as e:
        raise RuntimeError(f"Failed to load file: {e}") from e

    if 'SMILES' not in df.columns:
        raise ValueError("No SMILES column found.")

    return df

def setup_page() -> None:
    """Configure the Streamlit page and apply custom styling."""
    st.set_page_config(layout="wide")
    st.title("SDF/CSV Molecule Viewer with AgGrid")
    set_custom_aggrid_css()


def process_uploaded_file(uploaded_file: UploadedFile) -> None:
    """Load, filter and display the uploaded file."""
    try:
        raw_df = load_data(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as e:
        st.error(str(e))
        return

    df = prepare_dataframe(raw_df)
    df_filtered = filter_dataframe(df)
    grid_options = build_aggrid_options(df_filtered)
    st.subheader("Filtered Data with Molecule Images")
    display_aggrid_table(df_filtered, grid_options)


def main() -> None:
    setup_page()
    uploaded_file = st.file_uploader("Upload a file (.sdf or .csv)", type=['sdf', 'csv'])
    if uploaded_file:
        process_uploaded_file(uploaded_file)

if __name__ == "__main__":
    main()
