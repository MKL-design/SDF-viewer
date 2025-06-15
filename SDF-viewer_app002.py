"""Streamlit application for viewing large SDF/CSV files.

Only a single slider is used for paging through the data. Molecule drawings are
cached using an LRU cache so that only a limited number of SVGs are stored in
memory at any time.
"""

from functools import lru_cache
from io import BytesIO

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import rdMolDraw2D
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode
from streamlit.runtime.uploaded_file_manager import UploadedFile


def set_custom_aggrid_css() -> None:
    """Inject custom CSS for AgGrid header text wrapping, right alignment, and shrink sidebar width."""
    st.markdown(
        """
    <style>
    /* Shrink sidebar width (class name may change in Streamlit updates) */
    section[data-testid="stSidebar"] {
        min-width: 180px !important;
        max-width: 220px !important;
        width: 180px !important;
    }
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
    .custom-header {
        white-space: normal !important;
        text-align: right !important;
        justify-content: flex-end !important;
        line-height: 1.2 !important;
        font-size: 14px;
        padding: 4px;
        word-wrap: break-word !important;
    }
    .ag-cell { text-align: right !important; justify-content: flex-end !important; }
    .ag-cell-value { text-align: right !important; }
    .ag-cell-numeric { text-align: right !important; }
    .ag-cell-wrapper { justify-content: flex-end !important; }
    .ag-cell-text { text-align: right !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )


@lru_cache(maxsize=500)
def _cached_svg(smiles: str) -> str:
    """Return an SVG string for the given SMILES structure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(120, 100)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("\n", "")


def mol_to_svg_str(smiles: str) -> str:
    """Wrap the cached SVG in a div so AgGrid can render it."""
    svg = _cached_svg(smiles)
    return f"<div>{svg}</div>" if svg else ""


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allow filtering of DataFrame using a query string from the sidebar.
    Returns the filtered DataFrame.
    """
    st.sidebar.subheader("Filter Table")
    query = st.sidebar.text_input("Filter Query (e.g. MW > 300 & LogP < 5):")
    try:
        if query:
            df = df.query(query)
    except Exception as e:
        st.sidebar.error(f"Filter error: {e}")
    return df


def prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add index and structure columns and reorder for display.
    Returns a new DataFrame with columns: Idx, Structure, <original columns>.
    Only called on the current visible slice to save memory.
    """
    df = raw_df.copy()
    df.insert(0, "Idx", range(1, len(df) + 1))
    df.insert(
        1,
        "Structure",
        df["SMILES"].apply(lambda smi: mol_to_svg_str(smi) if pd.notna(smi) else ""),
    )
    return df


def get_svg_cellrenderer() -> JsCode:
    """
    JavaScript cell renderer to correctly render HTML/SVG in AgGrid cells.
    """
    return JsCode(
        """
        class HtmlCellRenderer {
            init(params) {
                this.eGui = document.createElement('div');
                this.eGui.innerHTML = params.value;
            }
            getGui() {
                return this.eGui;
            }
        }
    """
    )


def build_aggrid_options(df: pd.DataFrame) -> dict:
    """
    Build AgGrid options with explicit header wrapping and column sizing.
    """
    gb = GridOptionsBuilder.from_dataframe(df)

    svg_cellrenderer = get_svg_cellrenderer()

    # Index column
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
    )  # Right align content

    # Structure column - keep structure images centered for visual clarity
    gb.configure_column(
        "Structure",
        header_name="Structure",
        width=130,
        minWidth=130,
        maxWidth=130,
        resizable=False,
        cellRenderer=svg_cellrenderer,
        headerClass="custom-header",
        wrapHeaderText=True,
        cellStyle={"textAlign": "center"},
    )  # Center align structure images

    # All other columns - force width and right align
    for col in df.columns:
        if col not in ["Idx", "Structure"]:
            gb.configure_column(
                col,
                width=50,
                minWidth=50,
                maxWidth=200,  # Allow some expansion but start at 50
                flex=0,  # Disable flex sizing
                resizable=True,
                headerClass="custom-header",
                wrapHeaderText=True,
                suppressSizeToFit=True,
                suppressAutoSize=True,
                cellStyle={"textAlign": "right"},  # Right align content
                type="numericColumn",
            )  # Use numeric type for consistent right alignment

    # Configure default column properties with right alignment
    gb.configure_default_column(
        wrapText=True,
        suppressSizeToFit=True,
        suppressAutoSize=True,
        flex=0,  # Disable flex for all columns
        cellStyle={"textAlign": "right"},  # Right align all content by default
    )

    # Build grid options
    grid_options = gb.build()

    # Set row height and header height
    grid_options["getRowHeight"] = JsCode("function(params) { return 100; }")
    grid_options["headerHeight"] = 80

    # CRITICAL: These settings prevent all auto-sizing
    grid_options["suppressColumnVirtualisation"] = True
    grid_options["suppressAutoSize"] = True
    grid_options["suppressSizeToFit"] = True
    grid_options["skipHeaderOnAutoSize"] = True

    # Force column definitions to override any auto-sizing AND set right alignment
    grid_options["onGridReady"] = JsCode(
        """
        function(params) {
            // Force column widths and right alignment after grid is ready
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
            
            // Apply right alignment to all existing cells (except structure)
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

    return grid_options


def display_aggrid_table(df: pd.DataFrame, grid_options: dict) -> None:
    """
    Render the DataFrame in AgGrid, left-aligned with custom header styling.
    """
    st.markdown(
        "<div style='display: flex; justify-content: flex-start;'>",
        unsafe_allow_html=True,
    )
    AgGrid(
        df,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        height=600,
        fit_columns_on_grid_load=False,  # CRITICAL: Keep this False
        reload_data=False,  # Change to False to prevent reloading
        theme="streamlit",
        update_mode="MANUAL",  # Use MANUAL instead of MODEL_CHANGED
        columns_auto_size_mode=None,  # Disable auto-sizing completely
    )
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data
def load_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Load SDF or CSV data from the uploaded file bytes.

    The result is cached by Streamlit so repeated navigation does not reload the
    file from disk.
    """
    file_type = file_name.split(".")[-1].lower()
    try:
        if file_type == "sdf":
            df = PandasTools.LoadSDF(
                BytesIO(file_bytes), smilesName="SMILES", includeFingerprints=False
            )
        elif file_type == "csv":
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file type.")
    except Exception as e:
        raise RuntimeError(f"Failed to load file: {e}") from e

    if "SMILES" not in df.columns:
        raise ValueError("No SMILES column found.")
    return df


def setup_page() -> None:
    """
    Configure the Streamlit page and apply custom styling.
    """
    st.set_page_config(layout="wide")
    st.title("SDF/CSV Molecule Viewer with AgGrid")
    set_custom_aggrid_css()


def main() -> None:
    """Run the Streamlit application.

    Only a single slider controls pagination to provide smooth scrolling. The
    sidebar contains all other inputs.
    """
    setup_page()

    # Sidebar widgets
    uploaded_file = st.sidebar.file_uploader(
        "Upload a file (.sdf or .csv)", type=["sdf", "csv"]
    )
    page_size = st.sidebar.number_input(
        "Rows per page", min_value=1, max_value=200, value=20
    )
    st.sidebar.markdown("---")  # Divider for sidebar
    if not uploaded_file:
        st.sidebar.info("Please upload a file to begin.")
        return

    # Load & filter data
    try:
        raw_df = load_data(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as e:
        st.sidebar.error(str(e))
        return
    df_filtered = filter_dataframe(raw_df)
    n = len(df_filtered)
    if n == 0:
        st.warning("No rows to display after filtering.")
        return

    # Slider for row paging - only in main window
    max_start = max(n - page_size + 1, 1)
    start = st.slider(
        "Starting row",
        min_value=1,
        max_value=max_start,
        value=1,
        help=f"Show {page_size} rows starting here",
    )
    end = min(start + page_size - 1, n)

    # Only generate SVGs for visible window
    df_page = df_filtered.iloc[start - 1 : end].copy()
    df_page = prepare_dataframe(df_page)

    st.subheader(f"Rows {start}–{end} of {n}")
    grid_options = build_aggrid_options(df_page)
    display_aggrid_table(df_page, grid_options)


if __name__ == "__main__":
    main()
