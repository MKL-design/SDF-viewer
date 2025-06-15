"""Streamlit application for viewing large SDF/CSV files.

Enhanced navigation with pagination controls, search, and optimized memory usage.
Only visible molecules are rendered and cached using LRU cache.
"""

from functools import lru_cache
from io import BytesIO
import math

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
    section[data-testid="stSidebar"] {
        min-width: 220px !important;
        max-width: 400px !important;
        width: 280px !important;
        resize: horizontal !important;
        overflow: auto !important;
    }
    /* Pagination controls styling */
    .pagination-container {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 10px 0;
        justify-content: center;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 10px 0;
    }
    .pagination-button {
        background-color: #0066cc;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
    }
    .pagination-button:hover {
        background-color: #0052a3;
    }
    .pagination-button:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
    }
    /* AgGrid styling */
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


def search_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allow searching/filtering of DataFrame using multiple methods.
    Returns the filtered DataFrame.
    """
    st.sidebar.subheader("🔍 Search & Filter")
    
    # Text search in SMILES
    smiles_search = st.sidebar.text_input("Search SMILES:", placeholder="e.g., benzene, CCO")
    
    # Column-based search
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        st.sidebar.write("**Numeric Filters:**")
        filters = {}
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            col_min, col_max = float(df[col].min()), float(df[col].max())
            if col_min != col_max:  # Only show slider if there's a range
                filters[col] = st.sidebar.slider(
                    f"{col}",
                    min_value=col_min,
                    max_value=col_max,
                    value=(col_min, col_max),
                    key=f"filter_{col}"
                )
    
    # Advanced query
    with st.sidebar.expander("Advanced Query"):
        query = st.text_input("Pandas Query:", placeholder="e.g., MW > 300 & LogP < 5")
    
    # Apply filters
    filtered_df = df.copy()
    
    try:
        # SMILES search
        if smiles_search:
            mask = filtered_df['SMILES'].str.contains(smiles_search, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        # Numeric filters
        if 'filters' in locals():
            for col, (min_val, max_val) in filters.items():
                filtered_df = filtered_df[
                    (filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)
                ]
        
        # Advanced query
        if query:
            filtered_df = filtered_df.query(query)
            
    except Exception as e:
        st.sidebar.error(f"Filter error: {e}")
    
    if len(filtered_df) != len(df):
        st.sidebar.success(f"Filtered: {len(filtered_df):,} / {len(df):,} rows")
    
    return filtered_df


def prepare_dataframe(raw_df: pd.DataFrame, start_idx: int = 0) -> pd.DataFrame:
    """
    Add index and structure columns and reorder for display.
    Returns a new DataFrame with columns: Idx, Structure, <original columns>.
    start_idx allows showing global row numbers even in filtered/paginated views.
    """
    df = raw_df.copy()
    df.insert(0, "Idx", range(start_idx + 1, start_idx + len(df) + 1))
    df.insert(
        1,
        "Structure",
        df["SMILES"].apply(lambda smi: mol_to_svg_str(smi) if pd.notna(smi) else ""),
    )
    return df


def get_svg_cellrenderer() -> JsCode:
    """JavaScript cell renderer to correctly render HTML/SVG in AgGrid cells."""
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
    """Build AgGrid options with explicit header wrapping and column sizing."""
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
    )

    # Structure column
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
    )

    # All other columns
    for col in df.columns:
        if col not in ["Idx", "Structure"]:
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

    gb.configure_default_column(
        wrapText=True,
        suppressSizeToFit=True,
        suppressAutoSize=True,
        flex=0,
        cellStyle={"textAlign": "right"},
    )

    grid_options = gb.build()
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

    return grid_options


def display_pagination_controls(current_page: int, total_pages: int, key_prefix: str = ""):
    """
    Display enhanced pagination controls with jump-to-page functionality.
    Returns the new page number (1-indexed).
    """
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        first_page = st.button("⏮️ First", key=f"{key_prefix}_first", 
                              disabled=(current_page == 1))
        
    with col2:
        prev_page = st.button("◀️ Prev", key=f"{key_prefix}_prev", 
                             disabled=(current_page == 1))
    
    with col3:
        # Jump to page input
        jump_page = st.number_input(
            f"Page {current_page} of {total_pages}",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            key=f"{key_prefix}_jump"
        )
    
    with col4:
        next_page = st.button("Next ▶️", key=f"{key_prefix}_next", 
                             disabled=(current_page == total_pages))
    
    with col5:
        last_page = st.button("Last ⏭️", key=f"{key_prefix}_last", 
                             disabled=(current_page == total_pages))
    
    # Determine new page
    new_page = current_page
    if first_page:
        new_page = 1
    elif prev_page and current_page > 1:
        new_page = current_page - 1
    elif next_page and current_page < total_pages:
        new_page = current_page + 1
    elif last_page:
        new_page = total_pages
    elif jump_page != current_page:
        new_page = jump_page
    
    return new_page


def display_aggrid_table(df: pd.DataFrame, grid_options: dict) -> None:
    """Render the DataFrame in AgGrid with custom styling."""
    st.markdown(
        "<div style='display: flex; justify-content: flex-start;'>",
        unsafe_allow_html=True,
    )
    AgGrid(
        df,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        height=600,
        fit_columns_on_grid_load=False,
        reload_data=False,
        theme="streamlit",
        update_mode="MANUAL",
        columns_auto_size_mode=None,
    )
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data
def load_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Load SDF or CSV data from the uploaded file bytes."""
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
    """Configure the Streamlit page and apply custom styling."""
    st.set_page_config(layout="wide", page_title="Molecule Viewer", page_icon="🧪")
    st.title("🧪 SDF/CSV Molecule Viewer")
    st.markdown("*Enhanced navigation with search, filtering, and pagination*")
    set_custom_aggrid_css()


def main() -> None:
    """Run the Streamlit application with enhanced navigation."""
    setup_page()

    # Initialize session state for pagination
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1

    # Sidebar widgets
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload a file (.sdf or .csv)", type=["sdf", "csv"]
    )
    
    # Page size with reasonable options
    page_size_options = [10, 20, 50, 100, 200]
    page_size = st.sidebar.selectbox(
        "📄 Rows per page",
        options=page_size_options,
        index=1  # Default to 20
    )
    
    # Performance settings
    with st.sidebar.expander("⚙️ Performance Settings"):
        cache_size = st.number_input(
            "SVG Cache Size", 
            min_value=100, 
            max_value=2000, 
            value=500,
            help="Number of molecule structures to keep in memory"
        )
        # Update cache size (this would require restarting the app to take effect)
        st.info("Cache size changes require app restart")
    
    st.sidebar.markdown("---")
    
    if not uploaded_file:
        st.sidebar.info("Please upload a file to begin.")
        st.info("👆 Upload an SDF or CSV file in the sidebar to get started!")
        return

    # Load & filter data
    try:
        raw_df = load_data(uploaded_file.getvalue(), uploaded_file.name)
        st.sidebar.success(f"✅ Loaded {len(raw_df):,} molecules")
    except Exception as e:
        st.sidebar.error(f"❌ {str(e)}")
        return
    
    # Search and filter
    df_filtered = search_dataframe(raw_df)
    n = len(df_filtered)
    
    if n == 0:
        st.warning("No rows to display after filtering.")
        return

    # Calculate pagination
    total_pages = math.ceil(n / page_size)
    
    # Reset page if it's out of bounds (e.g., after filtering)
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = 1
    
    # Display pagination controls at the top
    st.markdown("### Navigation")
    new_page = display_pagination_controls(
        st.session_state.current_page, 
        total_pages, 
        "top"
    )
    
    # Update session state if page changed
    if new_page != st.session_state.current_page:
        st.session_state.current_page = new_page
        st.rerun()
    
    # Calculate slice indices
    start_idx = (st.session_state.current_page - 1) * page_size
    end_idx = min(start_idx + page_size, n)
    
    # Display current view info
    st.markdown(
        f"**Showing rows {start_idx + 1:,}–{end_idx:,} of {n:,} "
        f"({len(raw_df):,} total)**"
    )
    
    # Get the current page of data
    df_page = df_filtered.iloc[start_idx:end_idx].copy()
    df_page = prepare_dataframe(df_page, start_idx)
    
    # Display the table
    grid_options = build_aggrid_options(df_page)
    display_aggrid_table(df_page, grid_options)
    
    # Display pagination controls at the bottom
    st.markdown("---")
    new_page_bottom = display_pagination_controls(
        st.session_state.current_page, 
        total_pages, 
        "bottom"
    )
    
    if new_page_bottom != st.session_state.current_page:
        st.session_state.current_page = new_page_bottom
        st.rerun()
    
    # Display cache info
    with st.expander("📊 Performance Info"):
        cache_info = _cached_svg.cache_info()
        st.write(f"**SVG Cache:** {cache_info.currsize}/{cache_info.maxsize} cached, "
                f"{cache_info.hits} hits, {cache_info.misses} misses")
        st.write(f"**Cache hit rate:** {cache_info.hits/(cache_info.hits + cache_info.misses)*100:.1f}%")


if __name__ == "__main__":
    main()