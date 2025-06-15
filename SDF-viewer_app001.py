import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import PandasTools
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode

def set_custom_aggrid_css():
    """Inject custom CSS for AgGrid header text wrapping and right alignment."""
    st.markdown("""
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
    """, unsafe_allow_html=True)

@st.cache_data
def mol_to_svg_str(smiles: str) -> str:
    """Convert a SMILES string to an HTML-embeddable SVG."""
    mol = Chem.MolFromSmiles(smiles)
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
        st.error(f"Filter error: {e}")
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

def get_svg_cellrenderer():
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

def build_aggrid_options(df: pd.DataFrame):
    """
    Build AgGrid options with explicit header wrapping and column sizing.
    """
    gb = GridOptionsBuilder.from_dataframe(df)

    svg_cellrenderer = get_svg_cellrenderer()
    
    # Index column
    gb.configure_column("Idx", 
                       header_name="Idx", 
                       width=60,
                       minWidth=60,
                       maxWidth=60,
                       pinned='left', 
                       resizable=False, 
                       headerClass="custom-header",
                       wrapHeaderText=True,
                       cellStyle={'textAlign': 'right'})  # Right align content
    
    # Structure column - keep structure images left aligned for visual clarity
    gb.configure_column("Structure", 
                       header_name="Structure", 
                       width=130,
                       minWidth=130,
                       maxWidth=130,
                       resizable=False, 
                       cellRenderer=svg_cellrenderer, 
                       headerClass="custom-header",
                       wrapHeaderText=True,
                       cellStyle={'textAlign': 'center'})  # Center align structure images
    
    # All other columns - FORCE WIDTH TO 50px and right align
    for col in df.columns:
        if col not in ["Idx", "Structure"]:
            gb.configure_column(col, 
                              width=50,
                              minWidth=50,
                              maxWidth=200,  # Allow some expansion but start at 50
                              flex=0,  # Disable flex sizing
                              resizable=True, 
                              headerClass="custom-header",
                              wrapHeaderText=True,
                              suppressSizeToFit=True,
                              suppressAutoSize=True,
                              cellStyle={'textAlign': 'right'},  # Right align content
                              type='numericColumn')  # Use numeric type for consistent right alignment

    # Configure default column properties with right alignment
    gb.configure_default_column(
        wrapText=True, 
        suppressSizeToFit=True,
        suppressAutoSize=True,
        flex=0,  # Disable flex for all columns
        cellStyle={'textAlign': 'right'}  # Right align all content by default
    )
    
    # Build grid options
    grid_options = gb.build()
    
    # Set row height and header height
    grid_options['getRowHeight'] = JsCode("function(params) { return 100; }")
    grid_options['headerHeight'] = 80
    
    # CRITICAL: These settings prevent all auto-sizing
    grid_options['suppressColumnVirtualisation'] = True
    grid_options['suppressAutoSize'] = True
    grid_options['suppressSizeToFit'] = True
    grid_options['skipHeaderOnAutoSize'] = True
    
    # Force column definitions to override any auto-sizing AND set right alignment
    grid_options['onGridReady'] = JsCode("""
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
    """)
    
    return grid_options

def display_aggrid_table(df: pd.DataFrame, grid_options: dict):
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

def load_data(uploaded_file):
    """Load SDF or CSV file and return as DataFrame."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'sdf':
        df = PandasTools.LoadSDF(uploaded_file, smilesName='SMILES', includeFingerprints=False)
    elif file_type == 'csv':
        df = pd.read_csv(uploaded_file)
        if 'SMILES' not in df.columns:
            st.error("CSV must contain a 'SMILES' column.")
            st.stop()
    else:
        st.error("Unsupported file type.")
        st.stop()
    if 'SMILES' not in df.columns:
        st.error("No SMILES found.")
        st.stop()
    return df

def main():
    st.set_page_config(layout="wide")
    st.title("SDF/CSV Molecule Viewer with AgGrid")
    set_custom_aggrid_css()

    uploaded_file = st.file_uploader("Upload a file (.sdf or .csv)", type=['sdf', 'csv'])

    if uploaded_file:
        raw_df = load_data(uploaded_file)
        df = prepare_dataframe(raw_df)
        df_filtered = filter_dataframe(df)
        grid_options = build_aggrid_options(df_filtered)
        st.subheader("Filtered Data with Molecule Images")
        display_aggrid_table(df_filtered, grid_options)

if __name__ == "__main__":
    main()