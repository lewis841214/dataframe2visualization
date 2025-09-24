"""
Utility functions for Streamlit functionality.
"""

import streamlit as st
from typing import Any, Dict, List, Optional
import pandas as pd

def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'dataframe_loaded' not in st.session_state:
        st.session_state.dataframe_loaded = False
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'column_metadata' not in st.session_state:
        st.session_state.column_metadata = None
    
    if 'current_image_viewer' not in st.session_state:
        st.session_state.current_image_viewer = None
    
    if 'table_controls_state' not in st.session_state:
        st.session_state.table_controls_state = {}

def show_sidebar_info() -> None:
    """Display information in the sidebar."""
    st.sidebar.title("Dataframe2Visualization")
    st.sidebar.markdown("---")
    
    # App information
    st.sidebar.markdown("**About**")
    st.sidebar.markdown("Interactive DataFrame display tool with image support")
    
    # Features
    st.sidebar.markdown("**Features**")
    st.sidebar.markdown("• Mixed data type support")
    st.sidebar.markdown("• Image thumbnail display")
    st.sidebar.markdown("• Interactive image viewing")
    st.sidebar.markdown("• Search and filtering")
    st.sidebar.markdown("• Export capabilities")
    
    # Statistics
    if st.session_state.dataframe_loaded:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Current Data**")
        
        if st.session_state.processed_data:
            total_images = sum(
                meta.get('image_count', 0) 
                for meta in st.session_state.column_metadata.values()
            )
            st.sidebar.metric("Total Images", total_images)
        
        if st.session_state.column_metadata:
            image_columns = [
                col for col, meta in st.session_state.column_metadata.items()
                if meta.get('contains_images', False)
            ]
            st.sidebar.metric("Image Columns", len(image_columns))

def display_error_message(error: Exception, context: str = "") -> None:
    """
    Display error message in a user-friendly way.
    
    Args:
        error: Exception that occurred
        context: Additional context about the error
    """
    st.error(f"**Error{': ' + context if context else ''}**")
    st.error(str(error))
    
    if hasattr(error, '__traceback__'):
        st.exception(error)

def display_success_message(message: str) -> None:
    """
    Display success message.
    
    Args:
        message: Success message to display
    """
    st.success(message)

def display_info_message(message: str) -> None:
    """
    Display information message.
    
    Args:
        message: Information message to display
    """
    st.info(message)

def create_download_button(data: Any, filename: str, mime_type: str, 
                          button_text: str = "Download") -> None:
    """
    Create a download button for various data types.
    
    Args:
        data: Data to download
        filename: Name of the file
        mime_type: MIME type of the file
        button_text: Text to display on the button
    """
    if isinstance(data, str):
        st.download_button(
            label=button_text,
            data=data,
            file_name=filename,
            mime=mime_type
        )
    elif isinstance(data, bytes):
        st.download_button(
            label=button_text,
            data=data,
            file_name=filename,
            mime=mime_type
        )
    else:
        st.error("Unsupported data type for download")

def create_file_uploader(accepted_types: List[str], 
                        help_text: str = "Upload a file") -> Optional[Any]:
    """
    Create a file uploader with specified accepted types.
    
    Args:
        accepted_types: List of accepted file extensions
        help_text: Help text for the uploader
        
    Returns:
        Uploaded file object or None
    """
    return st.file_uploader(
        "Choose a file",
        type=accepted_types,
        help=help_text
    )

def create_dataframe_uploader() -> Optional[pd.DataFrame]:
    """
    Create a file uploader specifically for DataFrames.
    
    Returns:
        Uploaded DataFrame or None
    """
    uploaded_file = st.file_uploader(
        "Upload DataFrame",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
        help="Upload a CSV, Excel, JSON, or Parquet file"
    )
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            elif file_extension == 'parquet':
                df = pd.read_parquet(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    return None

def create_sample_data_selector() -> Optional[str]:
    """
    Create a selector for sample data.
    
    Returns:
        Selected sample data option or None
    """
    sample_options = [
        "None",
        "Random Images",
        "Mixed Data Types",
        "Large Dataset",
        "Performance Test (5000 points)",
        "Performance Test (10000 points)"
    ]
    
    selected = st.selectbox(
        "Load Sample Data",
        options=sample_options,
        help="Choose from predefined sample datasets"
    )
    
    if selected == "None":
        return None
    
    return selected

def display_dataframe_info(df: pd.DataFrame) -> None:
    """
    Display information about the DataFrame.
    
    Args:
        df: DataFrame to analyze
    """
    st.markdown("### DataFrame Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Info**")
        st.markdown(f"• Shape: {df.shape}")
        st.markdown(f"• Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        st.markdown(f"• Data Types: {len(df.dtypes.unique())}")
    
    with col2:
        st.markdown("**Column Types**")
        type_counts = df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            st.markdown(f"• {dtype}: {count}")

def create_progress_bar(total_steps: int, description: str = "Processing...") -> st.progress:
    """
    Create a progress bar for long-running operations.
    
    Args:
        total_steps: Total number of steps
        description: Description of the operation
        
    Returns:
        Streamlit progress bar object
    """
    return st.progress(0, text=description)

def update_progress_bar(progress_bar: st.progress, current_step: int, 
                       total_steps: int, description: str = None) -> None:
    """
    Update progress bar.
    
    Args:
        progress_bar: Progress bar to update
        current_step: Current step number
        total_steps: Total number of steps
        description: Optional new description
    """
    progress = (current_step + 1) / total_steps
    progress_bar.progress(progress, text=description)
