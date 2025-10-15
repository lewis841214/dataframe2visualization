"""
Main Streamlit application for Dataframe2Visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io

# Import our modules
from src.core.dataframe_processor import DataFrameProcessor
from src.core.data_validator import DataValidator
from src.ui.table_display import InteractiveTableDisplay
from src.ui.image_viewer import ImageModalViewer
from src.ui.controls import TableControls
from src.utils.streamlit_utils import (
    init_session_state, show_sidebar_info, display_error_message,
    display_success_message, create_dataframe_uploader, create_sample_data_selector,
    display_dataframe_info
)
from src.examples.sample_data import create_sample_dataframe

# Page configuration
st.set_page_config(
    page_title="Dataframe2Visualization",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function."""
    # Initialize session state
    init_session_state()
    
    # Sidebar
    show_sidebar_info()
    
    # Main content
    st.title("🖼️ Dataframe2Visualization")
    st.markdown("Interactive DataFrame display tool with image support")
    
    # File upload section
    st.header("📁 Data Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_df = create_dataframe_uploader()
    
    with col2:
        sample_option = create_sample_data_selector()
    
    # Load data
    df = None
    if uploaded_df is not None:
        df = uploaded_df
        st.session_state.dataframe_loaded = True
    elif sample_option and sample_option != "None":
        df = create_sample_dataframe(sample_option)
        st.session_state.dataframe_loaded = True
    
    # Process and display data
    if df is not None and st.session_state.dataframe_loaded:
        try:
            # Display basic DataFrame info
            # display_dataframe_info(df)
            
            # Process DataFrame
            # st.header("⚙️ Processing Data")
            with st.spinner("Processing DataFrame..."):
                processor = DataFrameProcessor()
                result = processor.process_dataframe(df)
                
                # Store processed data in session state
                st.session_state.processed_data = result['processed_data']
                st.session_state.column_metadata = result['column_metadata']
                
                # st.success(f"Successfully processed DataFrame with {result['original_shape'][0]} rows and {result['original_shape'][1]} columns")
                
                # # Display processing results
                # col1, col2, col3 = st.columns(3)
                # with col1:
                #     st.metric("Total Images", result['column_metadata'].get('total_images', 0))
                # with col2:
                #     image_columns = [col for col, meta in result['column_metadata'].items() 
                #                    if meta.get('contains_images', False)]
                #     st.metric("Image Columns", len(image_columns))
                # with col3:
                #     st.metric("Cache Usage", f"{processor.get_cache_info()['cache_usage_percent']:.1f}%")
            
            # Display interactive table
            # st.header("📋 Interactive Table")
            
            # Initialize UI components
            table_display = InteractiveTableDisplay()
            table_controls = TableControls()
            image_viewer = ImageModalViewer()
            
            # Get display DataFrame with processed images
            display_df = processor.get_display_dataframe()
            
            # Render controls on the display DataFrame (which has base64 images)
            controls_result = table_controls.render_controls(display_df, result['column_metadata'])
            
            # Get the final DataFrame with controls applied
            final_df = controls_result['final_data']
            
            # Render table with controlled data (now includes Pareto filtering)
            table_display.render_table(final_df, result['column_metadata'], result['processed_data'])
            
            # Data Analysis Section (separate from table)
            st.header("📊 Data Analysis")
            
            # Data scope selection
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_scope = st.selectbox(
                    "Analysis Scope",
                    ["Current Page", "Full Dataset", "Sampled (5000 points)"],
                    key="analysis_scope",
                    help="Choose which data to use for scatter plot analysis"
                )
            
            with col2:
                if analysis_scope == "Sampled (5000 points)":
                    max_sample_size = min(10000, len(df))
                    sample_size = st.slider(
                        "Sample Size", 
                        1000, 
                        max_sample_size, 
                        5000,
                        key="sample_size",
                        help=f"Number of data points to sample (max: {max_sample_size})"
                    )
                else:
                    sample_size = None
            
            # Get appropriate data for analysis
            if analysis_scope == "Current Page":
                analysis_df = final_df
                st.info(f"📄 Analyzing current page data: {len(analysis_df)} points")
            elif analysis_scope == "Full Dataset":
                analysis_df = df
                st.info(f"📊 Analyzing full dataset: {len(analysis_df)} points")
            else:  # Sampled
                if len(df) <= sample_size:
                    analysis_df = df
                    st.info(f"📊 Dataset size ({len(df)}) is smaller than sample size ({sample_size}). Using full dataset.")
                else:
                    analysis_df = df.sample(n=sample_size, random_state=42)
                    st.info(f"🎯 Analyzing sampled data: {len(analysis_df)} points (randomly sampled from {len(df)} total)")
            
            # Render scatter plot analysis with selected data
            table_display.render_scatter_plot_analysis(analysis_df)
            
            # Handle image clicks
            st.markdown("---")
            st.markdown("### Image Viewer")
            
            # Create a simple image viewer using session state
            if 'selected_image' not in st.session_state:
                st.session_state.selected_image = None
            
            # Show image viewer if an image is selected
            if st.session_state.selected_image:
                col_name, row_idx = st.session_state.selected_image
                if col_name in result['processed_data'] and row_idx < len(result['processed_data'][col_name]):
                    image_data = result['processed_data'][col_name][row_idx]
                    if isinstance(image_data, dict) and image_data.get('type') == 'image':
                        image_viewer.show_image_modal(image_data, f"{col_name}_{row_idx}")
                        
                        # Add a close button
                        if st.button("Close Image Viewer"):
                            st.session_state.selected_image = None
                            st.rerun()
            
            # Add controls to view specific images
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Show Sample Image"):
                    # Find first image to display
                    for col_name, col_data in result['processed_data'].items():
                        for idx, item in enumerate(col_data):
                            if isinstance(item, dict) and item.get('type') == 'image':
                                st.session_state.selected_image = (col_name, idx)
                                st.rerun()
                                break
                        else:
                            continue
                        break
            
            with col2:
                # Let users select which image to view
                image_columns = [col for col, meta in result['column_metadata'].items() 
                               if meta.get('contains_images', False)]
                
                if image_columns:
                    selected_col = st.selectbox("Select Image Column", image_columns)
                    if selected_col in result['processed_data']:
                        image_count = sum(1 for item in result['processed_data'][selected_col] 
                                       if isinstance(item, dict) and item.get('type') == 'image')
                        if image_count > 0:
                            selected_idx = st.selectbox("Select Image Index", range(image_count))
                            if st.button(f"View Image {selected_idx}"):
                                st.session_state.selected_image = (selected_col, selected_idx)
                                st.rerun()
            
            # Cache management
            st.header("🗄️ Cache Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear Image Cache"):
                    processor.clear_cache()
                    st.success("Image cache cleared")
                    st.rerun()
            
            with col2:
                cache_info = processor.get_cache_info()
                st.metric("Cache Size", f"{cache_info['cache_size']}/{cache_info['max_cache_size']}")
            
        except Exception as e:
            display_error_message(e, "while processing DataFrame")
            st.session_state.dataframe_loaded = False
    
    # Instructions
    if not st.session_state.dataframe_loaded:
        st.header("📖 How to Use")
        st.markdown("""
        1. **Upload Data**: Use the file uploader to upload a CSV, Excel, JSON, or Parquet file
        2. **Load Sample Data**: Choose from predefined sample datasets to see the tool in action
        3. **View Images**: Click on image thumbnails in the table to view them in full size
        4. **Filter & Search**: Use the controls to search, filter, and sort your data
        5. **Export**: Download your data in various formats
        
        **Supported Image Formats**: PNG, JPG, JPEG, BMP, TIFF, GIF
        **Supported Data Types**: Numbers, text, 2D/3D numpy arrays, PIL images, file paths
        """)
        
        # Show sample data preview
        st.header("🎯 Sample Data Preview")
        sample_df_raw = create_sample_dataframe("Mixed Data Types")
        
        # Process the sample DataFrame for display
        sample_processor = DataFrameProcessor()
        processed_sample_result = sample_processor.process_dataframe(sample_df_raw)
        sample_display_df = sample_processor.get_display_dataframe()
        
        st.dataframe(sample_display_df.head(), width='stretch')
        
        st.info("💡 **Tip**: Try loading the 'Mixed Data Types' sample to see images and data together!")

if __name__ == "__main__":
    main()
