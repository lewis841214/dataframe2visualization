"""
Interactive table display module for Dataframe2Visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Tuple
from scipy.stats import rankdata
from ..config.settings import AppConfig

class InteractiveTableDisplay:
    """Renders DataFrame with mixed content and handles image interactions."""
    
    def __init__(self):
        """Initialize the InteractiveTableDisplay."""
        self.clicked_image_key = None
        self.clicked_image_data = None
    
    def render_table(self, df: pd.DataFrame, column_metadata: Dict[str, Any], 
                    processed_data: Dict[str, Any]) -> None:
        """
        Render the interactive table with mixed content.
        
        Args:
            df: DataFrame to display
            column_metadata: Metadata about DataFrame columns
            processed_data: Processed data from DataFrameProcessor
        """
        # st.markdown("### Data Table")
        
        # Display table information
        self._render_table_info(df, column_metadata)
        
        # Create interactive table
        self._create_interactive_table(df, column_metadata, processed_data)
        
        # Handle image clicks
        self._handle_image_interactions()
    
    def _render_table_info(self, df: pd.DataFrame, column_metadata: Dict[str, Any]) -> None:
        """
        Render table information and statistics.
        
        Args:
            df: DataFrame to display
            column_metadata: Column metadata
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(df))
        
        with col2:
            st.metric("Total Columns", len(df.columns))
        
        with col3:
            image_columns = [col for col, meta in column_metadata.items() 
                           if meta.get('contains_images', False)]
            st.metric("Image Columns", len(image_columns))
        
        with col4:
            total_images = sum(meta.get('image_count', 0) for meta in column_metadata.values())
            st.metric("Total Images", total_images)
        
        st.markdown("---")
    
    def _create_interactive_table(self, df: pd.DataFrame, column_metadata: Dict[str, Any],
                                processed_data: Dict[str, Any]) -> None:
        """
        Create the interactive table with clickable images.
        
        Args:
            df: DataFrame to display
            column_metadata: Column metadata
            processed_data: Processed data
        """
        # Use Streamlit's native dataframe rendering (safer and no iframe security issues)
        self._render_streamlit_dataframe(df, column_metadata, processed_data)
        
        # Note: Custom HTML rendering removed due to iframe sandbox security warnings
        # The native st.dataframe provides sufficient functionality without security risks
    
    def _generate_html_table(self, df: pd.DataFrame, column_metadata: Dict[str, Any],
                           processed_data: Dict[str, Any]) -> str:
        """
        Generate HTML table with clickable images.
        
        Args:
            df: DataFrame to display
            column_metadata: Column metadata
            processed_data: Processed data
            
        Returns:
            HTML string for the table
        """
        html_parts = []
        
        # Get configurable CSS width from session state
        css_width = st.session_state.get("text_css_width", 200)
        
        # CSS styles with configurable width
        html_parts.append(f"""
        <style>
        .dataframe-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }}
        .dataframe-table th, .dataframe-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }}
        .dataframe-table th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .dataframe-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .dataframe-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .clickable-image {{
            cursor: pointer;
            border: 2px solid transparent;
            transition: border-color 0.3s;
        }}
        .clickable-image:hover {{
            border-color: #007bff;
        }}
        .image-cell {{
            text-align: center;
            vertical-align: middle;
        }}
        .text-cell {{
            max-width: {css_width}px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        </style>
        """)
        
        # Table header
        html_parts.append('<table class="dataframe-table">')
        html_parts.append('<thead><tr>')
        for col in df.columns:
            col_meta = column_metadata.get(col, {})
            if col_meta.get('contains_images', False):
                html_parts.append(f'<th class="image-cell">{col} (Images)</th>')
            else:
                html_parts.append(f'<th>{col}</th>')
        html_parts.append('</tr></thead>')
        
        # Table body
        html_parts.append('<tbody>')
        for idx, row in df.iterrows():
            html_parts.append('<tr>')
            for col in df.columns:
                col_meta = column_metadata.get(col, {})
                cell_value = row[col]
                
                if col_meta.get('contains_images', False):
                    # Handle image cells
                    cell_html = self._generate_image_cell_html(cell_value, col, idx, processed_data)
                    html_parts.append(f'<td class="image-cell">{cell_html}</td>')
                else:
                    # Handle text/numeric cells
                    cell_html = self._generate_text_cell_html(cell_value)
                    html_parts.append(f'<td class="text-cell">{cell_html}</td>')
            
            html_parts.append('</tr>')
        html_parts.append('</tbody>')
        html_parts.append('</table>')
        
        # JavaScript for image clicks
        html_parts.append(self._generate_javascript())
        
        return ''.join(html_parts)
    
    def _generate_image_cell_html(self, cell_value: Any, col_name: str, row_idx: int,
                                processed_data: Dict[str, Any]) -> str:
        """
        Generate HTML for image cells.
        
        Args:
            cell_value: Cell value
            col_name: Column name
            row_idx: Row index
            processed_data: Processed data
            
        Returns:
            HTML string for the image cell
        """
        # Handle None and NaN values safely
        if cell_value is None:
            return '<span style="color: #999;">N/A</span>'
        
        try:
            if pd.isna(cell_value):
                return '<span style="color: #999;">N/A</span>'
        except (ValueError, TypeError):
            # If pd.isna fails, assume it's not None
            pass
        
        # Check if this is a base64 image string
        if isinstance(cell_value, str) and cell_value.startswith('data:image'):
            # This is a processed image thumbnail
            image_key = f"{col_name}_{row_idx}"
            return f'''
            <img src="{cell_value}" 
                 alt="Image {image_key}" 
                 style="max-width: 100px; max-height: 100px; object-fit: contain; cursor: pointer; border: 1px solid #ddd; border-radius: 4px;"
                 title="Click to view full size">
            '''
        
        # Check if this is an error message
        elif isinstance(cell_value, str) and cell_value.startswith('Error:'):
            return f'<span style="color: #dc3545;">{cell_value}</span>'
        
        # Check if this is a numpy array (raw image data)
        elif hasattr(cell_value, 'shape') and len(cell_value.shape) in [2, 3]:
            return f'<span style="color: #007bff;">🖼️ Image ({cell_value.shape[0]}x{cell_value.shape[1]})</span>'
        
        # Fallback for other data types
        else:
            return f'<span>{str(cell_value)}</span>'
    
    def _generate_text_cell_html(self, cell_value: Any) -> str:
        """
        Generate HTML for text/numeric cells.
        
        Args:
            cell_value: Cell value
            
        Returns:
            HTML string for the text cell
        """
        # Handle None and NaN values safely
        if cell_value is None:
            return '<span style="color: #999;">N/A</span>'
        
        try:
            if pd.isna(cell_value):
                return '<span style="color: #999;">N/A</span>'
        except (ValueError, TypeError):
            # If pd.isna fails, assume it's not None
            pass
        
        # Get configurable character limit from session state
        char_limit = st.session_state.get("text_char_limit", 50)
        
        # Truncate long text based on configurable limit
        text = str(cell_value)
        if len(text) > char_limit:
            text = text[:char_limit-3] + "..."
        
        return f'<span title="{str(cell_value)}">{text}</span>'
    
    def _generate_javascript(self) -> str:
        """Generate JavaScript for image interactions."""
        return """
        <script>
        function showImage(imageKey, colName, rowIdx) {
            // Create a custom event to communicate with Streamlit
            const event = new CustomEvent('imageClick', {
                detail: {
                    imageKey: imageKey,
                    colName: colName,
                    rowIdx: rowIdx
                }
            });
            document.dispatchEvent(event);
            
            // Also try to communicate with Streamlit directly
            if (window.parent && window.parent.postMessage) {
                window.parent.postMessage({
                    type: 'imageClick',
                    imageKey: imageKey,
                    colName: colName,
                    rowIdx: rowIdx
                }, '*');
            }
        }
        
        // Add click event listeners to all clickable images
        document.addEventListener('DOMContentLoaded', function() {
            const images = document.querySelectorAll('.clickable-image');
            images.forEach(img => {
                img.addEventListener('click', function() {
                    const imageKey = this.alt.replace('Image ', '');
                    const colName = this.closest('td').getAttribute('data-col');
                    const rowIdx = this.closest('tr').getAttribute('data-row');
                    showImage(imageKey, colName, rowIdx);
                });
            });
        });
        </script>
        """
    
    def _render_streamlit_dataframe(self, df: pd.DataFrame, column_metadata: Dict[str, Any],
                                  processed_data: Dict[str, Any]) -> None:
        """
        Alternative: Render using Streamlit's dataframe with custom formatting.
        
        Args:
            df: DataFrame to display
            column_metadata: Column metadata
            processed_data: Processed data
        """
        # Create a display DataFrame with formatted content
        display_df = df.copy()
        
        # Format image columns
        for col in df.columns:
            col_meta = column_metadata.get(col, {})
            if col_meta.get('contains_images', False):
                # Replace image data with clickable placeholders
                display_df[col] = display_df[col].apply(
                    lambda x: "🖼️ Click to view" if pd.notna(x) else "N/A"
                )
        
        # Display using Streamlit's dataframe
        st.dataframe(display_df, width='stretch', height=AppConfig.DEFAULT_TABLE_HEIGHT)
    
    def _handle_image_interactions(self) -> None:
        """Handle image click interactions."""
        # This would be implemented to work with the HTML table
        # For now, we'll use Streamlit's session state to track clicks
        
        if 'image_clicked' not in st.session_state:
            st.session_state.image_clicked = False
        
        if 'clicked_image_data' not in st.session_state:
            st.session_state.clicked_image_data = None
        
        # Check if an image was clicked (this would be set by JavaScript)
        if st.session_state.image_clicked:
            st.session_state.image_clicked = False
            # Handle the clicked image
            self._process_image_click(st.session_state.clicked_image_data)
    
    def _process_image_click(self, image_data: Dict[str, Any]) -> None:
        """
        Process image click event.
        
        Args:
            image_data: Data about the clicked image
        """
        if not image_data:
            return
        
        # Store clicked image information
        self.clicked_image_key = image_data.get('imageKey')
        self.clicked_image_data = image_data
        
        # Show success message
        st.success(f"Image clicked: {self.clicked_image_key}")
        
        # This would trigger the image modal viewer
        # The main app would handle this
    
    def get_clicked_image_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the clicked image."""
        if self.clicked_image_key and self.clicked_image_data:
            return {
                'key': self.clicked_image_key,
                'data': self.clicked_image_data
            }
        return None
    
    def clear_clicked_image(self) -> None:
        """Clear the clicked image state."""
        self.clicked_image_key = None
        self.clicked_image_data = None
    
    def render_scatter_plot_analysis(self, df: pd.DataFrame) -> None:
        """
        Render scatter plot analysis interface.
        
        Args:
            df: DataFrame to analyze
        """
        st.markdown("---")
        st.markdown("### 📊 Scatter Plot Analysis")
        
        # Check data size and provide memory warnings
        data_size = len(df)
        if data_size > 10000:
            st.warning("⚠️ **Large Dataset Warning**: Analyzing {:,} data points. This may take longer and use more memory. Consider using sampling for better performance.".format(data_size))
        elif data_size > 5000:
            st.info("ℹ️ **Performance Notice**: Analyzing {:,} data points. Plot generation may take a moment.".format(data_size))
        else:
            st.success("✅ Analyzing {:,} data points - optimal performance expected.".format(data_size))
        
        # Get numeric columns
        numeric_cols = self._get_numeric_columns(df)
        
        if len(numeric_cols) < 1:
            st.warning("⚠️ Need at least 1 numeric column for plotting.")
            st.info(f"Available numeric columns: {numeric_cols}")
            return
        
        # Plot type selection
        st.markdown("**📊 Plot Type Selection**")
        plot_type = st.radio(
            "Select Plot Type",
            ["1D CDF", "1D Histogram",  "2D Scatter Plot", "3D Scatter Plot"],
            key="plot_type_selector",
            horizontal=True
        )
        
        st.markdown("---")
        
        # Column selection based on plot type
        if plot_type == "1D Histogram":
            # Single column selection for histogram
            selected_col1 = st.selectbox(
                "Select Column for Histogram",
                numeric_cols,
                key="histogram_col"
            )
            
            selected_col2 = None
            selected_col3 = None
            
            # Validate and create histogram
            if selected_col1:
                is_valid, error_msg = self._validate_column_for_histogram(df, selected_col1)
                
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                    return
                
                # Calculate histogram statistics
                hist_stats = self._calculate_histogram_statistics(df, selected_col1)
                
                if not hist_stats:
                    st.error("❌ Unable to calculate histogram statistics.")
                    return
                
                # Display statistics
                self._display_histogram_statistics(hist_stats, selected_col1)
                
                # Create and display histogram
                with st.spinner(f"Generating histogram for {len(df)} data points..."):
                    self._create_histogram(df, selected_col1, hist_stats)
        
        elif plot_type == "1D CDF":
            # Single column selection for CDF
            selected_col1 = st.selectbox(
                "Select Column for CDF",
                numeric_cols,
                key="cdf_col"
            )
            
            selected_col2 = None
            selected_col3 = None
            
            # Validate and create CDF
            if selected_col1:
                is_valid, error_msg = self._validate_column_for_histogram(df, selected_col1)
                
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                    return
                
                # Calculate statistics (reuse histogram stats)
                hist_stats = self._calculate_histogram_statistics(df, selected_col1)
                
                if not hist_stats:
                    st.error("❌ Unable to calculate statistics.")
                    return
                
                # Display statistics
                self._display_histogram_statistics(hist_stats, selected_col1)
                
                # Create and display CDF
                with st.spinner(f"Generating CDF for {len(df)} data points..."):
                    self._create_cdf(df, selected_col1, hist_stats)
        
        elif plot_type == "2D Scatter Plot":
            # Check for sufficient columns
            if len(numeric_cols) < 2:
                st.warning("⚠️ Need at least 2 numeric columns for 2D scatter plotting.")
                st.info(f"Available numeric columns: {numeric_cols}")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_col1 = st.selectbox(
                    "Select First Column (X-axis)",
                    numeric_cols,
                    key="scatter_col1"
                )
            
            with col2:
                selected_col2 = st.selectbox(
                    "Select Second Column (Y-axis)",
                    numeric_cols,
                    key="scatter_col2"
                )
            
            selected_col3 = None
        else:  # 3D Scatter Plot
            # Check for sufficient columns
            if len(numeric_cols) < 3:
                st.warning("⚠️ Need at least 3 numeric columns for 3D scatter plotting.")
                st.info(f"Available numeric columns: {numeric_cols}")
                return
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_col1 = st.selectbox(
                    "Select X-axis Column",
                    numeric_cols,
                    key="scatter_col1_3d"
                )
            
            with col2:
                selected_col2 = st.selectbox(
                    "Select Y-axis Column",
                    numeric_cols,
                    key="scatter_col2_3d"
                )
            
            with col3:
                selected_col3 = st.selectbox(
                    "Select Z-axis Column",
                    numeric_cols,
                    key="scatter_col3_3d"
                )
        
        # Validate selection
        if plot_type == "2D Scatter Plot":
            is_valid, error_msg = self._validate_columns_for_plotting(df, selected_col1, selected_col2)
        else:
            is_valid, error_msg = self._validate_columns_for_3d_plotting(df, selected_col1, selected_col2, selected_col3)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return
        
        # Calculate statistics (for 2D only, 3D will show basic stats)
        if plot_type == "2D Scatter Plot":
            stats = self._calculate_correlation_statistics(df, selected_col1, selected_col2)
            
            if not stats:
                st.error("❌ Unable to calculate statistics.")
                return
            
            # Display statistics
            self._display_statistics_summary(stats)
            
            # Display detailed statistics
            self._display_detailed_statistics(stats, selected_col1, selected_col2)
            
            # Create and display scatter plot with performance indicator
            with st.spinner(f"Generating 2D scatter plot for {len(df)} data points..."):
                self._create_scatter_plot(df, selected_col1, selected_col2, stats)
            
            # Display interpretation
            self._display_correlation_interpretation(stats, selected_col1, selected_col2)
        else:
            # 3D Plot
            stats = self._calculate_3d_statistics(df, selected_col1, selected_col2, selected_col3)
            
            if not stats:
                st.error("❌ Unable to calculate statistics.")
                return
            
            # Display 3D statistics
            self._display_3d_statistics_summary(stats, selected_col1, selected_col2, selected_col3)
            
            # Create and display 3D scatter plot
            with st.spinner(f"Generating 3D scatter plot for {len(df)} data points..."):
                self._create_3d_scatter_plot(df, selected_col1, selected_col2, selected_col3, stats)
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric columns from DataFrame."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    
    def _normalize_to_percentiles(self, data: pd.Series) -> np.ndarray:
        """
        Normalize numeric data to percentiles for density-aware coloring.
        
        This method converts values to their percentile ranks, so that colors
        are distributed based on data density rather than absolute values.
        
        Args:
            data: Numeric data to normalize
            
        Returns:
            Array of percentile values (0-1) representing data density
            
        Example:
            If 99% of data is between 0-1 and 1% is between 1-1000,
            the 0-1 range will use 99% of the color spectrum.
        """
        # Use rankdata to compute percentile ranks
        # method='average' handles ties by averaging their ranks
        ranks = rankdata(data, method='average')
        
        # Normalize ranks to 0-1 range (percentiles)
        percentiles = (ranks - 1) / (len(data) - 1) if len(data) > 1 else np.array([0.5])
        
        return percentiles
    
    def _validate_column_for_histogram(self, df: pd.DataFrame, col: str) -> Tuple[bool, str]:
        """Validate that selected column is suitable for histogram plotting."""
        if col not in df.columns:
            return False, "Selected column does not exist in the DataFrame."
        
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, "Column must be numeric for histogram plotting."
        
        # Check for sufficient data
        valid_data = df[col].dropna()
        if len(valid_data) < 1:
            return False, "Insufficient data for plotting. Need at least 1 valid data point."
        
        return True, "Column is valid for histogram plotting."
    
    def _validate_columns_for_plotting(self, df: pd.DataFrame, col1: str, col2: str) -> Tuple[bool, str]:
        """Validate that selected columns are suitable for scatter plotting."""
        if col1 not in df.columns or col2 not in df.columns:
            return False, "One or both selected columns do not exist in the DataFrame."
        
        if col1 == col2:
            return False, "Please select two different columns for scatter plotting."
        
        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
            return False, "Both columns must be numeric for scatter plotting."
        
        # Check for sufficient data
        valid_data = df[[col1, col2]].dropna()
        if len(valid_data) < 2:
            return False, "Insufficient data for plotting. Need at least 2 valid data points."
        
        return True, "Columns are valid for plotting."
    
    def _calculate_correlation_statistics(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, float]:
        """Calculate correlation and covariance between two columns."""
        valid_data = df[[col1, col2]].dropna()
        
        if len(valid_data) < 2:
            return {}
        
        # Calculate correlation
        correlation = valid_data[col1].corr(valid_data[col2])
        
        # Calculate covariance
        covariance = valid_data[col1].cov(valid_data[col2])
        
        # Calculate additional statistics
        stats = {
            'correlation': correlation,
            'covariance': covariance,
            'pearson_r': correlation,
            'spearman_r': valid_data[col1].corr(valid_data[col2], method='spearman'),
            'kendall_tau': valid_data[col1].corr(valid_data[col2], method='kendall'),
            'n_points': len(valid_data),
            'col1_mean': valid_data[col1].mean(),
            'col1_std': valid_data[col1].std(),
            'col2_mean': valid_data[col2].mean(),
            'col2_std': valid_data[col2].std(),
            'col1_min': valid_data[col1].min(),
            'col1_max': valid_data[col1].max(),
            'col2_min': valid_data[col2].min(),
            'col2_max': valid_data[col2].max()
        }
        
        return stats
    
    def _calculate_histogram_statistics(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Calculate statistics for histogram."""
        valid_data = df[col].dropna()
        
        if len(valid_data) < 1:
            return {}
        
        # Calculate comprehensive statistics
        stats = {
            'n_points': len(valid_data),
            'mean': valid_data.mean(),
            'median': valid_data.median(),
            'std': valid_data.std(),
            'min': valid_data.min(),
            'max': valid_data.max(),
            'q25': valid_data.quantile(0.25),
            'q75': valid_data.quantile(0.75),
            'iqr': valid_data.quantile(0.75) - valid_data.quantile(0.25),
            'skewness': valid_data.skew(),
            'kurtosis': valid_data.kurtosis(),
            'range': valid_data.max() - valid_data.min(),
        }
        
        return stats
    
    def _display_statistics_summary(self, stats: Dict[str, float]) -> None:
        """Display summary statistics in metrics."""
        st.subheader("📈 Statistical Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Correlation (r)", f"{stats['correlation']:.3f}")
        
        with col2:
            st.metric("Covariance", f"{stats['covariance']:.3f}")
        
        with col3:
            st.metric("Data Points", stats['n_points'])
        
        with col4:
            st.metric("Spearman's ρ", f"{stats['spearman_r']:.3f}")
    
    def _display_histogram_statistics(self, stats: Dict[str, Any], col: str) -> None:
        """Display statistics for histogram."""
        st.subheader("📈 Distribution Statistics")
        
        # Top row metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Points", f"{stats['n_points']:,}")
        
        with col2:
            st.metric("Mean", f"{stats['mean']:.3f}")
        
        with col3:
            st.metric("Median", f"{stats['median']:.3f}")
        
        with col4:
            st.metric("Std Dev", f"{stats['std']:.3f}")
        
        # Second row metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min", f"{stats['min']:.3f}")
        
        with col2:
            st.metric("Max", f"{stats['max']:.3f}")
        
        with col3:
            st.metric("Range", f"{stats['range']:.3f}")
        
        with col4:
            st.metric("IQR", f"{stats['iqr']:.3f}")
        
        # Distribution shape metrics
        st.markdown("**📊 Distribution Shape**")
        col1, col2 = st.columns(2)
        
        with col1:
            skew_value = stats['skewness']
            if abs(skew_value) < 0.5:
                skew_desc = "Approximately Symmetric"
            elif skew_value > 0:
                skew_desc = "Right-skewed (Positive)"
            else:
                skew_desc = "Left-skewed (Negative)"
            st.write(f"**Skewness**: {skew_value:.3f} - {skew_desc}")
        
        with col2:
            kurt_value = stats['kurtosis']
            if abs(kurt_value) < 0.5:
                kurt_desc = "Normal-like"
            elif kurt_value > 0:
                kurt_desc = "Heavy-tailed (Leptokurtic)"
            else:
                kurt_desc = "Light-tailed (Platykurtic)"
            st.write(f"**Kurtosis**: {kurt_value:.3f} - {kurt_desc}")
        
        st.markdown("---")
    
    def _display_detailed_statistics(self, stats: Dict[str, float], col1: str, col2: str) -> None:
        """Display detailed statistics for both columns."""
        st.subheader("📊 Detailed Statistics")
        
        col1_stats, col2_stats = st.columns(2)
        
        with col1_stats:
            st.write(f"**{col1} Statistics:**")
            st.write(f"- Mean: {stats['col1_mean']:.3f}")
            st.write(f"- Std Dev: {stats['col1_std']:.3f}")
            st.write(f"- Range: {stats['col1_min']:.3f} to {stats['col1_max']:.3f}")
        
        with col2_stats:
            st.write(f"**{col2} Statistics:**")
            st.write(f"- Mean: {stats['col2_mean']:.3f}")
            st.write(f"- Std Dev: {stats['col2_std']:.3f}")
            st.write(f"- Range: {stats['col2_min']:.3f} to {stats['col2_max']:.3f}")
    
    def _create_scatter_plot(self, df: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float]) -> None:
        """Create and display scatter plot with trend line."""
        st.subheader("🎨 Scatter Plot")
        
        # Plot size controls
        st.markdown("**📏 Plot Size Controls**")
        
        # Size presets
        size_preset = st.selectbox(
            "Quick Size Preset",
            ["Custom", "Small (6x4)", "Medium (8x6)", "Large (12x8)", "Extra Large (16x10)", "Browser Friendly (8x5)"],
            key="size_preset"
        )
        
        # Apply preset sizes
        if size_preset == "Small (6x4)":
            plot_width, plot_height = 6, 4
        elif size_preset == "Medium (8x6)":
            plot_width, plot_height = 8, 6
        elif size_preset == "Large (12x8)":
            plot_width, plot_height = 12, 8
        elif size_preset == "Extra Large (16x10)":
            plot_width, plot_height = 16, 10
        elif size_preset == "Browser Friendly (8x5)":
            plot_width, plot_height = 8, 5
        else:
            # Custom size controls
            col1_size, col2_size = st.columns(2)
            
            with col1_size:
                plot_width = st.slider(
                    "Plot Width (inches)", 
                    min_value=4, 
                    max_value=20, 
                    value=10, 
                    step=1,
                    key="plot_width"
                )
            
            with col2_size:
                plot_height = st.slider(
                    "Plot Height (inches)", 
                    min_value=3, 
                    max_value=15, 
                    value=6, 
                    step=1,
                    key="plot_height"
                )
        
        # Display current size
        st.info(f"📐 Current plot size: {plot_width} × {plot_height} inches")
        
        # Heat column selection for Heat Map Scatter
        col_heat, col_flow = st.columns(2)
        
        with col_heat:
            # Get all columns for heat mapping (both numeric and categorical)
            all_cols = df.columns.tolist()
            heat_column = st.selectbox(
                "Select Heat Column (for coloring)",
                ["None"] + all_cols,
                key="heat_column_selector"
            )
            if heat_column == "None":
                heat_column = None
        
        with col_flow:
            # Get all columns for flow name display
            all_cols = df.columns.tolist()
            flow_name_column = st.selectbox(
                "Select Flow Name Column",
                ["None"] + all_cols,
                key="flow_name_selector"
            )
            if flow_name_column == "None":
                flow_name_column = None
        
        # Create heat map scatter plot
        self._create_heat_map_scatter_plot(df, col1, col2, stats, plot_width, plot_height, heat_column, flow_name_column)
        
        # Add plot options
        self._add_plot_options(df, col1, col2, stats, plot_width, plot_height, heat_column, flow_name_column)
    
    
    def _create_heat_map_scatter_plot(self, df: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], 
                                    width: int, height: int, heat_column: str = None, flow_name_column: str = None) -> None:
        """Create heat map scatter plot with colored dots and flow names."""
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="husl")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Get valid data
        columns_needed = [col1, col2]
        if heat_column:
            columns_needed.append(heat_column)
        if flow_name_column:
            columns_needed.append(flow_name_column)
        
        valid_data = df[columns_needed].dropna()
        
        if len(valid_data) == 0:
            st.error("❌ No valid data points found for the selected columns.")
            return
        
        # Create scatter plot
        if heat_column and heat_column in valid_data.columns:
            # Extract heat data from the already filtered valid_data
            heat_data = valid_data[heat_column]
            
            # Ensure heat_data is a Series
            if isinstance(heat_data, pd.DataFrame):
                heat_data = heat_data.iloc[:, 0]  # Take first column if it's a DataFrame
            
            # Double-check that all arrays have the same length
            x_data = valid_data[col1]
            y_data = valid_data[col2]
            
            if len(heat_data) != len(x_data) or len(heat_data) != len(y_data):
                st.error(f"❌ Data length mismatch: x={len(x_data)}, y={len(y_data)}, heat={len(heat_data)}")
                return
            
            # Check if heat column is numeric or categorical
            if pd.api.types.is_numeric_dtype(heat_data):
                # Numeric coloring with percentile normalization (density-aware)
                # Convert values to percentiles so colors reflect data density
                percentile_values = self._normalize_to_percentiles(heat_data)
                
                scatter = ax.scatter(
                    x_data, 
                    y_data, 
                    c=percentile_values,  # Use percentiles instead of raw values
                    cmap='viridis', 
                    alpha=0.7, 
                    s=80,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                # Add colorbar with percentile-based labeling
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(f'{heat_column} (Percentile-based)', fontsize=10, fontweight='bold')
                
                # Add annotation about normalization
                min_val = heat_data.min()
                max_val = heat_data.max()
                median_val = heat_data.median()
                ax.text(0.02, 0.02, 
                       f'Value range: [{min_val:.3g}, {max_val:.3g}]\nMedian: {median_val:.3g}\nColors: density-based',
                       transform=ax.transAxes, 
                       verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                       fontsize=8)
            else:
                # Categorical coloring
                unique_categories = heat_data.unique()
                n_categories = len(unique_categories)
                
                # Use a colormap that works well for categorical data
                if n_categories <= 10:
                    colors = plt.cm.Set3(np.linspace(0, 1, n_categories))
                else:
                    colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
                
                # Create color mapping
                color_map = dict(zip(unique_categories, colors))
                
                # Plot each category separately
                for category in unique_categories:
                    mask = heat_data == category
                    category_data = valid_data[mask]
                    
                    ax.scatter(
                        x_data[mask], 
                        y_data[mask], 
                        c=[color_map[category]], 
                        label=str(category),
                        alpha=0.7, 
                        s=80,
                        edgecolors='black',
                        linewidth=0.5
                    )
                
                # Add legend for categories
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            # Default color
            scatter = ax.scatter(
                valid_data[col1], 
                valid_data[col2], 
                alpha=0.7, 
                s=80,
                color='steelblue',
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add flow names under dots if specified
        if flow_name_column and flow_name_column in valid_data.columns:
            for idx, row in valid_data.iterrows():
                x_pos = row[col1]
                y_pos = row[col2]
                flow_name = str(row[flow_name_column])
                
                # Add text below the point
                ax.text(
                    x_pos, 
                    y_pos - (valid_data[col2].max() - valid_data[col2].min()) * 0.05, 
                    flow_name, 
                    fontsize=3, 
                    ha='center', 
                    va='top',
                    rotation=0,
                    alpha=0.7
                )
        
        # Add trend line
        if len(valid_data) > 1:
            z = np.polyfit(valid_data[col1], valid_data[col2], 1)
            p = np.poly1d(z)
            ax.plot(valid_data[col1], p(valid_data[col1]), "r--", alpha=0.8, linewidth=2, label='Trend Line')
            ax.legend()
        
        # Customize plot
        ax.set_xlabel(col1, fontsize=12, fontweight='bold')
        ax.set_ylabel(col2, fontsize=12, fontweight='bold')
        
        title = f"Heat Map Scatter: {col1} vs {col2}"
        if heat_column:
            title += f" (colored by {heat_column})"
        if flow_name_column:
            title += f" (labels: {flow_name_column})"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"n = {len(valid_data)}\n"
        stats_text += f"r = {stats['correlation']:.3f}\n"
        stats_text += f"Cov = {stats['covariance']:.3f}"
        
        if heat_column:
            stats_text += f"\nHeat: {heat_column}"
        if flow_name_column:
            stats_text += f"\nLabels: {flow_name_column}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def _add_plot_options(self, df: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], width: int, height: int, heat_column: str = None, flow_name_column: str = None) -> None:
        """Add plot options and export functionality."""
        st.markdown("**Plot Options:**")
        col1_opt, col2_opt = st.columns(2)
        
        with col1_opt:
            if st.button("📥 Download Current Plot as PNG"):
                # Create heat map scatter plot for download
                fig = self._create_downloadable_heat_map_scatter_plot(df, col1, col2, stats, width, height, heat_column, flow_name_column)
                
                # Save to buffer
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                # Create download button
                plot_name = f"heat_map_scatter_{col1}_vs_{col2}"
                if heat_column:
                    plot_name += f"_colored_by_{heat_column}"
                if flow_name_column:
                    plot_name += f"_labeled_by_{flow_name_column}"
                
                st.download_button(
                    label="Click to download",
                    data=buf.getvalue(),
                    file_name=f"{plot_name}.png",
                    mime="image/png"
                )
                plt.close(fig)
        
        with col2_opt:
            # Export statistics to CSV
            stats_df = pd.DataFrame([stats])
            csv = stats_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Statistics CSV",
                data=csv,
                file_name=f"statistics_{col1}_vs_{col2}.csv",
                mime="text/csv"
            )
    
    
    def _create_downloadable_heat_map_scatter_plot(self, df: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], 
                                                 width: int, height: int, heat_column: str = None, flow_name_column: str = None) -> plt.Figure:
        """Create downloadable heat map scatter plot."""
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="husl")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Get valid data
        columns_needed = [col1, col2]
        if heat_column:
            columns_needed.append(heat_column)
        if flow_name_column:
            columns_needed.append(flow_name_column)
        
        valid_data = df[columns_needed].dropna()
        
        if len(valid_data) == 0:
            return fig
        
        # Create scatter plot
        if heat_column and heat_column in valid_data.columns:
            # Extract heat data from the already filtered valid_data
            heat_data = valid_data[heat_column]
            
            # Ensure heat_data is a Series
            if isinstance(heat_data, pd.DataFrame):
                heat_data = heat_data.iloc[:, 0]  # Take first column if it's a DataFrame
            
            # Double-check that all arrays have the same length
            x_data = valid_data[col1]
            y_data = valid_data[col2]
            
            if len(heat_data) != len(x_data) or len(heat_data) != len(y_data):
                # Return empty figure if there's a mismatch
                return fig
            
            # Check if heat column is numeric or categorical
            if pd.api.types.is_numeric_dtype(heat_data):
                # Numeric coloring with percentile normalization (density-aware)
                # Convert values to percentiles so colors reflect data density
                percentile_values = self._normalize_to_percentiles(heat_data)
                
                scatter = ax.scatter(
                    x_data, 
                    y_data, 
                    c=percentile_values,  # Use percentiles instead of raw values
                    cmap='viridis', 
                    alpha=0.7, 
                    s=80,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                # Add colorbar with percentile-based labeling
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(f'{heat_column} (Percentile-based)', fontsize=10, fontweight='bold')
                
                # Add annotation about normalization
                min_val = heat_data.min()
                max_val = heat_data.max()
                median_val = heat_data.median()
                ax.text(0.02, 0.02, 
                       f'Value range: [{min_val:.3g}, {max_val:.3g}]\nMedian: {median_val:.3g}\nColors: density-based',
                       transform=ax.transAxes, 
                       verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                       fontsize=8)
            else:
                # Categorical coloring
                unique_categories = heat_data.unique()
                n_categories = len(unique_categories)
                
                # Use a colormap that works well for categorical data
                if n_categories <= 10:
                    colors = plt.cm.Set3(np.linspace(0, 1, n_categories))
                else:
                    colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
                
                # Create color mapping
                color_map = dict(zip(unique_categories, colors))
                
                # Plot each category separately
                for category in unique_categories:
                    mask = heat_data == category
                    category_data = valid_data[mask]
                    
                    ax.scatter(
                        x_data[mask], 
                        y_data[mask], 
                        c=[color_map[category]], 
                        label=str(category),
                        alpha=0.7, 
                        s=80,
                        edgecolors='black',
                        linewidth=0.5
                    )
                
                # Add legend for categories
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            # Default color
            scatter = ax.scatter(
                valid_data[col1], 
                valid_data[col2], 
                alpha=0.7, 
                s=80,
                color='steelblue',
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add flow names under dots if specified
        if flow_name_column and flow_name_column in valid_data.columns:
            for idx, row in valid_data.iterrows():
                x_pos = row[col1]
                y_pos = row[col2]
                flow_name = str(row[flow_name_column])
                
                # Add text below the point
                ax.text(
                    x_pos, 
                    y_pos - (valid_data[col2].max() - valid_data[col2].min()) * 0.05, 
                    flow_name, 
                    fontsize=3, 
                    ha='center', 
                    va='top',
                    rotation=0,
                    alpha=0.7
                )
        
        # Add trend line
        if len(valid_data) > 1:
            z = np.polyfit(valid_data[col1], valid_data[col2], 1)
            p = np.poly1d(z)
            ax.plot(valid_data[col1], p(valid_data[col1]), "r--", alpha=0.8, linewidth=2, label='Trend Line')
            ax.legend()
        
        # Customize plot
        ax.set_xlabel(col1, fontsize=12, fontweight='bold')
        ax.set_ylabel(col2, fontsize=12, fontweight='bold')
        
        title = f"Heat Map Scatter: {col1} vs {col2}"
        if heat_column:
            title += f" (colored by {heat_column})"
        if flow_name_column:
            title += f" (labels: {flow_name_column})"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"n = {len(valid_data)}\n"
        stats_text += f"r = {stats['correlation']:.3f}\n"
        stats_text += f"Cov = {stats['covariance']:.3f}"
        
        if heat_column:
            stats_text += f"\nHeat: {heat_column}"
        if flow_name_column:
            stats_text += f"\nLabels: {flow_name_column}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, col: str, stats: Dict[str, Any]) -> None:
        """Create and display histogram with customization options."""
        st.subheader("📊 Histogram")
        
        # Get valid data
        valid_data = df[col].dropna()
        
        if len(valid_data) == 0:
            st.error("❌ No valid data points found.")
            return
        
        # Histogram customization options
        st.markdown("**🎨 Histogram Options**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Bin selection
            bin_method = st.selectbox(
                "Bin Method",
                ["Auto (Sturges)", "Fixed Count", "Fixed Width", "sqrt", "doane", "scott", "rice"],
                key="hist_bin_method"
            )
        
        with col2:
            # Color selection
            hist_color = st.color_picker("Histogram Color", "#1f77b4", key="hist_color")
        
        with col3:
            # Overlay options
            show_kde = st.checkbox("Show KDE (Density)", value=True, key="show_kde")
            show_normal = st.checkbox("Show Normal Curve", value=False, key="show_normal")
        
        # Additional customization
        col1, col2 = st.columns(2)
        
        with col1:
            show_stats_lines = st.checkbox("Show Mean/Median Lines", value=True, key="show_stats_lines")
        
        with col2:
            show_quartiles = st.checkbox("Show Quartile Lines", value=False, key="show_quartiles")
        
        # Determine bins
        if bin_method == "Fixed Count":
            num_bins = st.slider("Number of Bins", min_value=5, max_value=100, value=30, key="num_bins")
            bins = num_bins
        elif bin_method == "Fixed Width":
            bin_width = st.number_input(
                "Bin Width", 
                min_value=0.001, 
                max_value=float(stats['range']), 
                value=float(stats['range'] / 30),
                format="%.6g",
                key="bin_width"
            )
            bins = np.arange(stats['min'], stats['max'] + bin_width, bin_width)
        elif bin_method == "Auto (Sturges)":
            bins = 'sturges'
        else:
            bins = bin_method.lower()
        
        # Plot size controls
        size_col1, size_col2 = st.columns(2)
        
        with size_col1:
            plot_width = st.slider("Plot Width (inches)", min_value=6, max_value=16, value=10, key="hist_width")
        
        with size_col2:
            plot_height = st.slider("Plot Height (inches)", min_value=4, max_value=12, value=6, key="hist_height")
        
        st.info(f"📐 Current plot size: {plot_width} × {plot_height} inches")
        
        # Create the histogram plot
        self._render_histogram_plot(
            valid_data, col, stats, bins, hist_color, 
            show_kde, show_normal, show_stats_lines, show_quartiles,
            plot_width, plot_height
        )
        
        # Download options
        self._add_histogram_download_options(
            valid_data, col, stats, bins, hist_color,
            show_kde, show_normal, show_stats_lines, show_quartiles,
            plot_width, plot_height
        )
    
    def _render_histogram_plot(self, data: pd.Series, col: str, stats: Dict[str, Any],
                               bins: Any, color: str, show_kde: bool, show_normal: bool,
                               show_stats_lines: bool, show_quartiles: bool,
                               width: int, height: int) -> None:
        """Render the histogram plot with all options."""
        sns.set_theme(style="whitegrid")
        
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Plot histogram
        n, bin_edges, patches = ax.hist(
            data, 
            bins=bins, 
            color=color, 
            alpha=0.7,
            edgecolor='black',
            linewidth=1.2,
            density=show_kde or show_normal  # Normalize if showing density curves
        )
        
        # Update patch colors
        for patch in patches:
            patch.set_facecolor(color)
        
        # Add KDE curve
        if show_kde:
            from scipy import stats as scipy_stats
            kde = scipy_stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE', alpha=0.8)
        
        # Add normal distribution curve
        if show_normal:
            from scipy import stats as scipy_stats
            x_range = np.linspace(data.min(), data.max(), 200)
            normal_curve = scipy_stats.norm.pdf(x_range, stats['mean'], stats['std'])
            ax.plot(x_range, normal_curve, 'g--', linewidth=2, label='Normal Dist.', alpha=0.8)
        
        # Add statistical lines
        if show_stats_lines:
            ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.3f}", alpha=0.8)
            ax.axvline(stats['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.3f}", alpha=0.8)
        
        # Add quartile lines
        if show_quartiles:
            ax.axvline(stats['q25'], color='purple', linestyle=':', linewidth=1.5, label=f"Q1: {stats['q25']:.3f}", alpha=0.7)
            ax.axvline(stats['q75'], color='purple', linestyle=':', linewidth=1.5, label=f"Q3: {stats['q75']:.3f}", alpha=0.7)
        
        # Customize plot
        ax.set_xlabel(col, fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency' if not (show_kde or show_normal) else 'Density', fontsize=12, fontweight='bold')
        ax.set_title(f'Histogram: {col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if show_kde or show_normal or show_stats_lines or show_quartiles:
            ax.legend(loc='best', fontsize=10)
        
        # Add statistics text box
        stats_text = f"n = {stats['n_points']:,}\n"
        stats_text += f"μ = {stats['mean']:.3f}\n"
        stats_text += f"σ = {stats['std']:.3f}\n"
        stats_text += f"Range = {stats['range']:.3f}"
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
               fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def _add_histogram_download_options(self, data: pd.Series, col: str, stats: Dict[str, Any],
                                        bins: Any, color: str, show_kde: bool, show_normal: bool,
                                        show_stats_lines: bool, show_quartiles: bool,
                                        width: int, height: int) -> None:
        """Add download options for histogram."""
        st.markdown("**📥 Download Options**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Histogram as PNG"):
                # Create the plot
                sns.set_theme(style="whitegrid")
                fig, ax = plt.subplots(figsize=(width, height))
                
                n, bin_edges, patches = ax.hist(
                    data, bins=bins, color=color, alpha=0.7,
                    edgecolor='black', linewidth=1.2,
                    density=show_kde or show_normal
                )
                
                for patch in patches:
                    patch.set_facecolor(color)
                
                if show_kde:
                    from scipy import stats as scipy_stats
                    kde = scipy_stats.gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 200)
                    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE', alpha=0.8)
                
                if show_normal:
                    from scipy import stats as scipy_stats
                    x_range = np.linspace(data.min(), data.max(), 200)
                    normal_curve = scipy_stats.norm.pdf(x_range, stats['mean'], stats['std'])
                    ax.plot(x_range, normal_curve, 'g--', linewidth=2, label='Normal Dist.', alpha=0.8)
                
                if show_stats_lines:
                    ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.3f}", alpha=0.8)
                    ax.axvline(stats['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.3f}", alpha=0.8)
                
                if show_quartiles:
                    ax.axvline(stats['q25'], color='purple', linestyle=':', linewidth=1.5, label=f"Q1: {stats['q25']:.3f}", alpha=0.7)
                    ax.axvline(stats['q75'], color='purple', linestyle=':', linewidth=1.5, label=f"Q3: {stats['q75']:.3f}", alpha=0.7)
                
                ax.set_xlabel(col, fontsize=12, fontweight='bold')
                ax.set_ylabel('Frequency' if not (show_kde or show_normal) else 'Density', fontsize=12, fontweight='bold')
                ax.set_title(f'Histogram: {col}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                if show_kde or show_normal or show_stats_lines or show_quartiles:
                    ax.legend(loc='best', fontsize=10)
                
                stats_text = f"n = {stats['n_points']:,}\nμ = {stats['mean']:.3f}\nσ = {stats['std']:.3f}\nRange = {stats['range']:.3f}"
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                       fontsize=10)
                
                plt.tight_layout()
                
                # Save to buffer
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="Click to download PNG",
                    data=buf.getvalue(),
                    file_name=f"histogram_{col}.png",
                    mime="image/png"
                )
                plt.close(fig)
        
        with col2:
            # Export statistics to CSV
            stats_df = pd.DataFrame([stats])
            csv = stats_df.to_csv(index=False)
            st.download_button(
                label="📈 Download Statistics CSV",
                data=csv,
                file_name=f"histogram_statistics_{col}.csv",
                mime="text/csv"
            )
    
    def _create_cdf(self, df: pd.DataFrame, col: str, stats: Dict[str, Any]) -> None:
        """Create and display empirical CDF with customization options."""
        st.subheader("📊 Empirical CDF")
        
        valid_data = df[col].dropna()
        if len(valid_data) == 0:
            st.error("❌ No valid data points found.")
            return
        
        # CDF customization options
        st.markdown("**🎨 CDF Options**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            line_color = st.color_picker("Line Color", "#1f77b4", key="cdf_color")
        with col2:
            show_markers = st.checkbox("Show Markers", value=False, key="cdf_markers")
        with col3:
            show_percentiles = st.checkbox("Show 25/50/75% Lines", value=True, key="cdf_percentiles")
        
        # Optional max cap
        st.markdown("**🔧 Value Capping**")
        cap_col1, cap_col2 = st.columns(2)
        with cap_col1:
            enable_cap = st.checkbox("Enable max value cap", value=False, key="cdf_enable_cap")
        with cap_col2:
            cap_value = st.number_input(
                "Max value (values > max will be set to max)",
                min_value=float(valid_data.min()) if len(valid_data) else 0.0,
                value=float(min(200.0, float(valid_data.max()))) if len(valid_data) else 200.0,
                step=1.0,
                key="cdf_cap_value"
            )
        
        processed_data = valid_data.clip(upper=cap_value) if enable_cap else valid_data
        if enable_cap:
            st.info(f"Values greater than {cap_value:.6g} are set to {cap_value:.6g} for this CDF.")

        # Plot size controls
        size_col1, size_col2 = st.columns(2)
        with size_col1:
            plot_width = st.slider("Plot Width (inches)", min_value=6, max_value=16, value=10, key="cdf_width")
        with size_col2:
            plot_height = st.slider("Plot Height (inches)", min_value=4, max_value=12, value=6, key="cdf_height")
        
        st.info(f"📐 Current plot size: {plot_width} × {plot_height} inches")
        
        # Render CDF
        self._render_cdf_plot(
            processed_data, col, stats, line_color, show_markers, show_percentiles, plot_width, plot_height
        )
        
        # Download options
        self._add_cdf_download_options(
            processed_data, col, stats, line_color, show_markers, show_percentiles, plot_width, plot_height
        )
    
    def _render_cdf_plot(self, data: pd.Series, col: str, stats: Dict[str, Any],
                         line_color: str, show_markers: bool, show_percentiles: bool,
                         width: int, height: int) -> None:
        """Render the empirical CDF plot."""
        sns.set_theme(style="whitegrid")
        
        sorted_vals = np.sort(data.values)
        n = len(sorted_vals)
        y = np.arange(1, n + 1) / n
        
        fig, ax = plt.subplots(figsize=(width, height))
        
        ax.plot(
            sorted_vals,
            y,
            color=line_color,
            linewidth=2,
            marker='o' if show_markers else None,
            markersize=3 if show_markers else 0,
            label='Empirical CDF'
        )
        
        if show_percentiles:
            q25 = stats['q25']
            q50 = stats['median']
            q75 = stats['q75']
            for q_val, q_label in [(q25, '25%'), (q50, '50%'), (q75, '75%')]:
                ax.axvline(q_val, color='purple', linestyle=':', linewidth=1.5, alpha=0.7, label=f"{q_label} = {q_val:.3f}")
        
        ax.set_xlabel(col, fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'Empirical CDF: {col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.legend(loc='best', fontsize=10)
        
        stats_text = f"n = {stats['n_points']:,}\nμ = {stats['mean']:.3f}\nσ = {stats['std']:.3f}\nRange = {stats['range']:.3f}"
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
               fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def _add_cdf_download_options(self, data: pd.Series, col: str, stats: Dict[str, Any],
                                   line_color: str, show_markers: bool, show_percentiles: bool,
                                   width: int, height: int) -> None:
        """Add download options for CDF plot."""
        st.markdown("**📥 Download Options**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Download CDF as PNG"):
                sns.set_theme(style="whitegrid")
                
                sorted_vals = np.sort(data.values)
                n = len(sorted_vals)
                y = np.arange(1, n + 1) / n
                
                fig, ax = plt.subplots(figsize=(width, height))
                ax.plot(
                    sorted_vals,
                    y,
                    color=line_color,
                    linewidth=2,
                    marker='o' if show_markers else None,
                    markersize=3 if show_markers else 0,
                    label='Empirical CDF'
                )
                if show_percentiles:
                    q25 = stats['q25']
                    q50 = stats['median']
                    q75 = stats['q75']
                    for q_val, q_label in [(q25, '25%'), (q50, '50%'), (q75, '75%')]:
                        ax.axvline(q_val, color='purple', linestyle=':', linewidth=1.5, alpha=0.7, label=f"{q_label} = {q_val:.3f}")
                ax.set_xlabel(col, fontsize=12, fontweight='bold')
                ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
                ax.set_title(f'Empirical CDF: {col}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=10)
                
                stats_text = f"n = {stats['n_points']:,}\nμ = {stats['mean']:.3f}\nσ = {stats['std']:.3f}\nRange = {stats['range']:.3f}"
                ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                       fontsize=10)
                
                plt.tight_layout()
                
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="Click to download PNG",
                    data=buf.getvalue(),
                    file_name=f"cdf_{col}.png",
                    mime="image/png"
                )
                plt.close(fig)
        
        with col2:
            stats_df = pd.DataFrame([stats])
            csv = stats_df.to_csv(index=False)
            st.download_button(
                label="📈 Download Statistics CSV",
                data=csv,
                file_name=f"cdf_statistics_{col}.csv",
                mime="text/csv"
            )
    
    def _display_correlation_interpretation(self, stats: Dict[str, float], col1: str, col2: str) -> None:
        """Display correlation interpretation and insights."""
        st.subheader("🔍 Correlation Interpretation")
        
        correlation_strength = abs(stats['correlation'])
        if correlation_strength >= 0.8:
            strength = "very strong"
        elif correlation_strength >= 0.6:
            strength = "strong"
        elif correlation_strength >= 0.4:
            strength = "moderate"
        elif correlation_strength >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        direction = "positive" if stats['correlation'] > 0 else "negative"
        
        st.info(f"""
        **Correlation Analysis:**
        - The correlation coefficient (r = {stats['correlation']:.3f}) indicates a {strength} {direction} relationship between {col1} and {col2}.
        - A correlation of {stats['correlation']:.3f} means that as {col1} increases, {col2} tends to {'increase' if stats['correlation'] > 0 else 'decrease'}.
        - The covariance of {stats['covariance']:.3f} indicates the direction and magnitude of the linear relationship.
        """)
        
        # Additional insights
        if abs(stats['correlation']) > 0.7:
            st.success("💡 **Strong Correlation Detected:** This suggests a meaningful relationship between the variables.")
        elif abs(stats['correlation']) > 0.3:
            st.warning("⚠️ **Moderate Correlation:** There is some relationship, but it may not be strong enough for reliable predictions.")
        else:
            st.info("ℹ️ **Weak Correlation:** Little to no linear relationship detected between these variables.")
    
    def _validate_columns_for_3d_plotting(self, df: pd.DataFrame, col1: str, col2: str, col3: str) -> Tuple[bool, str]:
        """Validate that selected columns are suitable for 3D scatter plotting."""
        if col1 not in df.columns or col2 not in df.columns or col3 not in df.columns:
            return False, "One or more selected columns do not exist in the DataFrame."
        
        if col1 == col2 or col1 == col3 or col2 == col3:
            return False, "Please select three different columns for 3D scatter plotting."
        
        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]) or not pd.api.types.is_numeric_dtype(df[col3]):
            return False, "All three columns must be numeric for 3D scatter plotting."
        
        # Check for sufficient data
        valid_data = df[[col1, col2, col3]].dropna()
        if len(valid_data) < 2:
            return False, "Insufficient data for plotting. Need at least 2 valid data points."
        
        return True, "Columns are valid for 3D plotting."
    
    def _calculate_3d_statistics(self, df: pd.DataFrame, col1: str, col2: str, col3: str) -> Dict[str, float]:
        """Calculate statistics for three columns."""
        valid_data = df[[col1, col2, col3]].dropna()
        
        if len(valid_data) < 2:
            return {}
        
        stats = {
            'n_points': len(valid_data),
            'col1_mean': valid_data[col1].mean(),
            'col1_std': valid_data[col1].std(),
            'col1_min': valid_data[col1].min(),
            'col1_max': valid_data[col1].max(),
            'col2_mean': valid_data[col2].mean(),
            'col2_std': valid_data[col2].std(),
            'col2_min': valid_data[col2].min(),
            'col2_max': valid_data[col2].max(),
            'col3_mean': valid_data[col3].mean(),
            'col3_std': valid_data[col3].std(),
            'col3_min': valid_data[col3].min(),
            'col3_max': valid_data[col3].max(),
        }
        
        return stats
    
    def _display_3d_statistics_summary(self, stats: Dict[str, float], col1: str, col2: str, col3: str) -> None:
        """Display summary statistics for 3D data."""
        st.subheader("📈 Statistical Summary (3D)")
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.write(f"**{col1} (X-axis):**")
            st.write(f"- Mean: {stats['col1_mean']:.3f}")
            st.write(f"- Std Dev: {stats['col1_std']:.3f}")
            st.write(f"- Range: {stats['col1_min']:.3f} to {stats['col1_max']:.3f}")
        
        with col_stat2:
            st.write(f"**{col2} (Y-axis):**")
            st.write(f"- Mean: {stats['col2_mean']:.3f}")
            st.write(f"- Std Dev: {stats['col2_std']:.3f}")
            st.write(f"- Range: {stats['col2_min']:.3f} to {stats['col2_max']:.3f}")
        
        with col_stat3:
            st.write(f"**{col3} (Z-axis):**")
            st.write(f"- Mean: {stats['col3_mean']:.3f}")
            st.write(f"- Std Dev: {stats['col3_std']:.3f}")
            st.write(f"- Range: {stats['col3_min']:.3f} to {stats['col3_max']:.3f}")
        
        st.metric("Total Data Points", stats['n_points'])
    
    def _create_3d_scatter_plot(self, df: pd.DataFrame, col1: str, col2: str, col3: str, stats: Dict[str, float]) -> None:
        """Create and display interactive 3D scatter plot with Plotly."""
        st.subheader("🎨 3D Scatter Plot")
        
        # Heat column and flow name selection
        st.markdown("**🔥 Visualization Options**")
        col_heat, col_flow = st.columns(2)
        
        with col_heat:
            # Get all columns for heat mapping (both numeric and categorical)
            all_cols = df.columns.tolist()
            heat_column = st.selectbox(
                "Select Heat Column (for coloring)",
                ["None"] + all_cols,
                key="heat_column_selector_3d"
            )
            if heat_column == "None":
                heat_column = None
        
        with col_flow:
            # Get all columns for flow name display
            all_cols = df.columns.tolist()
            flow_name_column = st.selectbox(
                "Select Flow Name Column (for hover labels)",
                ["None"] + all_cols,
                key="flow_name_selector_3d"
            )
            if flow_name_column == "None":
                flow_name_column = None
        
        # Get valid data
        columns_needed = [col1, col2, col3]
        if heat_column:
            columns_needed.append(heat_column)
        if flow_name_column:
            columns_needed.append(flow_name_column)
        
        valid_data = df[columns_needed].dropna()
        
        if len(valid_data) == 0:
            st.error("❌ No valid data points found for the selected columns.")
            return
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Prepare hover text
        if flow_name_column:
            hover_text = valid_data[flow_name_column].astype(str)
        else:
            hover_text = None
        
        # Handle coloring
        if heat_column and heat_column in valid_data.columns:
            heat_data = valid_data[heat_column]
            
            if pd.api.types.is_numeric_dtype(heat_data):
                # Numeric coloring with percentile normalization (density-aware)
                # Convert values to percentiles so colors reflect data density
                percentile_values = self._normalize_to_percentiles(heat_data)
                
                # Create hover text with both original value and percentile
                if flow_name_column:
                    custom_hover = [
                        f"Value: {val:.3g} (Percentile: {perc*100:.1f}%)<br>{name}"
                        for val, perc, name in zip(heat_data, percentile_values, hover_text)
                    ]
                else:
                    custom_hover = [
                        f"Value: {val:.3g} (Percentile: {perc*100:.1f}%)"
                        for val, perc in zip(heat_data, percentile_values)
                    ]
                
                fig.add_trace(go.Scatter3d(
                    x=valid_data[col1],
                    y=valid_data[col2],
                    z=valid_data[col3],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=percentile_values,  # Use percentiles instead of raw values
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title=f'{heat_column}<br>(Percentile)',
                            tickformat='.0%'
                        ),
                        line=dict(width=0.5, color='DarkSlateGrey')
                    ),
                    text=custom_hover,
                    hovertemplate=f'<b>{col1}</b>: %{{x}}<br><b>{col2}</b>: %{{y}}<br><b>{col3}</b>: %{{z}}<br><b>{heat_column}</b>: %{{text}}<extra></extra>',
                    name='Data Points'
                ))
            else:
                # Categorical coloring - plot each category separately
                unique_categories = heat_data.unique()
                for category in unique_categories:
                    mask = heat_data == category
                    category_data = valid_data[mask]
                    
                    if flow_name_column:
                        category_hover = hover_text[mask]
                    else:
                        category_hover = None
                    
                    fig.add_trace(go.Scatter3d(
                        x=category_data[col1],
                        y=category_data[col2],
                        z=category_data[col3],
                        mode='markers',
                        marker=dict(
                            size=6,
                            line=dict(width=0.5, color='DarkSlateGrey')
                        ),
                        text=category_hover,
                        hovertemplate=f'<b>{col1}</b>: %{{x}}<br><b>{col2}</b>: %{{y}}<br><b>{col3}</b>: %{{z}}<br><b>{heat_column}</b>: {category}<br>%{{text}}<extra></extra>' if flow_name_column else f'<b>{col1}</b>: %{{x}}<br><b>{col2}</b>: %{{y}}<br><b>{col3}</b>: %{{z}}<br><b>{heat_column}</b>: {category}<extra></extra>',
                        name=str(category)
                    ))
        else:
            # Default coloring
            fig.add_trace(go.Scatter3d(
                x=valid_data[col1],
                y=valid_data[col2],
                z=valid_data[col3],
                mode='markers',
                marker=dict(
                    size=6,
                    color='steelblue',
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                text=hover_text,
                hovertemplate=f'<b>{col1}</b>: %{{x}}<br><b>{col2}</b>: %{{y}}<br><b>{col3}</b>: %{{z}}<br>%{{text}}<extra></extra>' if flow_name_column else f'<b>{col1}</b>: %{{x}}<br><b>{col2}</b>: %{{y}}<br><b>{col3}</b>: %{{z}}<extra></extra>',
                name='Data Points'
            ))
        
        # Update layout
        title = f"3D Scatter Plot: {col1} vs {col2} vs {col3}"
        if heat_column:
            title += f" (colored by {heat_column})"
        if flow_name_column:
            title += f" (labels: {flow_name_column})"
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=col1,
                yaxis_title=col2,
                zaxis_title=col3,
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray'),
                zaxis=dict(gridcolor='lightgray'),
            ),
            width=900,
            height=700,
            showlegend=True,
            hovermode='closest'
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Download options
        self._add_3d_plot_options(fig, col1, col2, col3, stats, heat_column, flow_name_column)
    
    def _add_3d_plot_options(self, fig: go.Figure, col1: str, col2: str, col3: str, 
                            stats: Dict[str, float], heat_column: str = None, 
                            flow_name_column: str = None) -> None:
        """Add download options for 3D plots."""
        st.markdown("**📥 Download Options:**")
        col1_opt, col2_opt = st.columns(2)
        
        with col1_opt:
            # Download as HTML
            import io
            buffer = io.StringIO()
            fig.write_html(buffer)
            html_bytes = buffer.getvalue().encode()
            
            plot_name = f"3d_scatter_{col1}_vs_{col2}_vs_{col3}"
            if heat_column:
                plot_name += f"_colored_by_{heat_column}"
            if flow_name_column:
                plot_name += f"_labeled_by_{flow_name_column}"
            
            st.download_button(
                label="📊 Download Interactive 3D Plot (HTML)",
                data=html_bytes,
                file_name=f"{plot_name}.html",
                mime="text/html"
            )
        
        with col2_opt:
            # Export statistics to CSV
            stats_df = pd.DataFrame([stats])
            csv = stats_df.to_csv(index=False)
            st.download_button(
                label="📈 Download Statistics CSV",
                data=csv,
                file_name=f"statistics_3d_{col1}_{col2}_{col3}.csv",
                mime="text/csv"
            )
        
        st.info("💡 **Tip:** The 3D plot is fully interactive! Use your mouse to rotate, zoom, and explore the data from different angles.")
    
