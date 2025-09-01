"""
Interactive table display module for Dataframe2Visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple
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
        st.markdown("### Data Table")
        
        # Display table information
        self._render_table_info(df, column_metadata)
        
        # Create interactive table
        self._create_interactive_table(df, column_metadata, processed_data)
        
        # Handle image clicks
        self._handle_image_interactions()
        
        # Add scatter plot analysis
        self.render_scatter_plot_analysis(df)
    
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
        # Create a custom HTML table for better image handling
        html_table = self._generate_html_table(df, column_metadata, processed_data)
        
        # Display the HTML table
        st.components.v1.html(html_table, height=AppConfig.DEFAULT_TABLE_HEIGHT, scrolling=True)
        
        # Alternative: Use Streamlit's dataframe with custom rendering
        # self._render_streamlit_dataframe(df, column_metadata, processed_data)
    
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
        
        # CSS styles
        html_parts.append("""
        <style>
        .dataframe-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        .dataframe-table th, .dataframe-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }
        .dataframe-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .dataframe-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .dataframe-table tr:hover {
            background-color: #f5f5f5;
        }
        .clickable-image {
            cursor: pointer;
            border: 2px solid transparent;
            transition: border-color 0.3s;
        }
        .clickable-image:hover {
            border-color: #007bff;
        }
        .image-cell {
            text-align: center;
            vertical-align: middle;
        }
        .text-cell {
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
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
        
        # Truncate long text
        text = str(cell_value)
        if len(text) > 50:
            text = text[:47] + "..."
        
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
        
        # Get numeric columns
        numeric_cols = self._get_numeric_columns(df)
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for scatter plotting.")
            st.info(f"Available numeric columns: {numeric_cols}")
            return
        
        # Column selection
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
        
        # Validate selection
        is_valid, error_msg = self._validate_columns_for_plotting(df, selected_col1, selected_col2)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return
        
        # Calculate statistics
        stats = self._calculate_correlation_statistics(df, selected_col1, selected_col2)
        
        if not stats:
            st.error("❌ Unable to calculate statistics.")
            return
        
        # Display statistics
        self._display_statistics_summary(stats)
        
        # Display detailed statistics
        self._display_detailed_statistics(stats, selected_col1, selected_col2)
        
        # Create and display scatter plot
        self._create_scatter_plot(df, selected_col1, selected_col2, stats)
        
        # Display interpretation
        self._display_correlation_interpretation(stats, selected_col1, selected_col2)
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric columns from DataFrame."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    
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
        
        # Plot type selection
        plot_type = st.selectbox(
            "Select Plot Type",
            ["Enhanced Seaborn", "Classic Matplotlib", "Joint Plot", "Regression Plot", "Heat Map Scatter"],
            key="plot_type_selector"
        )
        
        # Heat column selection (only for Heat Map Scatter)
        heat_column = None
        flow_name_column = None
        if plot_type == "Heat Map Scatter":
            col_heat, col_flow = st.columns(2)
            
            with col_heat:
                # Get all numeric columns for heat mapping
                all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                heat_column = st.selectbox(
                    "Select Heat Column (for coloring)",
                    ["None"] + all_numeric_cols,
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
        
        # Get valid data
        valid_data = df[[col1, col2]].dropna()
        
        if plot_type == "Enhanced Seaborn":
            self._create_enhanced_seaborn_plot(valid_data, col1, col2, stats, plot_width, plot_height)
        elif plot_type == "Classic Matplotlib":
            self._create_classic_matplotlib_plot(valid_data, col1, col2, stats, plot_width, plot_height)
        elif plot_type == "Joint Plot":
            self._create_joint_plot(valid_data, col1, col2, stats, plot_width, plot_height)
        elif plot_type == "Regression Plot":
            self._create_regression_plot(valid_data, col1, col2, stats, plot_width, plot_height)
        elif plot_type == "Heat Map Scatter":
            self._create_heat_map_scatter_plot(df, col1, col2, stats, plot_width, plot_height, heat_column, flow_name_column)
        
        # Add plot options
        self._add_plot_options(df, col1, col2, stats, plot_width, plot_height)
    
    def _create_enhanced_seaborn_plot(self, valid_data: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], width: int, height: int) -> None:
        """Create enhanced seaborn scatter plot with confidence intervals."""
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="husl")
        
        # Create figure with adjustable size
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Create enhanced scatter plot with seaborn
        sns.regplot(
            data=valid_data, 
            x=col1, 
            y=col2,
            scatter_kws={'alpha': 0.6, 's': 60, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2, 'linestyle': '--'},
            ax=ax,
            ci=95  # 95% confidence interval
        )
        
        # Customize plot
        ax.set_xlabel(col1, fontsize=12, fontweight='bold')
        ax.set_ylabel(col2, fontsize=12, fontweight='bold')
        ax.set_title(f"Enhanced Scatter Plot: {col1} vs {col2}", fontsize=14, fontweight='bold')
        
        # Add statistics text box
        stats_text = f"n = {stats['n_points']}\n"
        stats_text += f"r = {stats['correlation']:.3f}\n"
        stats_text += f"Cov = {stats['covariance']:.3f}\n"
        stats_text += f"95% CI shown"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def _create_classic_matplotlib_plot(self, valid_data: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], width: int, height: int) -> None:
        """Create classic matplotlib scatter plot."""
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Create scatter plot
        scatter = ax.scatter(valid_data[col1], valid_data[col2], alpha=0.6, s=50, color='steelblue')
        
        # Add trend line
        if len(valid_data) > 1:
            z = np.polyfit(valid_data[col1], valid_data[col2], 1)
            p = np.poly1d(z)
            ax.plot(valid_data[col1], p(valid_data[col1]), "r--", alpha=0.8, linewidth=2, label='Trend Line')
        
        # Customize plot
        ax.set_xlabel(col1, fontsize=12, fontweight='bold')
        ax.set_ylabel(col2, fontsize=12, fontweight='bold')
        ax.set_title(f"Classic Scatter Plot: {col1} vs {col2}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics text box
        stats_text = f"n = {stats['n_points']}\n"
        stats_text += f"r = {stats['correlation']:.3f}\n"
        stats_text += f"Cov = {stats['covariance']:.3f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def _create_joint_plot(self, valid_data: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], width: int, height: int) -> None:
        """Create seaborn joint plot with histograms and KDE."""
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="husl")
        
        # Create joint plot with adjustable size
        g = sns.jointplot(
            data=valid_data,
            x=col1,
            y=col2,
            kind="scatter",
            height=height,
            joint_kws={'alpha': 0.6, 's': 50},
            marginal_kws={'bins': 20, 'kde': True}
        )
        
        # Add trend line
        if len(valid_data) > 1:
            sns.regplot(
                data=valid_data, 
                x=col1, 
                y=col2,
                scatter=False,
                line_kws={'color': 'red', 'linewidth': 2, 'linestyle': '--'},
                ax=g.ax_joint
            )
        
        # Customize titles
        g.fig.suptitle(f"Joint Plot: {col1} vs {col2}", y=1.02, fontsize=14, fontweight='bold')
        g.ax_joint.set_xlabel(col1, fontsize=12, fontweight='bold')
        g.ax_joint.set_ylabel(col2, fontsize=12, fontweight='bold')
        
        # Add statistics text
        stats_text = f"n = {stats['n_points']}\nr = {stats['correlation']:.3f}"
        g.ax_joint.text(0.02, 0.98, stats_text, transform=g.ax_joint.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        plt.tight_layout()
        st.pyplot(g.fig)
        plt.close(g.fig)
    
    def _create_regression_plot(self, valid_data: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], width: int, height: int) -> None:
        """Create seaborn regression plot with confidence intervals and residuals."""
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="husl")
        
        # Create subplots for main plot and residuals with adjustable size
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height), height_ratios=[3, 1])
        
        # Main regression plot
        sns.regplot(
            data=valid_data, 
            x=col1, 
            y=col2,
            scatter_kws={'alpha': 0.6, 's': 60, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2},
            ax=ax1,
            ci=95
        )
        
        ax1.set_xlabel(col1, fontsize=12, fontweight='bold')
        ax1.set_ylabel(col2, fontsize=12, fontweight='bold')
        ax1.set_title(f"Regression Plot: {col1} vs {col2}", fontsize=14, fontweight='bold')
        
        # Residuals plot
        if len(valid_data) > 1:
            # Calculate residuals
            z = np.polyfit(valid_data[col1], valid_data[col2], 1)
            p = np.poly1d(z)
            predicted = p(valid_data[col1])
            residuals = valid_data[col2] - predicted
            
            # Plot residuals
            ax2.scatter(valid_data[col1], residuals, alpha=0.6, s=50, color='orange')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax2.set_xlabel(col1, fontsize=12, fontweight='bold')
            ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
            ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"n = {stats['n_points']}\nr = {stats['correlation']:.3f}\nR² = {stats['correlation']**2:.3f}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
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
            # Color by heat column
            scatter = ax.scatter(
                valid_data[col1], 
                valid_data[col2], 
                c=valid_data[heat_column], 
                cmap='viridis', 
                alpha=0.7, 
                s=80,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f'{heat_column} (Heat Value)', fontsize=10, fontweight='bold')
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
                    y_pos - (valid_data[col2].max() - valid_data[col2].min()) * 0.02, 
                    flow_name, 
                    fontsize=8, 
                    ha='center', 
                    va='top',
                    rotation=45,
                    alpha=0.8
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
    
    def _add_plot_options(self, df: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], width: int, height: int) -> None:
        """Add plot options and export functionality."""
        st.markdown("**Plot Options:**")
        col1_opt, col2_opt = st.columns(2)
        
        with col1_opt:
            if st.button("📥 Download Current Plot as PNG"):
                # Get the current plot type
                plot_type = st.session_state.get("plot_type_selector", "Enhanced Seaborn")
                
                # Create appropriate plot for download
                valid_data = df[[col1, col2]].dropna()
                
                if plot_type == "Enhanced Seaborn":
                    fig = self._create_downloadable_seaborn_plot(valid_data, col1, col2, stats, width, height)
                elif plot_type == "Joint Plot":
                    fig = self._create_downloadable_joint_plot(valid_data, col1, col2, stats, width, height)
                elif plot_type == "Regression Plot":
                    fig = self._create_downloadable_regression_plot(valid_data, col1, col2, stats, width, height)
                elif plot_type == "Heat Map Scatter":
                    heat_col = st.session_state.get("heat_column_selector", "None")
                    flow_col = st.session_state.get("flow_name_selector", "None")
                    heat_col = None if heat_col == "None" else heat_col
                    flow_col = None if flow_col == "None" else flow_col
                    fig = self._create_downloadable_heat_map_scatter_plot(df, col1, col2, stats, width, height, heat_col, flow_col)
                else:
                    fig = self._create_downloadable_matplotlib_plot(valid_data, col1, col2, stats, width, height)
                
                # Save to buffer
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                # Create download button
                st.download_button(
                    label="Click to download",
                    data=buf.getvalue(),
                    file_name=f"scatter_plot_{col1}_vs_{col2}_{plot_type.lower().replace(' ', '_')}.png",
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
    
    def _create_downloadable_seaborn_plot(self, valid_data: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], width: int, height: int) -> plt.Figure:
        """Create downloadable seaborn plot."""
        sns.set_theme(style="whitegrid", palette="husl")
        fig, ax = plt.subplots(figsize=(width, height))
        
        sns.regplot(
            data=valid_data, 
            x=col1, 
            y=col2,
            scatter_kws={'alpha': 0.6, 's': 60, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2, 'linestyle': '--'},
            ax=ax,
            ci=95
        )
        
        ax.set_xlabel(col1, fontsize=12, fontweight='bold')
        ax.set_ylabel(col2, fontsize=12, fontweight='bold')
        ax.set_title(f"Enhanced Scatter Plot: {col1} vs {col2}", fontsize=14, fontweight='bold')
        
        stats_text = f"n = {stats['n_points']}\nr = {stats['correlation']:.3f}\nCov = {stats['covariance']:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        return fig
    
    def _create_downloadable_joint_plot(self, valid_data: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], width: int, height: int) -> plt.Figure:
        """Create downloadable joint plot."""
        sns.set_theme(style="whitegrid", palette="husl")
        g = sns.jointplot(
            data=valid_data,
            x=col1,
            y=col2,
            kind="scatter",
            height=height,
            joint_kws={'alpha': 0.6, 's': 50},
            marginal_kws={'bins': 20, 'kde': True}
        )
        
        if len(valid_data) > 1:
            sns.regplot(
                data=valid_data, 
                x=col1, 
                y=col2,
                scatter=False,
                line_kws={'color': 'red', 'linewidth': 2, 'linestyle': '--'},
                ax=g.ax_joint
            )
        
        g.fig.suptitle(f"Joint Plot: {col1} vs {col2}", y=1.02, fontsize=14, fontweight='bold')
        g.ax_joint.set_xlabel(col1, fontsize=12, fontweight='bold')
        g.ax_joint.set_ylabel(col2, fontsize=12, fontweight='bold')
        
        return g.fig
    
    def _create_downloadable_regression_plot(self, valid_data: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], width: int, height: int) -> plt.Figure:
        """Create downloadable regression plot."""
        sns.set_theme(style="whitegrid", palette="husl")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height), height_ratios=[3, 1])
        
        sns.regplot(
            data=valid_data, 
            x=col1, 
            y=col2,
            scatter_kws={'alpha': 0.6, 's': 60, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2},
            ax=ax1,
            ci=95
        )
        
        ax1.set_xlabel(col1, fontsize=12, fontweight='bold')
        ax1.set_ylabel(col2, fontsize=12, fontweight='bold')
        ax1.set_title(f"Regression Plot: {col1} vs {col2}", fontsize=14, fontweight='bold')
        
        if len(valid_data) > 1:
            z = np.polyfit(valid_data[col1], valid_data[col2], 1)
            p = np.poly1d(z)
            predicted = p(valid_data[col1])
            residuals = valid_data[col2] - predicted
            
            ax2.scatter(valid_data[col1], residuals, alpha=0.6, s=50, color='orange')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax2.set_xlabel(col1, fontsize=12, fontweight='bold')
            ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
            ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        return fig
    
    def _create_downloadable_matplotlib_plot(self, valid_data: pd.DataFrame, col1: str, col2: str, stats: Dict[str, float], width: int, height: int) -> plt.Figure:
        """Create downloadable matplotlib plot."""
        fig, ax = plt.subplots(figsize=(width, height))
        
        ax.scatter(valid_data[col1], valid_data[col2], alpha=0.6, s=50, color='steelblue')
        
        if len(valid_data) > 1:
            z = np.polyfit(valid_data[col1], valid_data[col2], 1)
            p = np.poly1d(z)
            ax.plot(valid_data[col1], p(valid_data[col1]), "r--", alpha=0.8, linewidth=2, label='Trend Line')
        
        ax.set_xlabel(col1, fontsize=12, fontweight='bold')
        ax.set_ylabel(col2, fontsize=12, fontweight='bold')
        ax.set_title(f"Classic Scatter Plot: {col1} vs {col2}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        stats_text = f"n = {stats['n_points']}\nr = {stats['correlation']:.3f}\nCov = {stats['covariance']:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return fig
    
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
            # Color by heat column
            scatter = ax.scatter(
                valid_data[col1], 
                valid_data[col2], 
                c=valid_data[heat_column], 
                cmap='viridis', 
                alpha=0.7, 
                s=80,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f'{heat_column} (Heat Value)', fontsize=10, fontweight='bold')
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
                    y_pos - (valid_data[col2].max() - valid_data[col2].min()) * 0.02, 
                    flow_name, 
                    fontsize=8, 
                    ha='center', 
                    va='top',
                    rotation=45,
                    alpha=0.8
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
