"""
Interactive table display module for Dataframe2Visualization.
"""

import streamlit as st
import pandas as pd
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
                 class="clickable-image" 
                 onclick="showImage('{image_key}', '{col_name}', {row_idx})"
                 style="max-width: 100px; max-height: 100px; object-fit: contain;">
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
