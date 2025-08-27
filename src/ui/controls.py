"""
Table controls module for Dataframe2Visualization.
"""

import streamlit as st
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from ..config.settings import AppConfig

class TableControls:
    """Provides search, filter, and pagination controls for the table."""
    
    def __init__(self):
        """Initialize the TableControls."""
        self.current_page = 1
        self.items_per_page = AppConfig.MAX_ROWS_PER_PAGE
        self.search_term = ""
        self.column_filters = {}
        self.sort_column = None
        self.sort_ascending = True
    
    def render_controls(self, df: pd.DataFrame, column_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render all control elements.
        
        Args:
            df: DataFrame to control
            column_metadata: Metadata about DataFrame columns
            
        Returns:
            Dict containing control state and filtered data
        """
        st.markdown("### Table Controls")
        
        # Search and filter controls
        filtered_df = self._render_search_and_filters(df, column_metadata)
        
        # Sorting controls (apply to filtered data first)
        sorted_df = self._render_sorting_controls(filtered_df)
        
        # Pagination controls (apply to sorted data)
        paginated_df = self._render_pagination(sorted_df)
        
        # Display settings
        self._render_display_settings()
        
        # Export controls
        self._render_export_controls(df)
        
        return {
            'filtered_data': filtered_df,
            'sorted_data': sorted_df,
            'final_data': paginated_df,
            'controls_state': self._get_controls_state()
        }
    
    def _render_search_and_filters(self, df: pd.DataFrame, column_metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Render search and filter controls.
        
        Args:
            df: DataFrame to filter
            column_metadata: Column metadata
            
        Returns:
            Filtered DataFrame
        """
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Global search
            self.search_term = st.text_input(
                "Search across all columns",
                value=self.search_term,
                placeholder="Enter search term...",
                help="Search for text in any column"
            )
        
        with col2:
            # Items per page
            self.items_per_page = st.selectbox(
                "Items per page",
                options=[10, 25, 50, 100],
                index=[10, 25, 50, 100].index(self.items_per_page),
                help="Number of rows to display per page"
            )
        
        # Column-specific filters
        col1, col2 = st.columns([3, 1])
        
        with col1:
            show_filters = st.checkbox("Show column filters", value=False)
        
        with col2:
            if st.button("Reset All Filters", help="Clear all search and filter settings"):
                self.reset_controls()
                st.rerun()
        
        if show_filters:
            self._render_column_filters(df, column_metadata)
        
        # Show active filters summary
        if self.search_term or self.column_filters:
            self._show_active_filters_summary()
        
        # Apply search filter
        filtered_df = df.copy()
        if self.search_term:
            filtered_df = self._apply_search_filter(filtered_df, self.search_term)
        
        # Apply column filters
        if self.column_filters:
            filtered_df = self._apply_column_filters(filtered_df)
        
        return filtered_df
    
    def _render_column_filters(self, df: pd.DataFrame, column_metadata: Dict[str, Any]) -> None:
        """
        Render filters for individual columns.
        
        Args:
            df: DataFrame to filter
            column_metadata: Column metadata
        """
        st.markdown("#### Column Filters")
        
        # Create filters for each column
        for col_name in df.columns:
            col_meta = column_metadata.get(col_name, {})
            
            if col_meta.get('contains_images', False):
                # Skip image columns for text filtering
                continue
            
            # Get unique values for the column
            unique_values = df[col_name].dropna().unique()
            
            if len(unique_values) <= 20:  # Only show filter if reasonable number of values
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Filter operation
                    filter_op = st.selectbox(
                        f"Filter operation for {col_name}",
                        options=["equals", "contains", "starts with", "ends with", "greater than", "less than"],
                        key=f"filter_op_{col_name}",
                        help=f"Choose how to filter {col_name}"
                    )
                
                with col2:
                    # Filter values
                    if filter_op in ["equals", "contains", "starts with", "ends with"]:
                        col2a, col2b = st.columns([3, 1])
                        
                        with col2a:
                            selected_values = st.multiselect(
                                f"Values for {col_name}",
                                options=unique_values,
                                default=self.column_filters.get(col_name, {}).get('values', []),
                                key=f"filter_values_{col_name}",
                                help=f"Select values to filter {col_name}"
                            )
                        
                        with col2b:
                            if st.button("Clear", key=f"clear_filter_{col_name}", help=f"Clear filter for {col_name}"):
                                if col_name in self.column_filters:
                                    del self.column_filters[col_name]
                                st.rerun()
                        
                        if selected_values:
                            self.column_filters[col_name] = {
                                'operation': filter_op,
                                'values': selected_values
                            }
                        elif col_name in self.column_filters:
                            del self.column_filters[col_name]
                    
                    elif filter_op in ["greater than", "less than"]:
                        # For numeric comparisons
                        if pd.api.types.is_numeric_dtype(df[col_name]):
                            min_val = float(df[col_name].min())
                            max_val = float(df[col_name].max())
                            
                            if filter_op == "greater than":
                                col2a, col2b = st.columns([3, 1])
                                
                                with col2a:
                                    threshold = st.number_input(
                                        f"Greater than value for {col_name}",
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=min_val,
                                        step=(max_val - min_val) / 100,
                                        key=f"filter_threshold_{col_name}"
                                    )
                                
                                with col2b:
                                    if st.button("Clear", key=f"clear_filter_{col_name}", help=f"Clear filter for {col_name}"):
                                        if col_name in self.column_filters:
                                            del self.column_filters[col_name]
                                        st.rerun()
                                
                                self.column_filters[col_name] = {
                                    'operation': filter_op,
                                    'threshold': threshold
                                }
                                
                            else:  # less than
                                col2a, col2b = st.columns([3, 1])
                                
                                with col2a:
                                    threshold = st.number_input(
                                        f"Less than value for {col_name}",
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=max_val,
                                        step=(max_val - min_val) / 100,
                                        key=f"filter_threshold_{col_name}"
                                    )
                                
                                with col2b:
                                    if st.button("Clear", key=f"clear_filter_{col_name}", help=f"Clear filter for {col_name}"):
                                        if col_name in self.column_filters:
                                            del self.column_filters[col_name]
                                        st.rerun()
                                
                                self.column_filters[col_name] = {
                                    'operation': filter_op,
                                    'threshold': threshold
                                }
                        else:
                            st.warning(f"Column {col_name} is not numeric for comparison operations")
    
    def _apply_search_filter(self, df: pd.DataFrame, search_term: str) -> pd.DataFrame:
        """
        Apply search filter across all columns.
        
        Args:
            df: DataFrame to filter
            search_term: Search term
            
        Returns:
            Filtered DataFrame
        """
        if not search_term:
            return df
        
        # Create a mask for rows that contain the search term
        mask = pd.Series([False] * len(df), index=df.index)
        
        for col in df.columns:
            try:
                # Convert column to string and search
                col_str = df[col].astype(str)
                col_mask = col_str.str.contains(search_term, case=False, na=False)
                mask = mask | col_mask
            except:
                # Skip columns that can't be converted to string
                continue
        
        return df[mask]
    
    def _apply_column_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column-specific filters.
        
        Args:
            df: DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        for col_name, filter_config in self.column_filters.items():
            if col_name not in filtered_df.columns:
                continue
                
            operation = filter_config.get('operation', 'equals')
            
            if operation == "equals":
                values = filter_config.get('values', [])
                if values:
                    filtered_df = filtered_df[filtered_df[col_name].isin(values)]
                    
            elif operation == "contains":
                values = filter_config.get('values', [])
                if values:
                    mask = filtered_df[col_name].astype(str).str.contains('|'.join(values), case=False, na=False)
                    filtered_df = filtered_df[mask]
                    
            elif operation == "starts with":
                values = filter_config.get('values', [])
                if values:
                    mask = filtered_df[col_name].astype(str).str.startswith(tuple(values), na=False)
                    filtered_df = filtered_df[mask]
                    
            elif operation == "ends with":
                values = filter_config.get('values', [])
                if values:
                    mask = filtered_df[col_name].astype(str).str.endswith(tuple(values), na=False)
                    filtered_df = filtered_df[mask]
                    
            elif operation == "greater than":
                threshold = filter_config.get('threshold')
                if threshold is not None:
                    filtered_df = filtered_df[filtered_df[col_name] > threshold]
                    
            elif operation == "less than":
                threshold = filter_config.get('threshold')
                if threshold is not None:
                    filtered_df = filtered_df[filtered_df[col_name] < threshold]
        
        return filtered_df
    
    def _render_pagination(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Render pagination controls.
        
        Args:
            df: DataFrame to paginate
            
        Returns:
            Paginated DataFrame
        """
        total_rows = len(df)
        total_pages = (total_rows + self.items_per_page - 1) // self.items_per_page
        
        if total_pages <= 1:
            return df
        
        st.markdown("#### Pagination")
        
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            if st.button("← Previous", disabled=self.current_page <= 1):
                self.current_page = max(1, self.current_page - 1)
                st.rerun()
        
        with col2:
            st.markdown(f"Page {self.current_page} of {total_pages}")
        
        with col3:
            new_page = st.number_input(
                "Go to page",
                min_value=1,
                max_value=total_pages,
                value=self.current_page,
                step=1
            )
            if new_page != self.current_page:
                self.current_page = new_page
                st.rerun()
        
        with col4:
            if st.button("Next →", disabled=self.current_page >= total_pages):
                self.current_page = min(total_pages, self.current_page + 1)
                st.rerun()
        
        # Apply pagination
        start_idx = (self.current_page - 1) * self.items_per_page
        end_idx = start_idx + self.items_per_page
        
        return df.iloc[start_idx:end_idx]
    
    def _render_sorting_controls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Render sorting controls.
        
        Args:
            df: DataFrame to sort
            
        Returns:
            Sorted DataFrame
        """
        if len(df) == 0:
            return df
        
        st.markdown("#### Sorting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Column selection for sorting
            sortable_columns = [col for col in df.columns if df[col].dtype in ['object', 'int64', 'float64']]
            if sortable_columns:
                self.sort_column = st.selectbox(
                    "Sort by column",
                    options=sortable_columns,
                    index=0 if not self.sort_column else sortable_columns.index(self.sort_column),
                    help="Select column to sort by"
                )
        
        with col2:
            # Sort order
            self.sort_ascending = st.checkbox(
                "Sort ascending",
                value=self.sort_ascending,
                help="Check for ascending order, uncheck for descending"
            )
        
        # Apply sorting
        if self.sort_column and self.sort_column in df.columns:
            try:
                df = df.sort_values(by=self.sort_column, ascending=self.sort_ascending)
            except:
                st.warning(f"Could not sort by column '{self.sort_column}'")
        
        return df
    
    def _render_display_settings(self) -> None:
        """Render display settings controls."""
        st.markdown("#### Display Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show/hide controls
            show_controls = st.checkbox(
                "Show control panel",
                value=True,
                help="Show or hide the control panel"
            )
        
        with col2:
            # Table height
            table_height = st.slider(
                "Table height",
                min_value=300,
                max_value=1000,
                value=AppConfig.DEFAULT_TABLE_HEIGHT,
                step=50,
                help="Height of the table in pixels"
            )
    
    def _prepare_dataframe_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for export by handling numpy arrays and other complex types.
        
        Args:
            df: Original DataFrame
            
        Returns:
            Clean DataFrame ready for export
        """
        export_df = df.copy()
        for col in export_df.columns:
            export_df[col] = export_df[col].apply(lambda x: str(x) if hasattr(x, 'shape') else x)
        return export_df
    
    def _show_active_filters_summary(self) -> None:
        """
        Display a summary of currently active filters.
        """
        st.markdown("#### Active Filters")
        
        active_filters = []
        
        if self.search_term:
            active_filters.append(f"**Search**: '{self.search_term}'")
        
        for col_name, filter_config in self.column_filters.items():
            operation = filter_config.get('operation', 'equals')
            
            if operation in ["equals", "contains", "starts with", "ends with"]:
                values = filter_config.get('values', [])
                if values:
                    active_filters.append(f"**{col_name}** {operation}: {', '.join(map(str, values))}")
                    
            elif operation in ["greater than", "less than"]:
                threshold = filter_config.get('threshold')
                if threshold is not None:
                    active_filters.append(f"**{col_name}** {operation} {threshold}")
        
        if active_filters:
            for filter_desc in active_filters:
                st.markdown(f"• {filter_desc}")
        else:
            st.markdown("*No active filters*")
    
    def _render_export_controls(self, df: pd.DataFrame) -> None:
        """
        Render export controls.
        
        Args:
            df: DataFrame to export
        """
        st.markdown("#### Export Options")
        
        # Prepare clean DataFrame for export
        export_df = self._prepare_dataframe_for_export(df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as CSV
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="dataframe_export.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export as Excel
            try:
                import io
                buffer = io.BytesIO()
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='Sheet1')
                buffer.seek(0)
                
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name="dataframe_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.info("Excel export requires openpyxl package")
        
        with col3:
            # Export as JSON
            json_data = export_df.to_json(index=False, orient='records')
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="dataframe_export.json",
                mime="application/json"
            )
    
    def _get_controls_state(self) -> Dict[str, Any]:
        """Get current state of all controls."""
        return {
            'current_page': self.current_page,
            'items_per_page': self.items_per_page,
            'search_term': self.search_term,
            'column_filters': self.column_filters.copy(),
            'sort_column': self.sort_column,
            'sort_ascending': self.sort_ascending
        }
    
    def reset_controls(self) -> None:
        """Reset all controls to default values."""
        self.current_page = 1
        self.items_per_page = AppConfig.MAX_ROWS_PER_PAGE
        self.search_term = ""
        self.column_filters.clear()
        self.sort_column = None
        self.sort_ascending = True
    
    def get_filtered_row_count(self, original_df: pd.DataFrame) -> int:
        """
        Get the number of rows after applying filters.
        
        Args:
            original_df: Original DataFrame
            
        Returns:
            Number of filtered rows
        """
        filtered_df = original_df.copy()
        
        if self.search_term:
            filtered_df = self._apply_search_filter(filtered_df, self.search_term)
        
        if self.column_filters:
            filtered_df = self._apply_column_filters(filtered_df)
        
        return len(filtered_df)
