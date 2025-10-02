"""
Table controls module for Dataframe2Visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
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
        
        # Pareto frontier filter controls (apply to filtered data)
        pareto_filtered_df = self._render_pareto_frontier_controls(filtered_df)
        
        # Sorting controls (apply to Pareto filtered data)
        sorted_df = self._render_sorting_controls(pareto_filtered_df)
        
        # No pagination - show all data
        final_df = sorted_df
        
        # Display settings
        self._render_display_settings()
        
        # Export controls
        self._render_export_controls(final_df)
        
        return {
            'filtered_data': filtered_df,
            'pareto_filtered_data': pareto_filtered_df,
            'sorted_data': sorted_df,
            'final_data': final_df,
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
            is_numeric = pd.api.types.is_numeric_dtype(df[col_name])
            
            # Determine if this is a discrete or continuous column
            is_discrete = len(unique_values) <= 20
            is_continuous_numeric = is_numeric and not is_discrete
            
            # Show filters for discrete columns or continuous numeric columns
            if is_discrete or is_continuous_numeric:
                st.markdown(f"**{col_name}**")
                
                if is_continuous_numeric:
                    # Continuous numeric column - only show range-based filters
                    self._render_continuous_numeric_filter(df, col_name)
                else:
                    # Discrete column - show all filter types
                    self._render_discrete_column_filter(df, col_name, unique_values, is_numeric)
    
    def _render_continuous_numeric_filter(self, df: pd.DataFrame, col_name: str) -> None:
        """
        Render filter controls for continuous numeric columns.
        
        Args:
            df: DataFrame containing the column
            col_name: Name of the numeric column to filter
        """
        min_val = float(df[col_name].min())
        max_val = float(df[col_name].max())
        
        # Display value range hint
        st.caption(f"📊 Data Range: [{min_val:.4g}, {max_val:.4g}]")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Filter operation selection
            filter_op = st.selectbox(
                "Operation",
                options=["greater than", "less than", "between"],
                key=f"filter_op_{col_name}",
                help=f"Choose filtering operation for {col_name}"
            )
        
        with col2:
            if filter_op == "between":
                # Range filter
                col2a, col2b = st.columns(2)
                
                with col2a:
                    lower_bound = st.number_input(
                        "Min value",
                        value=min_val,
                        format="%.6g",
                        key=f"filter_lower_{col_name}",
                        help=f"Lower bound (data range: [{min_val:.4g}, {max_val:.4g}])"
                    )
                
                with col2b:
                    upper_bound = st.number_input(
                        "Max value",
                        value=max_val,
                        format="%.6g",
                        key=f"filter_upper_{col_name}",
                        help=f"Upper bound (data range: [{min_val:.4g}, {max_val:.4g}])"
                    )
                
                # Validation - just show warnings, still apply filter
                if lower_bound > upper_bound:
                    st.warning(f"⚠️ Min value ({lower_bound:.4g}) should be ≤ Max value ({upper_bound:.4g})")
                
                if lower_bound < min_val or upper_bound > max_val:
                    st.info(f"ℹ️ Filter range extends beyond data range [{min_val:.4g}, {max_val:.4g}]")
                
                # Always apply the filter
                self.column_filters[col_name] = {
                    'operation': filter_op,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            
            elif filter_op == "greater than":
                threshold = st.number_input(
                    "Threshold value",
                    value=min_val,
                    format="%.6g",
                    key=f"filter_threshold_{col_name}",
                    help=f"Values greater than this (data range: [{min_val:.4g}, {max_val:.4g}])"
                )
                
                # Show info if outside data range
                if threshold < min_val or threshold > max_val:
                    st.info(f"ℹ️ Threshold is outside data range [{min_val:.4g}, {max_val:.4g}]")
                
                # Always apply the filter
                self.column_filters[col_name] = {
                    'operation': filter_op,
                    'threshold': threshold
                }
            
            else:  # less than
                threshold = st.number_input(
                    "Threshold value",
                    value=max_val,
                    format="%.6g",
                    key=f"filter_threshold_{col_name}",
                    help=f"Values less than this (data range: [{min_val:.4g}, {max_val:.4g}])"
                )
                
                # Show info if outside data range
                if threshold < min_val or threshold > max_val:
                    st.info(f"ℹ️ Threshold is outside data range [{min_val:.4g}, {max_val:.4g}]")
                
                # Always apply the filter
                self.column_filters[col_name] = {
                    'operation': filter_op,
                    'threshold': threshold
                }
        
        with col3:
            if st.button("Clear", key=f"clear_filter_{col_name}", help=f"Clear filter for {col_name}"):
                if col_name in self.column_filters:
                    del self.column_filters[col_name]
                st.rerun()
    
    def _render_discrete_column_filter(self, df: pd.DataFrame, col_name: str, 
                                      unique_values: np.ndarray, is_numeric: bool) -> None:
        """
        Render filter controls for discrete columns (≤20 unique values).
        
        Args:
            df: DataFrame containing the column
            col_name: Name of the column to filter
            unique_values: Array of unique values in the column
            is_numeric: Whether the column is numeric type
        """
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Filter operation options depend on whether column is numeric
            if is_numeric:
                filter_options = ["equals", "greater than", "less than"]
            else:
                filter_options = ["equals", "contains", "starts with", "ends with"]
            
            filter_op = st.selectbox(
                "Operation",
                options=filter_options,
                key=f"filter_op_{col_name}",
                help=f"Choose how to filter {col_name}"
            )
        
        with col2:
            # Filter values
            if filter_op in ["equals", "contains", "starts with", "ends with"]:
                col2a, col2b = st.columns([3, 1])
                
                with col2a:
                    selected_values = st.multiselect(
                        "Values",
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
                # For numeric comparisons on discrete numeric columns
                min_val = float(df[col_name].min())
                max_val = float(df[col_name].max())
                
                col2a, col2b = st.columns([3, 1])
                
                with col2a:
                    threshold = st.number_input(
                        f"{filter_op.title()} value",
                        value=min_val if filter_op == "greater than" else max_val,
                        format="%.6g",
                        key=f"filter_threshold_{col_name}",
                        help=f"Data range: [{min_val:.4g}, {max_val:.4g}]"
                    )
                
                with col2b:
                    if st.button("Clear", key=f"clear_filter_{col_name}", help=f"Clear filter for {col_name}"):
                        if col_name in self.column_filters:
                            del self.column_filters[col_name]
                        st.rerun()
                
                # Show info if outside data range
                if threshold < min_val or threshold > max_val:
                    st.info(f"ℹ️ Threshold is outside data range [{min_val:.4g}, {max_val:.4g}]")
                
                self.column_filters[col_name] = {
                    'operation': filter_op,
                    'threshold': threshold
                }
    
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
            
            elif operation == "between":
                lower_bound = filter_config.get('lower_bound')
                upper_bound = filter_config.get('upper_bound')
                if lower_bound is not None and upper_bound is not None:
                    filtered_df = filtered_df[
                        (filtered_df[col_name] >= lower_bound) & 
                        (filtered_df[col_name] <= upper_bound)
                    ]
        
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
        
        col1, col2, col3 = st.columns(3)
        
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
        
        with col3:
            # Text truncation settings
            st.markdown("**Text Display:**")
            
            # Character limit for text truncation
            char_limit = st.number_input(
                "Max characters per cell",
                min_value=10,
                max_value=1000,
                value=50,
                step=10,
                key="text_char_limit",
                help="Maximum number of characters to display in each cell"
            )
            
            # CSS width limit
            css_width = st.number_input(
                "Max cell width (px)",
                min_value=100,
                max_value=800,
                value=200,
                step=50,
                key="text_css_width",
                help="Maximum width of table cells in pixels"
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
                    active_filters.append(f"**{col_name}** {operation} {threshold:.4g}")
            
            elif operation == "between":
                lower_bound = filter_config.get('lower_bound')
                upper_bound = filter_config.get('upper_bound')
                if lower_bound is not None and upper_bound is not None:
                    active_filters.append(f"**{col_name}** between [{lower_bound:.4g}, {upper_bound:.4g}]")
        
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
    
    def _render_pareto_frontier_controls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Render Pareto frontier filter controls.
        
        Args:
            df: DataFrame to filter
            
        Returns:
            DataFrame with Pareto filtering applied (if enabled)
        """
        # Check if Pareto filtering is enabled
        pareto_enabled = st.checkbox(
            "🎯 Enable Pareto Frontier Filter",
            key="pareto_enabled",
            help="Filter data to show only Pareto-optimal solutions"
        )
        
        if not pareto_enabled:
            return df
        
        # Get numeric columns for Pareto analysis
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ **Insufficient Data:** Need at least 2 numeric columns for Pareto frontier analysis.")
            return df
        
        # Pareto filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Objective 1:**")
            pareto_col1 = st.selectbox(
                "Column 1",
                numeric_cols,
                key="pareto_col1",
                help="First objective column"
            )
            pareto_obj1_type = st.selectbox(
                "Optimization",
                ["Maximize", "Minimize"],
                key="pareto_obj1_type",
                help="Whether to maximize or minimize this objective"
            )
        
        with col2:
            st.markdown("**Objective 2:**")
            pareto_col2 = st.selectbox(
                "Column 2",
                [col for col in numeric_cols if col != pareto_col1],
                key="pareto_col2",
                help="Second objective column"
            )
            pareto_obj2_type = st.selectbox(
                "Optimization",
                ["Maximize", "Minimize"],
                key="pareto_obj2_type",
                help="Whether to maximize or minimize this objective"
            )
        
        with col3:
            st.markdown("**Budget:**")
            budget = st.number_input(
                "Max Solutions",
                min_value=1,
                max_value=len(df),
                value=min(100, len(df)),
                key="pareto_budget",
                help="Maximum number of Pareto-optimal solutions to return"
            )
        
        # Apply Pareto filtering
        try:
            filtered_df = self._calculate_pareto_frontier(
                df, pareto_col1, pareto_col2, 
                pareto_obj1_type, pareto_obj2_type, budget
            )
            
            # Show results with tier information
            if len(filtered_df) == budget:
                st.success(f"✅ **Pareto Filter Applied:** {len(filtered_df)} solutions selected from {len(df)} total solutions (budget met).")
            else:
                st.success(f"✅ **Pareto Filter Applied:** {len(filtered_df)} solutions selected from {len(df)} total solutions (all available Pareto solutions).")
            
            return filtered_df
            
        except Exception as e:
            st.error(f"❌ **Error applying Pareto filter:** {str(e)}")
            return df
    
    def _calculate_pareto_frontier(self, df: pd.DataFrame, col1: str, col2: str, 
                                 obj1_type: str, obj2_type: str, budget: int) -> pd.DataFrame:
        """
        Calculate Pareto frontier based on two objectives.
        
        Args:
            df: DataFrame to analyze
            col1: First objective column
            col2: Second objective column
            obj1_type: "Maximize" or "Minimize" for first objective
            obj2_type: "Maximize" or "Minimize" for second objective
            budget: Maximum number of solutions to return
            
        Returns:
            DataFrame containing Pareto-optimal solutions
        """
        # Remove rows with NaN values in objective columns
        valid_data = df.dropna(subset=[col1, col2]).copy()
        
        if len(valid_data) == 0:
            st.error("❌ **No Valid Data:** All rows contain NaN values in the selected objective columns.")
            return df.head(0)
        
        # Normalize objectives (convert minimize to maximize by negating)
        obj1_values = valid_data[col1].values
        obj2_values = valid_data[col2].values
        
        if obj1_type == "Minimize":
            obj1_values = -obj1_values
        if obj2_type == "Minimize":
            obj2_values = -obj2_values
        
        # Calculate Pareto frontier using non-dominated sorting
        pareto_indices = self._non_dominated_sorting(obj1_values, obj2_values, budget)
        
        # Return filtered DataFrame
        return valid_data.iloc[pareto_indices].reset_index(drop=True)
    
    def _non_dominated_sorting(self, obj1_values: np.ndarray, obj2_values: np.ndarray, 
                              budget: int) -> List[int]:
        """
        Perform multi-tier non-dominated sorting to find Pareto-optimal solutions.
        
        This implements a multi-tier approach:
        - Tier 1: Non-dominated solutions (Pareto frontier)
        - Tier 2: Solutions dominated only by Tier 1
        - Tier 3: Solutions dominated by Tier 1 and Tier 2
        - Continue until budget is met
        
        Args:
            obj1_values: First objective values (normalized for maximization)
            obj2_values: Second objective values (normalized for maximization)
            budget: Maximum number of solutions to return
            
        Returns:
            List of indices of solutions from multiple Pareto tiers
        """
        n = len(obj1_values)
        all_indices = list(range(n))
        selected_indices = []
        remaining_indices = all_indices.copy()
        tier = 1
        
        while len(selected_indices) < budget and remaining_indices:
            # Find non-dominated solutions in remaining indices
            tier_indices = []
            
            for i in remaining_indices:
                is_dominated = False
                
                for j in remaining_indices:
                    if i == j:
                        continue
                    
                    # Check if solution j dominates solution i
                    if (obj1_values[j] >= obj1_values[i] and obj2_values[j] >= obj2_values[i] and
                        (obj1_values[j] > obj1_values[i] or obj2_values[j] > obj2_values[i])):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    tier_indices.append(i)
            
            if not tier_indices:
                # No more non-dominated solutions, break
                break
            
            # Sort tier solutions by combined objective value
            tier_values = []
            for idx in tier_indices:
                combined_value = obj1_values[idx] + obj2_values[idx]
                tier_values.append((combined_value, idx))
            
            tier_values.sort(reverse=True)
            
            # Add tier solutions to selected indices (up to budget)
            remaining_budget = budget - len(selected_indices)
            for _, idx in tier_values[:remaining_budget]:
                selected_indices.append(idx)
            
            # Remove selected solutions from remaining indices
            for idx in tier_indices:
                if idx in remaining_indices:
                    remaining_indices.remove(idx)
            
            tier += 1
        
        return selected_indices
