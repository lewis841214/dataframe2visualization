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
        # Initialize session state for persistent filters
        if 'column_filters' not in st.session_state:
            st.session_state.column_filters = {}
        
        self.current_page = 1
        self.items_per_page = AppConfig.MAX_ROWS_PER_PAGE
        self.search_term = ""
        self.column_filters = st.session_state.column_filters  # Reference to session state
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
        
        # # Show active filters summary
        # if self.search_term or self.column_filters:
        #     self._show_active_filters_summary()
        
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
        Render filters for individual columns using "add rules" mode.
        
        Args:
            df: DataFrame to filter
            column_metadata: Column metadata
        """
        st.markdown("#### Column Filters")
        
        # Get filterable columns (exclude image columns)
        filterable_columns = []
        for col_name in df.columns:
            col_meta = column_metadata.get(col_name, {})
            if not col_meta.get('contains_images', False):
                filterable_columns.append(col_name)
        
        if not filterable_columns:
            st.info("No filterable columns available.")
            return
        
        st.markdown("**➕ Add New Filter Rule:**")
        
        # Column selector
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_column = st.selectbox(
                "Select Column to Filter",
                options=["-- Select a column --"] + filterable_columns,
                key="filter_column_selector",
                help="Choose which column to add a filter rule for"
            )
        
        with col2:
            # Show column info
            if selected_column != "-- Select a column --":
                unique_count = df[selected_column].nunique()
                null_count = df[selected_column].isna().sum()
                if null_count > 0:
                    st.caption(f"📊 {unique_count} unique + {null_count} null")
                else:
                    st.caption(f"📊 {unique_count} unique values")
        
        # If a column is selected, show filter controls
        if selected_column != "-- Select a column --":
            self._render_add_filter_rule_interface(df, selected_column, column_metadata)
        
        # Display active filter rules first
        if self.column_filters:
            st.markdown("**🔧 Active Filter Rules:**")
            self._render_active_filter_rules(df)
            st.markdown("---")
        
    def _render_active_filter_rules(self, df: pd.DataFrame) -> None:
        """
        Display active filter rules with delete buttons.
        
        Args:
            df: DataFrame being filtered
        """
        # Create a copy of keys to avoid modification during iteration
        filter_columns = list(self.column_filters.keys())
        
        for col_name in filter_columns:
            filter_config = self.column_filters.get(col_name)
            if not filter_config:
                continue
                
            col1, col2, col3 = st.columns([2, 4, 1])
            
            with col1:
                st.write(f"**{col_name}**")
            
            with col2:
                # Display rule description
                rule_desc = self._get_filter_rule_description(filter_config)
                st.write(rule_desc)
            
            with col3:
                if st.button("🗑️", key=f"delete_rule_{col_name}", help=f"Delete filter for {col_name}"):
                    # Delete immediately from session state and rerun
                    if col_name in st.session_state.column_filters:
                        del st.session_state.column_filters[col_name]
                    st.rerun()
    
    def _get_filter_rule_description(self, filter_config: Dict[str, Any]) -> str:
        """
        Generate human-readable description of filter rule.
        
        Args:
            filter_config: Filter configuration
            
        Returns:
            Description string
        """
        operation = filter_config.get('operation', 'equals')
        
        if operation == "is null":
            return "is null (empty/None)"
        
        elif operation == "is not null":
            return "is not null (has value)"
        
        elif operation in ["equals", "contains", "starts with", "ends with"]:
            values = filter_config.get('values', [])
            if values:
                value_str = ", ".join(map(str, values[:3]))
                if len(values) > 3:
                    value_str += f" ... (+{len(values)-3} more)"
                return f"{operation}: {value_str}"
        
        elif operation in ["greater than", "less than"]:
            threshold = filter_config.get('threshold')
            if threshold is not None:
                return f"{operation} {threshold:.4g}"
        
        elif operation == "between":
            lower = filter_config.get('lower_bound')
            upper = filter_config.get('upper_bound')
            if lower is not None and upper is not None:
                return f"between [{lower:.4g}, {upper:.4g}]"
        
        return "Unknown rule"
    
    def _render_add_filter_rule_interface(self, df: pd.DataFrame, col_name: str, 
                                         column_metadata: Dict[str, Any]) -> None:
        """
        Render interface to configure and add a new filter rule.
        
        Args:
            df: DataFrame to filter
            col_name: Column to add filter for
            column_metadata: Column metadata
        """
        # Initialize session state for temporary rule configuration
        if 'temp_filter_config' not in st.session_state:
            st.session_state.temp_filter_config = {}
        
        # Get column properties (include NaN values for discrete columns)
        has_null = df[col_name].isna().any()
        unique_values = df[col_name].dropna().unique()
        is_numeric = pd.api.types.is_numeric_dtype(df[col_name])
        is_discrete = len(unique_values) <= 20
        is_continuous_numeric = is_numeric and not is_discrete
        
        st.markdown(f"**Configure filter for: `{col_name}`**")
        
        if is_continuous_numeric:
            # Continuous numeric column
            rule_config = self._render_continuous_filter_builder(df, col_name)
        else:
            # Discrete column
            rule_config = self._render_discrete_filter_builder(df, col_name, unique_values, is_numeric, has_null)
        
        # Add Rule button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("➕ Add Rule", key="add_filter_rule_btn", type="primary"):
                if rule_config:
                    # Explicitly update session state
                    st.session_state.column_filters[col_name] = rule_config
                    # Success message will flash briefly before rerun
                    st.success(f"✅ Filter rule added for {col_name}")
                    st.rerun()
                else:
                    st.warning("⚠️ Please configure the filter rule first")
        
        with col2:
            if st.button("Cancel", key="cancel_filter_rule_btn"):
                st.rerun()
    
    def _render_continuous_filter_builder(self, df: pd.DataFrame, col_name: str) -> Optional[Dict[str, Any]]:
        """
        Render filter builder for continuous numeric columns and return configuration.
        
        Args:
            df: DataFrame containing the column
            col_name: Name of the numeric column
            
        Returns:
            Filter configuration dict or None
        """
        min_val = float(df[col_name].min())
        max_val = float(df[col_name].max())
        
        # Display value range hint
        st.caption(f"📊 Data Range: [{min_val:.4g}, {max_val:.4g}]")
        
        # Filter operation selection
        filter_op = st.selectbox(
            "Operation",
            options=["greater than", "less than", "between"],
            key=f"filter_op_builder_{col_name}",
            help=f"Choose filtering operation for {col_name}"
        )
        
        if filter_op == "between":
            # Range filter
            col2a, col2b = st.columns(2)
            
            with col2a:
                lower_bound = st.number_input(
                    "Min value",
                    value=min_val,
                    format="%.6g",
                    key=f"filter_lower_builder_{col_name}",
                    help=f"Lower bound (data range: [{min_val:.4g}, {max_val:.4g}])"
                )
            
            with col2b:
                upper_bound = st.number_input(
                    "Max value",
                    value=max_val,
                    format="%.6g",
                    key=f"filter_upper_builder_{col_name}",
                    help=f"Upper bound (data range: [{min_val:.4g}, {max_val:.4g}])"
                )
            
            # Validation warnings
            if lower_bound > upper_bound:
                st.warning(f"⚠️ Min value ({lower_bound:.4g}) should be ≤ Max value ({upper_bound:.4g})")
            
            if lower_bound < min_val or upper_bound > max_val:
                st.info(f"ℹ️ Filter range extends beyond data range [{min_val:.4g}, {max_val:.4g}]")
            
            return {
                'operation': filter_op,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        elif filter_op == "greater than":
            threshold = st.number_input(
                "Threshold value",
                value=min_val,
                format="%.6g",
                key=f"filter_threshold_builder_{col_name}",
                help=f"Values greater than this (data range: [{min_val:.4g}, {max_val:.4g}])"
            )
            
            if threshold < min_val or threshold > max_val:
                st.info(f"ℹ️ Threshold is outside data range [{min_val:.4g}, {max_val:.4g}]")
            
            return {
                'operation': filter_op,
                'threshold': threshold
            }
        
        else:  # less than
            threshold = st.number_input(
                "Threshold value",
                value=max_val,
                format="%.6g",
                key=f"filter_threshold_builder_{col_name}",
                help=f"Values less than this (data range: [{min_val:.4g}, {max_val:.4g}])"
            )
            
            if threshold < min_val or threshold > max_val:
                st.info(f"ℹ️ Threshold is outside data range [{min_val:.4g}, {max_val:.4g}]")
            
            return {
                'operation': filter_op,
                'threshold': threshold
            }
    
    def _render_discrete_filter_builder(self, df: pd.DataFrame, col_name: str,
                                       unique_values: np.ndarray, is_numeric: bool, has_null: bool) -> Optional[Dict[str, Any]]:
        """
        Render filter builder for discrete columns and return configuration.
        
        Args:
            df: DataFrame containing the column
            col_name: Name of the column
            unique_values: Array of unique values (excluding NaN)
            is_numeric: Whether the column is numeric
            has_null: Whether the column contains null values
            
        Returns:
            Filter configuration dict or None
        """
        # Filter operation options depend on whether column is numeric
        if is_numeric:
            filter_options = ["equals", "is null", "is not null", "greater than", "less than"]
        else:
            filter_options = ["equals", "is null", "is not null", "contains", "starts with", "ends with"]
        
        filter_op = st.selectbox(
            "Operation",
            options=filter_options,
            key=f"filter_op_builder_{col_name}",
            help=f"Choose how to filter {col_name}"
        )
        
        # Handle null filtering operations
        if filter_op == "is null":
            return {
                'operation': 'is null'
            }
        
        if filter_op == "is not null":
            return {
                'operation': 'is not null'
            }
        
        if filter_op in ["equals", "contains", "starts with", "ends with"]:
            # Add None option if column has null values
            options_list = list(unique_values)
            if has_null:
                options_list = ["(None)"] + options_list
            
            selected_values = st.multiselect(
                "Values",
                options=options_list,
                key=f"filter_values_builder_{col_name}",
                help=f"Select values to filter {col_name}. Select '(None)' to filter for null/empty values."
            )
            
            if selected_values:
                return {
                    'operation': filter_op,
                    'values': selected_values
                }
            else:
                return None 
        
        elif filter_op in ["greater than", "less than"]:
            # For numeric comparisons on discrete numeric columns
            min_val = float(df[col_name].min())
            max_val = float(df[col_name].max())
            
            threshold = st.number_input(
                f"{filter_op.title()} value",
                value=min_val if filter_op == "greater than" else max_val,
                format="%.6g",
                key=f"filter_threshold_builder_{col_name}",
                help=f"Data range: [{min_val:.4g}, {max_val:.4g}]"
            )
            
            if threshold < min_val or threshold > max_val:
                st.info(f"ℹ️ Threshold is outside data range [{min_val:.4g}, {max_val:.4g}]")
            
            return {
                'operation': filter_op,
                'threshold': threshold
            }
        
        return None
    
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
            
            if operation == "is null":
                filtered_df = filtered_df[filtered_df[col_name].isna()]
            
            elif operation == "is not null":
                filtered_df = filtered_df[filtered_df[col_name].notna()]
            
            elif operation == "equals":
                values = filter_config.get('values', [])
                if values:
                    # Separate None and non-None values
                    none_selected = "(None)" in values
                    actual_values = [v for v in values if v != "(None)"]
                    
                    if none_selected and actual_values:
                        # Include both null and specified values
                        mask = filtered_df[col_name].isna() | filtered_df[col_name].isin(actual_values)
                        filtered_df = filtered_df[mask]
                    elif none_selected:
                        # Only null values
                        filtered_df = filtered_df[filtered_df[col_name].isna()]
                    else:
                        # Only actual values
                        filtered_df = filtered_df[filtered_df[col_name].isin(actual_values)]
                    
            elif operation == "contains":
                values = filter_config.get('values', [])
                if values:
                    # Separate None and non-None values
                    none_selected = "(None)" in values
                    actual_values = [v for v in values if v != "(None)"]
                    
                    if actual_values:
                        mask = filtered_df[col_name].astype(str).str.contains('|'.join(actual_values), case=False, na=False)
                    else:
                        mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
                    
                    if none_selected:
                        mask = mask | filtered_df[col_name].isna()
                    
                    filtered_df = filtered_df[mask]
                    
            elif operation == "starts with":
                values = filter_config.get('values', [])
                if values:
                    # Separate None and non-None values
                    none_selected = "(None)" in values
                    actual_values = [v for v in values if v != "(None)"]
                    
                    if actual_values:
                        mask = filtered_df[col_name].astype(str).str.startswith(tuple(actual_values), na=False)
                    else:
                        mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
                    
                    if none_selected:
                        mask = mask | filtered_df[col_name].isna()
                    
                    filtered_df = filtered_df[mask]
                    
            elif operation == "ends with":
                values = filter_config.get('values', [])
                if values:
                    # Separate None and non-None values
                    none_selected = "(None)" in values
                    actual_values = [v for v in values if v != "(None)"]
                    
                    if actual_values:
                        mask = filtered_df[col_name].astype(str).str.endswith(tuple(actual_values), na=False)
                    else:
                        mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
                    
                    if none_selected:
                        mask = mask | filtered_df[col_name].isna()
                    
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
    
    # def _show_active_filters_summary(self) -> None:
    #     """
    #     Display a summary of currently active filters.
    #     """
    #     st.markdown("#### Active Filters")
        
    #     active_filters = []
        
    #     if self.search_term:
    #         active_filters.append(f"**Search**: '{self.search_term}'")
        
    #     for col_name, filter_config in self.column_filters.items():
    #         operation = filter_config.get('operation', 'equals')
            
    #         if operation in ["equals", "contains", "starts with", "ends with"]:
    #             values = filter_config.get('values', [])
    #             if values:
    #                 active_filters.append(f"**{col_name}** {operation}: {', '.join(map(str, values))}")
                    
    #         elif operation in ["greater than", "less than"]:
    #             threshold = filter_config.get('threshold')
    #             if threshold is not None:
    #                 active_filters.append(f"**{col_name}** {operation} {threshold:.4g}")
            
    #         elif operation == "between":
    #             lower_bound = filter_config.get('lower_bound')
    #             upper_bound = filter_config.get('upper_bound')
    #             if lower_bound is not None and upper_bound is not None:
    #                 active_filters.append(f"**{col_name}** between [{lower_bound:.4g}, {upper_bound:.4g}]")
        
    #     if active_filters:
    #         for filter_desc in active_filters:
    #             st.markdown(f"• {filter_desc}")
    #     else:
    #         st.markdown("*No active filters*")
    
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
        # Explicitly clear session state filters
        st.session_state.column_filters.clear()
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
