"""
Tests for UI components.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from src.ui.table_display import InteractiveTableDisplay
from src.ui.controls import TableControls

class TestTableDisplay:
    """Test cases for InteractiveTableDisplay."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.display = InteractiveTableDisplay()
    
    def test_initialization(self):
        """Test component initialization."""
        assert self.display.clicked_image_key is None
        assert self.display.clicked_image_data is None
    
    def test_get_clicked_image_info(self):
        """Test getting clicked image information."""
        # Initially should be None
        info = self.display.get_clicked_image_info()
        assert info is None
        
        # Set some data
        self.display.clicked_image_key = "test_key"
        self.display.clicked_image_data = {"test": "data"}
        
        info = self.display.get_clicked_image_info()
        assert info is not None
        assert info['key'] == "test_key"
        assert info['data'] == {"test": "data"}
    
    def test_clear_clicked_image(self):
        """Test clearing clicked image state."""
        # Set some data
        self.display.clicked_image_key = "test_key"
        self.display.clicked_image_data = {"test": "data"}
        
        # Clear
        self.display.clear_clicked_image()
        
        assert self.display.clicked_image_key is None
        assert self.display.clicked_image_data is None

class TestTableControls:
    """Test cases for TableControls."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.controls = TableControls()
    
    def test_initialization(self):
        """Test component initialization."""
        assert self.controls.current_page == 1
        assert self.controls.items_per_page > 0
        assert self.controls.search_term == ""
        assert self.controls.column_filters == {}
    
    def test_reset_controls(self):
        """Test resetting controls to default values."""
        # Change some values
        self.controls.current_page = 5
        self.controls.search_term = "test"
        self.controls.column_filters = {"col": ["val"]}
        
        # Reset
        self.controls.reset_controls()
        
        assert self.controls.current_page == 1
        assert self.controls.search_term == ""
        assert self.controls.column_filters == {}
    
    def test_get_controls_state(self):
        """Test getting controls state."""
        state = self.controls._get_controls_state()
        
        assert 'current_page' in state
        assert 'items_per_page' in state
        assert 'search_term' in state
        assert 'column_filters' in state
        assert 'sort_column' in state
        assert 'sort_ascending' in state
    
    def test_get_filtered_row_count(self):
        """Test getting filtered row count."""
        # Create test DataFrame
        df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': [1, 2, 3]
        })
        
        # No filters applied
        count = self.controls.get_filtered_row_count(df)
        assert count == 3
        
        # Apply search filter
        self.controls.search_term = "a"
        count = self.controls.get_filtered_row_count(df)
        assert count == 1
