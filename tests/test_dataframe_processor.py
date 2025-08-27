"""
Tests for DataFrameProcessor class.
"""

import pytest
import pandas as pd
import numpy as np
from src.core.dataframe_processor import DataFrameProcessor

class TestDataFrameProcessor:
    """Test cases for DataFrameProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DataFrameProcessor()
    
    def test_process_dataframe_basic(self):
        """Test basic DataFrame processing."""
        # Create simple DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        
        result = self.processor.process_dataframe(df)
        
        assert 'processed_data' in result
        assert 'column_metadata' in result
        assert 'original_shape' in result
        assert result['original_shape'] == (3, 3)
    
    def test_process_dataframe_with_images(self):
        """Test DataFrame processing with image data."""
        # Create DataFrame with numpy arrays (images)
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Image1', 'Image2'],
            'image_data': [
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ]
        })
        
        result = self.processor.process_dataframe(df)
        
        assert 'processed_data' in result
        assert 'column_metadata' in result
        
        # Check if image column is detected
        image_columns = self.processor.get_image_columns()
        assert 'image_data' in image_columns
        
        # Check image count
        total_images = self.processor.get_image_count()
        assert total_images == 2
    
    def test_get_display_dataframe(self):
        """Test getting display-ready DataFrame."""
        # Process a DataFrame first
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob'],
            'image_data': [
                np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8),
                None
            ]
        })
        
        self.processor.process_dataframe(df)
        display_df = self.processor.get_display_dataframe()
        
        assert isinstance(display_df, pd.DataFrame)
        assert display_df.shape == (2, 3)
    
    def test_cache_management(self):
        """Test image cache management."""
        # Process DataFrame with images
        df = pd.DataFrame({
            'image_data': [np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)]
        })
        
        self.processor.process_dataframe(df)
        
        # Check cache info
        cache_info = self.processor.get_cache_info()
        assert 'cache_size' in cache_info
        assert 'max_cache_size' in cache_info
        
        # Clear cache
        self.processor.clear_cache()
        cache_info_after = self.processor.get_cache_info()
        assert cache_info_after['cache_size'] == 0
