"""
DataFrame processing module for Dataframe2Visualization.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Union, Optional
from .data_validator import DataValidator
from .image_handler import ImageHandler
from ..config.settings import AppConfig

class DataFrameProcessor:
    """Processes DataFrames with mixed data types including images."""
    
    def __init__(self):
        """Initialize the DataFrameProcessor."""
        self.validator = DataValidator()
        self.image_handler = ImageHandler()
        self.processed_data = {}
        self.column_metadata = {}
    
    def process_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process the input DataFrame and prepare it for display.
        
        Args:
            df: Input DataFrame to process
            
        Returns:
            Dict containing processed data and metadata
        """
        # Validate input
        if not self.validator.validate_dataframe(df):
            validation_summary = self.validator.get_validation_summary()
            raise ValueError(f"Invalid DataFrame: {validation_summary['errors']}")
        
        # Process each column
        processed_columns = {}
        column_info = {}
        
        for col_name in df.columns:
            col_data = df[col_name]
            processed_col, col_meta = self._process_column(col_name, col_data)
            processed_columns[col_name] = processed_col
            column_info[col_name] = col_meta
        
        # Store processed data
        self.processed_data = processed_columns
        self.column_metadata = column_info
        
        return {
            'processed_data': processed_columns,
            'column_metadata': column_info,
            'original_shape': df.shape,
            'validation_summary': self.validator.get_validation_summary()
        }
    
    def _process_column(self, col_name: str, col_data: pd.Series) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Process individual column data.
        
        Args:
            col_name: Column name
            col_data: Column data as pandas Series
            
        Returns:
            Tuple of (processed_data, metadata)
        """
        column_metadata = {
            'name': col_name,
            'dtype': str(col_data.dtype),
            'contains_images': False,
            'image_count': 0,
            'total_count': len(col_data),
            'null_count': col_data.isnull().sum()
        }
        
        processed_items = []
        
        for idx, item in enumerate(col_data):
            processed_item, item_meta = self._process_item(item, f"{col_name}_{idx}")
            processed_items.append(processed_item)
            
            if item_meta.get('is_image', False):
                column_metadata['contains_images'] = True
                column_metadata['image_count'] += 1
        
        return processed_items, column_metadata
    
    def _process_item(self, item: Any, item_key: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Process individual item in a column.
        
        Args:
            item: Item to process
            item_key: Unique key for the item
            
        Returns:
            Tuple of (processed_item, metadata)
        """
        item_metadata = {
            'is_image': False,
            'original_type': type(item).__name__,
            'processed_type': None,
            'error': None
        }
        
        try:
            # Handle None values first
            if item is None:
                return None, item_metadata
            
            # Handle numpy arrays specially to avoid pd.isna issues
            if isinstance(item, np.ndarray):
                # Check if array is image data
                is_image = self.validator._is_image_data(item)
                if is_image:
                    processed_item = self._process_image_item(item, item_key)
                    item_metadata['is_image'] = True
                    item_metadata['processed_type'] = 'image'
                    return processed_item, item_metadata
                else:
                    # Handle non-image arrays
                    processed_item = self._process_non_image_item(item)
                    item_metadata['processed_type'] = type(processed_item).__name__
                    return processed_item, item_metadata
            
            # For other types, use pandas isna
            try:
                if pd.isna(item):
                    return None, item_metadata
            except (ValueError, TypeError):
                # If pd.isna fails, assume it's not None
                pass
            
            # Check if item is image data
            is_image = self.validator._is_image_data(item)
            if is_image:
                processed_item = self._process_image_item(item, item_key)
                item_metadata['is_image'] = True
                item_metadata['processed_type'] = 'image'
                return processed_item, item_metadata
            
            else:
                # Handle non-image data
                processed_item = self._process_non_image_item(item)
                item_metadata['processed_type'] = type(processed_item).__name__
                return processed_item, item_metadata
                
        except Exception as e:
            item_metadata['error'] = str(e)
            return f"Error: {str(e)}", item_metadata
    
    def _process_image_item(self, item: Any, item_key: str) -> Dict[str, Any]:
        """
        Process image item and create display-ready data.
        
        Args:
            item: Image item to process
            item_key: Unique key for the item
            
        Returns:
            Dict containing processed image data
        """
        try:
            # Convert to PIL Image
            pil_image = self.image_handler.convert_to_pil_image(item)
            if pil_image is None:
                return {'error': 'Failed to convert to image', 'type': 'error'}
            
            # Normalize image
            pil_image = self.image_handler.normalize_image(pil_image)
            
            # Create thumbnail
            thumbnail = self.image_handler.get_cached_thumbnail(item_key, pil_image)
            
            # Convert to base64 for display
            thumbnail_b64 = self.image_handler.image_to_base64(thumbnail)
            
            return {
                'type': 'image',
                'thumbnail': thumbnail_b64,
                'original_image': pil_image,
                'thumbnail_size': thumbnail.size,
                'original_size': pil_image.size,
                'key': item_key
            }
        except Exception as e:
            return {'error': f'Image processing error: {str(e)}', 'type': 'error'}
    
    def _process_non_image_item(self, item: Any) -> Any:
        """
        Process non-image item.
        
        Args:
            item: Non-image item to process
            
        Returns:
            Processed item
        """
        # Handle numpy arrays that aren't images
        if isinstance(item, np.ndarray):
            if item.size <= 100:  # Small arrays can be displayed as text
                return str(item.tolist())
            else:
                return f"Array({item.shape}, {item.dtype})"
        
        # Handle other data types
        return item
    
    def get_processed_data(self) -> Dict[str, Any]:
        """Get the processed data."""
        return self.processed_data
    
    def get_column_metadata(self) -> Dict[str, Any]:
        """Get column metadata."""
        return self.column_metadata
    
    def get_image_columns(self) -> List[str]:
        """Get list of columns containing images."""
        return [col for col, meta in self.column_metadata.items() 
                if meta.get('contains_images', False)]
    
    def get_image_count(self) -> int:
        """Get total number of images across all columns."""
        total_images = 0
        for meta in self.column_metadata.values():
            total_images += meta.get('image_count', 0)
        return total_images
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get image cache information."""
        return self.image_handler.get_cache_info()
    
    def clear_cache(self) -> None:
        """Clear the image cache."""
        self.image_handler.clear_cache()
    
    def get_display_dataframe(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Get DataFrame ready for display with processed data.
        
        Args:
            max_rows: Maximum number of rows to display
            
        Returns:
            DataFrame ready for display
        """
        if not self.processed_data:
            raise ValueError("No processed data available. Call process_dataframe() first.")
        
        # Create display DataFrame
        display_data = {}
        for col_name, col_data in self.processed_data.items():
            display_data[col_name] = [self._prepare_for_display(item) for item in col_data]
        
        df = pd.DataFrame(display_data)
        
        # Limit rows if specified
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
        
        return df
    
    def _prepare_for_display(self, item: Any) -> Any:
        """
        Prepare item for display in the table.
        
        Args:
            item: Item to prepare
            
        Returns:
            Display-ready item
        """
        if isinstance(item, dict) and item.get('type') == 'image':
            return item['thumbnail']  # Return base64 string for display
        elif isinstance(item, dict) and item.get('type') == 'error':
            return item['error']
        else:
            return item
