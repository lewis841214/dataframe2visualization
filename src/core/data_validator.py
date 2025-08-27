"""
Data validation module for Dataframe2Visualization.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Union, Optional
from ..config.settings import AppConfig

class DataValidator:
    """Validates DataFrame structure and data integrity."""
    
    def __init__(self):
        """Initialize the DataValidator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate the input DataFrame.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()
        
        if not isinstance(df, pd.DataFrame):
            self.errors.append("Input must be a pandas DataFrame")
            return False
        
        if df.empty:
            self.errors.append("DataFrame cannot be empty")
            return False
        
        if df.shape[1] > AppConfig.MAX_COLUMNS_TO_DISPLAY:
            self.warnings.append(f"DataFrame has {df.shape[1]} columns, which may impact performance")
        
        if df.shape[0] > AppConfig.BATCH_PROCESSING_SIZE:
            self.warnings.append(f"DataFrame has {df.shape[0]} rows, consider using pagination")
        
        return len(self.errors) == 0
    
    def validate_column_data(self, column_data: pd.Series) -> Dict[str, Any]:
        """
        Validate data in a specific column.
        
        Args:
            column_data: Pandas Series containing column data
            
        Returns:
            Dict containing validation results
        """
        validation_result = {
            'is_valid': True,
            'data_type': str(column_data.dtype),
            'contains_images': False,
            'image_count': 0,
            'error_count': 0,
            'warnings': []
        }
        
        try:
            # Check for image-like data
            for item in column_data:
                if self._is_image_data(item):
                    validation_result['contains_images'] = True
                    validation_result['image_count'] += 1
                    
                    # Validate individual image
                    if not self._validate_image_item(item):
                        validation_result['error_count'] += 1
                        validation_result['is_valid'] = False
                        
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['warnings'].append(f"Error processing column: {str(e)}")
        
        return validation_result
    
    def _is_image_data(self, item: Any) -> bool:
        """
        Check if an item is image data.
        
        Args:
            item: Item to check
            
        Returns:
            bool: True if item is image data
        """
        if item is None:
            return False
        
        # Handle numpy arrays specially to avoid pd.isna issues
        if isinstance(item, np.ndarray):
            # Check if array is valid (not empty, reasonable dimensions)
            return len(item.shape) in [2, 3] and item.size > 0
        
        # For other types, use pandas isna
        try:
            if pd.isna(item):
                return False
        except (ValueError, TypeError):
            # If pd.isna fails, assume it's not None
            pass
        
        # Check for PIL Image
        try:
            from PIL import Image
            if isinstance(item, Image.Image):
                return True
        except ImportError:
            pass
        
        # Check for file paths
        if isinstance(item, str):
            return any(item.lower().endswith(ext) for ext in AppConfig.SUPPORTED_IMAGE_FORMATS)
        
        return False
    
    def _validate_image_item(self, item: Any) -> bool:
        """
        Validate individual image item.
        
        Args:
            item: Image item to validate
            
        Returns:
            bool: True if valid image
        """
        try:
            if isinstance(item, np.ndarray):
                # Check array dimensions and data type
                if len(item.shape) not in [2, 3]:
                    return False
                
                # Check for reasonable size
                if item.size > AppConfig.MAX_IMAGE_SIZE[0] * AppConfig.MAX_IMAGE_SIZE[1] * 3:
                    return False
                
                # Check data type
                if not np.issubdtype(item.dtype, np.number):
                    return False
                    
            elif isinstance(item, str):
                # Check file path
                import os
                if not os.path.exists(item):
                    return False
                
                # Check file size
                file_size_mb = os.path.getsize(item) / (1024 * 1024)
                if file_size_mb > AppConfig.MAX_IMAGE_FILE_SIZE_MB:
                    return False
                    
        except Exception:
            return False
        
        return True
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation results.
        
        Returns:
            Dict containing validation summary
        """
        return {
            'is_valid': len(self.errors) == 0,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy()
        }
    
    def add_error(self, error_message: str) -> None:
        """Add an error message."""
        if len(error_message) > AppConfig.MAX_ERROR_MESSAGE_LENGTH:
            error_message = error_message[:AppConfig.MAX_ERROR_MESSAGE_LENGTH] + "..."
        self.errors.append(error_message)
    
    def add_warning(self, warning_message: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning_message)
