"""
Image handling module for Dataframe2Visualization.
"""

import numpy as np
from PIL import Image
import io
import base64
from typing import Any, Optional, Tuple, Union
from ..config.settings import AppConfig

class ImageHandler:
    """Handles image processing and thumbnail generation."""
    
    def __init__(self):
        """Initialize the ImageHandler."""
        self._thumbnail_cache = {}
        self._max_cache_size = AppConfig.IMAGE_CACHE_SIZE
    
    def convert_to_pil_image(self, data: Any) -> Optional[Image.Image]:
        """
        Convert various data types to PIL Image.
        
        Args:
            data: Input data (numpy array, PIL Image, file path, etc.)
            
        Returns:
            PIL Image or None if conversion fails
        """
        try:
            if isinstance(data, Image.Image):
                return data.copy()
            
            elif isinstance(data, np.ndarray):
                return self._numpy_to_pil(data)
            
            elif isinstance(data, str):
                return self._file_path_to_pil(data)
            
            elif isinstance(data, bytes):
                return Image.open(io.BytesIO(data))
            
            else:
                return None
                
        except Exception as e:
            if AppConfig.DEBUG_MODE:
                print(f"Error converting to PIL Image: {e}")
            return None
    
    def _numpy_to_pil(self, array: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image.
        
        Args:
            array: Numpy array
            
        Returns:
            PIL Image
        """
        # Handle different array shapes
        if len(array.shape) == 2:
            # Grayscale image
            if array.dtype != np.uint8:
                # Normalize to 0-255 range
                min_val = float(array.min())
                max_val = float(array.max())
                if min_val < 0 or max_val > 255:
                    array = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    array = array.astype(np.uint8)
            return Image.fromarray(array, mode='L')
        
        elif len(array.shape) == 3:
            # Color image
            if array.shape[2] == 3:
                # RGB
                if array.dtype != np.uint8:
                    min_val = float(array.min())
                    max_val = float(array.max())
                    array = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                return Image.fromarray(array, mode='RGB')
            elif array.shape[2] == 4:
                # RGBA
                if array.dtype != np.uint8:
                    min_val = float(array.min())
                    max_val = float(array.max())
                    array = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                return Image.fromarray(array, mode='RGBA')
            elif array.shape[2] == 1:
                # Single channel
                array = array.squeeze()
                if array.dtype != np.uint8:
                    min_val = float(array.min())
                    max_val = float(array.max())
                    array = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                return Image.fromarray(array, mode='L')
        
        # Fallback: convert to RGB
        if array.dtype != np.uint8:
            min_val = float(array.min())
            max_val = float(array.max())
            array = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return Image.fromarray(array, mode='RGB')
    
    def _file_path_to_pil(self, file_path: str) -> Image.Image:
        """
        Load image from file path.
        
        Args:
            file_path: Path to image file
            
        Returns:
            PIL Image
        """
        return Image.open(file_path)
    
    def create_thumbnail(self, image: Image.Image, size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Create thumbnail from PIL Image.
        
        Args:
            image: PIL Image to resize
            size: Target size (width, height), uses default if None
            
        Returns:
            Thumbnail image
        """
        if size is None:
            size = AppConfig.get_thumbnail_size()
        
        # Create thumbnail while maintaining aspect ratio
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image
    
    def image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=AppConfig.THUMBNAIL_QUALITY)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    
    def get_cached_thumbnail(self, key: str, image: Image.Image) -> Image.Image:
        """
        Get or create cached thumbnail.
        
        Args:
            key: Cache key
            image: PIL Image
            
        Returns:
            Thumbnail image
        """
        if key in self._thumbnail_cache:
            return self._thumbnail_cache[key]
        
        # Create thumbnail and cache it
        thumbnail = self.create_thumbnail(image)
        self._thumbnail_cache[key] = thumbnail
        
        # Manage cache size
        if len(self._thumbnail_cache) > self._max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._thumbnail_cache))
            del self._thumbnail_cache[oldest_key]
        
        return thumbnail
    
    def clear_cache(self) -> None:
        """Clear the thumbnail cache."""
        self._thumbnail_cache.clear()
    
    def get_cache_info(self) -> dict:
        """Get cache information."""
        return {
            'cache_size': len(self._thumbnail_cache),
            'max_cache_size': self._max_cache_size,
            'cache_usage_percent': (len(self._thumbnail_cache) / self._max_cache_size) * 100
        }
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int], maintain_aspect: bool = True) -> Image.Image:
        """
        Resize image to specified dimensions.
        
        Args:
            image: PIL Image to resize
            size: Target size (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            # Calculate new size maintaining aspect ratio
            img_ratio = image.width / image.height
            target_ratio = size[0] / size[1]
            
            if img_ratio > target_ratio:
                # Image is wider than target
                new_width = size[0]
                new_height = int(size[0] / img_ratio)
            else:
                # Image is taller than target
                new_height = size[1]
                new_width = int(size[1] * img_ratio)
            
            size = (new_width, new_height)
        
        return image.resize(size, Image.Resampling.LANCZOS)
    
    def normalize_image(self, image: Image.Image) -> Image.Image:
        """
        Normalize image for consistent display.
        
        Args:
            image: PIL Image to normalize
            
        Returns:
            Normalized image
        """
        # Convert to RGB if necessary
        if image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')
        
        return image
