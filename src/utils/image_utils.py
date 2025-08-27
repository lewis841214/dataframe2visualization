"""
Utility functions for image processing.
"""

import numpy as np
from PIL import Image
import io
import base64
from typing import Any, Optional, Tuple, Union

def array_to_image(array: np.ndarray, normalize: bool = True) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array
        normalize: Whether to normalize values to 0-255 range
        
    Returns:
        PIL Image
    """
    if normalize:
        # Normalize to 0-255 range
        if array.min() != array.max():
            array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    
    # Handle different array shapes
    if len(array.shape) == 2:
        return Image.fromarray(array, mode='L')
    elif len(array.shape) == 3:
        if array.shape[2] == 3:
            return Image.fromarray(array, mode='RGB')
        elif array.shape[2] == 4:
            return Image.fromarray(array, mode='RGBA')
        elif array.shape[2] == 1:
            return Image.fromarray(array.squeeze(), mode='L')
    
    # Fallback
    return Image.fromarray(array, mode='RGB')

def image_to_base64(image: Image.Image, format: str = 'PNG', quality: int = 85) -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image
        format: Image format
        quality: Image quality (for JPEG)
        
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    
    if format.upper() == 'JPEG':
        image.save(buffer, format='JPEG', quality=quality)
    else:
        image.save(buffer, format=format)
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def create_thumbnail(image: Image.Image, size: Tuple[int, int], 
                    maintain_aspect: bool = True) -> Image.Image:
    """
    Create thumbnail from PIL Image.
    
    Args:
        image: PIL Image to resize
        size: Target size (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Thumbnail image
    """
    if maintain_aspect:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(size, Image.Resampling.LANCZOS)

def is_image_data(data: Any) -> bool:
    """
    Check if data is image-like.
    
    Args:
        data: Data to check
        
    Returns:
        True if data is image-like
    """
    if data is None:
        return False
    
    # Check for numpy arrays
    if isinstance(data, np.ndarray):
        return len(data.shape) in [2, 3] and data.size > 0
    
    # Check for PIL Image
    if isinstance(data, Image.Image):
        return True
    
    # Check for file paths
    if isinstance(data, str):
        import os
        return any(data.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'])
    
    return False

def get_image_info(image: Image.Image) -> dict:
    """
    Get information about an image.
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with image information
    """
    return {
        'size': image.size,
        'mode': image.mode,
        'format': getattr(image, 'format', None),
        'width': image.width,
        'height': image.height
    }

def convert_image_mode(image: Image.Image, target_mode: str = 'RGB') -> Image.Image:
    """
    Convert image to target mode.
    
    Args:
        image: PIL Image
        target_mode: Target color mode
        
    Returns:
        Converted image
    """
    if image.mode != target_mode:
        return image.convert(target_mode)
    return image
