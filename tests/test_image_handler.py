"""
Tests for ImageHandler class.
"""

import pytest
import numpy as np
from PIL import Image
from src.core.image_handler import ImageHandler

class TestImageHandler:
    """Test cases for ImageHandler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ImageHandler()
    
    def test_convert_numpy_array_to_pil(self):
        """Test converting numpy array to PIL Image."""
        # Create test array
        test_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        # Convert to PIL Image
        pil_image = self.handler.convert_to_pil_image(test_array)
        
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (32, 32)
        assert pil_image.mode == 'RGB'
    
    def test_convert_grayscale_array(self):
        """Test converting grayscale numpy array."""
        # Create grayscale array
        test_array = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        
        # Convert to PIL Image
        pil_image = self.handler.convert_to_pil_image(test_array)
        
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (32, 32)
        assert pil_image.mode == 'L'
    
    def test_create_thumbnail(self):
        """Test thumbnail creation."""
        # Create test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Create thumbnail
        thumbnail = self.handler.create_thumbnail(test_image, (50, 50))
        
        assert isinstance(thumbnail, Image.Image)
        assert thumbnail.size[0] <= 50
        assert thumbnail.size[1] <= 50
    
    def test_image_to_base64(self):
        """Test converting image to base64."""
        # Create test image
        test_image = Image.new('RGB', (10, 10), color='blue')
        
        # Convert to base64
        base64_str = self.handler.image_to_base64(test_image)
        
        assert isinstance(base64_str, str)
        assert base64_str.startswith('data:image/png;base64,')
    
    def test_cache_functionality(self):
        """Test image caching."""
        # Create test image
        test_image = Image.new('RGB', (20, 20), color='green')
        
        # Get cached thumbnail
        thumbnail1 = self.handler.get_cached_thumbnail('test_key', test_image)
        thumbnail2 = self.handler.get_cached_thumbnail('test_key', test_image)
        
        # Both should be the same object (cached)
        assert thumbnail1 is thumbnail2
        
        # Check cache info
        cache_info = self.handler.get_cache_info()
        assert cache_info['cache_size'] > 0
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        # Add some images to cache
        test_image = Image.new('RGB', (20, 20), color='yellow')
        self.handler.get_cached_thumbnail('test_key', test_image)
        
        # Clear cache
        self.handler.clear_cache()
        
        # Check cache is empty
        cache_info = self.handler.get_cache_info()
        assert cache_info['cache_size'] == 0
