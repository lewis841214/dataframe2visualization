"""
Sample data generation for Dataframe2Visualization.
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from typing import Dict, Any

def create_sample_dataframe(data_type: str) -> pd.DataFrame:
    """
    Create sample DataFrame based on specified type.
    
    Args:
        data_type: Type of sample data to create
        
    Returns:
        Sample DataFrame
    """
    if data_type == "Random Images":
        return _create_random_images_dataframe()
    elif data_type == "Mixed Data Types":
        return _create_mixed_data_dataframe()
    elif data_type == "Large Dataset":
        return _create_large_dataset_dataframe()
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def _create_random_images_dataframe() -> pd.DataFrame:
    """Create DataFrame with random generated images."""
    data = []
    
    for i in range(10):
        # Create random image
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Create metadata
        row = {
            'id': i + 1,
            'name': f'Image_{i+1:03d}',
            'category': np.random.choice(['Nature', 'Abstract', 'Pattern']),
            'size': f"{img_array.shape[0]}x{img_array.shape[1]}",
            'image_data': img_array,
            'score': round(np.random.uniform(0, 1), 3)
        }
        data.append(row)
    
    return pd.DataFrame(data)

def _create_mixed_data_dataframe() -> pd.DataFrame:
    """Create DataFrame with mixed data types including images."""
    data = []
    
    # Sample text data
    names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry']
    categories = ['A', 'B', 'C']
    
    for i in range(8):
        # Create different types of data
        if i % 3 == 0:
            # Create a simple pattern image
            img_array = _create_pattern_image(32, 32, i)
            image_data = img_array
        elif i % 3 == 1:
            # Create a gradient image
            img_array = _create_gradient_image(32, 32, i)
            image_data = img_array
        else:
            # No image for this row
            image_data = None
        
        row = {
            'id': i + 1,
            'name': names[i],
            'category': categories[i % len(categories)],
            'age': np.random.randint(20, 60),
            'score': round(np.random.uniform(0, 100), 2),
            'active': np.random.choice([True, False]),
            'image_data': image_data,
            'description': f'This is a sample description for {names[i]} with some additional text to make it longer.'
        }
        data.append(row)
    
    return pd.DataFrame(data)

def _create_large_dataset_dataframe() -> pd.DataFrame:
    """Create a larger dataset for performance testing."""
    data = []
    
    for i in range(100):
        # Create small thumbnail images for performance
        img_array = _create_simple_image(16, 16, i)
        
        row = {
            'id': i + 1,
            'name': f'Item_{i+1:03d}',
            'category': f'Cat_{i % 10}',
            'value': round(np.random.uniform(0, 1000), 2),
            'count': np.random.randint(1, 100),
            'image_data': img_array,
            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
        }
        data.append(row)
    
    return pd.DataFrame(data)

def _create_pattern_image(width: int, height: int, seed: int) -> np.ndarray:
    """Create a simple pattern image."""
    np.random.seed(seed)
    
    # Create a simple pattern
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some random patterns
    for y in range(height):
        for x in range(width):
            if (x + y) % 4 == 0:
                img_array[y, x] = [255, 0, 0]  # Red
            elif (x + y) % 4 == 1:
                img_array[y, x] = [0, 255, 0]  # Green
            elif (x + y) % 4 == 2:
                img_array[y, x] = [0, 0, 255]  # Blue
            else:
                img_array[y, x] = [255, 255, 0]  # Yellow
    
    return img_array

def _create_gradient_image(width: int, height: int, seed: int) -> np.ndarray:
    """Create a gradient image."""
    np.random.seed(seed)
    
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient
    for y in range(height):
        for x in range(width):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = int(255 * (x + y) / (width + height))
            img_array[y, x] = [r, g, b]
    
    return img_array

def _create_simple_image(width: int, height: int, seed: int) -> np.ndarray:
    """Create a very simple image for performance testing."""
    np.random.seed(seed)
    
    # Create a simple colored square
    color = np.random.randint(0, 255, 3)
    img_array = np.full((height, width, 3), color, dtype=np.uint8)
    
    # Add a simple border
    img_array[0, :] = [0, 0, 0]  # Top border
    img_array[-1, :] = [0, 0, 0]  # Bottom border
    img_array[:, 0] = [0, 0, 0]  # Left border
    img_array[:, -1] = [0, 0, 0]  # Right border
    
    return img_array

def create_text_image(text: str, width: int = 200, height: int = 100) -> np.ndarray:
    """
    Create an image with text for testing.
    
    Args:
        text: Text to display
        width: Image width
        height: Image height
        
    Returns:
        Image as numpy array
    """
    # Create a PIL image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Calculate text position (center)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill='black', font=font)
    
    # Convert to numpy array
    return np.array(img)

def create_sample_csv_data() -> str:
    """Create sample CSV data for testing."""
    data = """id,name,category,value,image_path
1,Image_001,Nature,85.5,images/nature_001.jpg
2,Image_002,Abstract,92.3,images/abstract_002.jpg
3,Image_003,Pattern,78.9,images/pattern_003.jpg
4,Image_004,Nature,91.2,images/nature_004.jpg
5,Image_005,Abstract,87.6,images/abstract_005.jpg"""
    
    return data

def create_sample_json_data() -> str:
    """Create sample JSON data for testing."""
    data = {
        "data": [
            {"id": 1, "name": "Image_001", "category": "Nature", "value": 85.5, "image_path": "images/nature_001.jpg"},
            {"id": 2, "name": "Image_002", "category": "Abstract", "value": 92.3, "image_path": "images/abstract_002.jpg"},
            {"id": 3, "name": "Image_003", "category": "Pattern", "value": 78.9, "image_path": "images/pattern_003.jpg"},
            {"id": 4, "name": "Image_004", "category": "Nature", "value": 91.2, "image_path": "images/nature_004.jpg"},
            {"id": 5, "name": "Image_005", "category": "Abstract", "value": 87.6, "image_path": "images/abstract_005.jpg"}
        ]
    }
    
    import json
    return json.dumps(data, indent=2)
