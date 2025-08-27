# Dataframe2Visualization 🖼️

An interactive Streamlit-based tool for displaying DataFrames containing both single values and 2D arrays/images with interactive image viewing capabilities.

## Features ✨

- **Mixed Data Support**: Handle numbers, strings, 2D/3D numpy arrays, PIL images, and file paths
- **Image Thumbnails**: Display small versions of images in table cells
- **Interactive Image Viewing**: Click on thumbnails to show full-size images
- **Advanced Controls**: Search, filter, sort, and paginate your data
- **Export Options**: Download data in CSV, Excel, and JSON formats
- **Performance Optimized**: Image caching and efficient processing
- **Responsive Design**: Works on different screen sizes

## Installation 🚀

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Option 1: Virtual Environment (Recommended)

#### Quick Setup (Recommended)
```bash
# On macOS/Linux:
./setup.sh

# On Windows:
setup.bat
```

#### Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Deactivate virtual environment when done
deactivate
```

### Option 2: Direct Installation
```bash
pip install -r requirements.txt
```

### Optional Dependencies
For enhanced functionality, you can install additional packages:
```bash
pip install opencv-python scikit-image
```

## Quick Start 🏃‍♂️

### 1. Run the Application
```bash
# Activate virtual environment (if using one)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the Streamlit app
streamlit run app.py
```

### 2. Load Sample Data
- Choose from predefined sample datasets
- Or upload your own CSV, Excel, JSON, or Parquet file

### 3. Explore Your Data
- View mixed content in an interactive table
- Click on image thumbnails to see full-size images
- Use search and filter controls to find specific data
- Export your data in various formats

## Usage Examples 📖

### Basic Usage
```python
import pandas as pd
import numpy as np
from src.core.dataframe_processor import DataFrameProcessor

# Create DataFrame with mixed data
df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'image_data': [
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        None
    ]
})

# Process DataFrame
processor = DataFrameProcessor()
result = processor.process_dataframe(df)

# Get display-ready DataFrame
display_df = processor.get_display_dataframe()
```

### Advanced Usage
```python
from src.ui.table_display import InteractiveTableDisplay
from src.ui.controls import TableControls

# Initialize components
table_display = InteractiveTableDisplay()
table_controls = TableControls()

# Render controls and get filtered data
controls_result = table_controls.render_controls(df, column_metadata)

# Display interactive table
table_display.render_table(display_df, column_metadata, processed_data)
```

## Project Structure 📁

```
dataframe2visualization/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── README.md             # This file
├── .cursorrules          # Project specifications
├── src/                  # Source code
│   ├── core/            # Core functionality
│   │   ├── dataframe_processor.py
│   │   ├── image_handler.py
│   │   └── data_validator.py
│   ├── ui/              # User interface components
│   │   ├── table_display.py
│   │   ├── image_viewer.py
│   │   └── controls.py
│   ├── utils/           # Utility functions
│   │   ├── image_utils.py
│   │   └── streamlit_utils.py
│   ├── config/          # Configuration
│   │   └── settings.py
│   └── examples/        # Sample data and examples
│       └── sample_data.py
├── tests/               # Test files
└── examples/            # Example usage
```

## Configuration ⚙️

### App Settings
Modify `src/config/settings.py` to customize:
- Image thumbnail sizes
- Maximum image dimensions
- Cache sizes
- Table display options
- Performance parameters

### Key Configuration Options
```python
# Image settings
THUMBNAIL_SIZE = (100, 100)
MAX_IMAGE_SIZE = (800, 600)
SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif']

# Performance settings
IMAGE_CACHE_SIZE = 100
BATCH_PROCESSING_SIZE = 1000
```

## Supported Data Types 📊

### Text & Numbers
- Strings
- Integers
- Floats
- Booleans
- Dates/Timestamps

### Images
- **Numpy Arrays**: 2D (grayscale) and 3D (RGB/RGBA) arrays
- **PIL Images**: Direct PIL Image objects
- **File Paths**: Paths to image files
- **Bytes**: Raw image data

### File Formats
- **Input**: CSV, Excel (.xlsx, .xls), JSON, Parquet
- **Output**: CSV, Excel, JSON

## API Reference 📚

### Core Classes

#### DataFrameProcessor
Main class for processing DataFrames with mixed content.

```python
processor = DataFrameProcessor()
result = processor.process_dataframe(df)
display_df = processor.get_display_dataframe()
```

#### ImageHandler
Handles image processing and thumbnail generation.

```python
handler = ImageHandler()
pil_image = handler.convert_to_pil_image(numpy_array)
thumbnail = handler.create_thumbnail(pil_image)
```

#### InteractiveTableDisplay
Renders interactive tables with clickable images.

```python
display = InteractiveTableDisplay()
display.render_table(df, column_metadata, processed_data)
```

## Testing 🧪

Run tests using pytest:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_dataframe_processor.py
```

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Development Setup 🛠️

### Install Development Dependencies
```bash
# Create and activate virtual environment (if not already done)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies including development tools
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

### Code Formatting
```bash
black src/ tests/
```

### Linting
```bash
flake8 src/ tests/
```

### Type Checking
```bash
mypy src/
```

## Performance Tips 💡

- **Image Sizes**: Keep images reasonably sized (under 1MB) for better performance
- **Batch Processing**: Use pagination for large datasets
- **Cache Management**: Monitor cache usage and clear when needed
- **Memory**: Large image datasets may require significant memory

## Troubleshooting 🔧

### Common Issues

#### PyArrow Compatibility Issues
- If you encounter PyArrow errors with numpy arrays, the export functions automatically handle this
- Complex data types (numpy arrays, images) are converted to strings during export
- This ensures compatibility with CSV, Excel, and JSON formats

#### Images Not Displaying
- Check if image data is in supported format
- Verify image dimensions are reasonable
- Check console for error messages

#### Performance Issues
- Reduce image sizes
- Use pagination for large datasets
- Clear image cache periodically

#### Import Errors
- Ensure all dependencies are installed
- Check Python version compatibility
- Verify import paths

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Built with [Streamlit](https://streamlit.io/)
- Image processing powered by [Pillow (PIL)](https://python-pillow.org/)
- Data manipulation with [Pandas](https://pandas.pydata.org/)
- Numerical operations with [NumPy](https://numpy.org/)

## Support 💬

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review example code in the `examples/` directory

---

**Happy visualizing! 🎉**
