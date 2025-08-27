"""
Dataframe2Visualization - Interactive DataFrame Display Tool

A Streamlit-based tool to display DataFrames containing both single values 
and 2D arrays/images with interactive image viewing capabilities.
"""

__version__ = "0.1.0"
__author__ = "Dataframe2Visualization Team"

from .core.dataframe_processor import DataFrameProcessor
from .core.image_handler import ImageHandler
from .core.data_validator import DataValidator
from .ui.table_display import InteractiveTableDisplay
from .ui.image_viewer import ImageModalViewer
from .ui.controls import TableControls

__all__ = [
    "DataFrameProcessor",
    "ImageHandler", 
    "DataValidator",
    "InteractiveTableDisplay",
    "ImageModalViewer",
    "TableControls"
]
