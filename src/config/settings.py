"""
Configuration settings for Dataframe2Visualization.
"""

from typing import Tuple, List

class AppConfig:
    """Application configuration constants."""
    
    # Image settings
    THUMBNAIL_SIZE: Tuple[int, int] = (100, 100)
    MAX_IMAGE_SIZE: Tuple[int, int] = (800, 600)
    SUPPORTED_IMAGE_FORMATS: List[str] = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif']
    
    # Table settings
    MAX_ROWS_PER_PAGE: int = 50
    DEFAULT_TABLE_HEIGHT: int = 600
    MAX_COLUMNS_TO_DISPLAY: int = 20
    
    # Performance settings
    IMAGE_CACHE_SIZE: int = 100
    BATCH_PROCESSING_SIZE: int = 1000
    MAX_IMAGE_FILE_SIZE_MB: int = 10
    
    # UI settings
    MODAL_WIDTH: int = 800
    MODAL_HEIGHT: int = 600
    THUMBNAIL_QUALITY: int = 85
    
    # Error handling
    MAX_ERROR_MESSAGE_LENGTH: int = 200
    SHOW_DETAILED_ERRORS: bool = False
    
    # Development settings
    DEBUG_MODE: bool = False
    LOG_LEVEL: str = "INFO"
    
    @classmethod
    def get_thumbnail_size(cls) -> Tuple[int, int]:
        """Get thumbnail size for images."""
        return cls.THUMBNAIL_SIZE
    
    @classmethod
    def get_max_image_size(cls) -> Tuple[int, int]:
        """Get maximum image size for display."""
        return cls.MAX_IMAGE_SIZE
    
    @classmethod
    def is_supported_format(cls, format_name: str) -> bool:
        """Check if image format is supported."""
        return format_name.lower() in cls.SUPPORTED_IMAGE_FORMATS
