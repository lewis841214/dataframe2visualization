"""
Image modal viewer module for Dataframe2Visualization.
"""

import streamlit as st
from PIL import Image
from typing import Any, Dict, Optional, Tuple
from ..config.settings import AppConfig

class ImageModalViewer:
    """Displays expanded images in a modal with navigation controls."""
    
    def __init__(self):
        """Initialize the ImageModalViewer."""
        self.current_image_key = None
        self.current_image_data = None
    
    def show_image_modal(self, image_data: Dict[str, Any], image_key: str) -> None:
        """
        Display image in a modal/expanded view.
        
        Args:
            image_data: Image data dictionary
            image_key: Unique key for the image
        """
        if not image_data or image_data.get('type') != 'image':
            st.error("Invalid image data")
            return
        
        # Store current image state
        self.current_image_key = image_key
        self.current_image_data = image_data
        
        # Create modal-like container
        with st.container():
            st.markdown("---")
            st.markdown("### Image Viewer")
            
            # Display image information
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Image Key:** {image_key}")
            
            with col2:
                if 'original_size' in image_data:
                    st.markdown(f"**Size:** {image_data['original_size'][0]} × {image_data['original_size'][1]}")
            
            with col3:
                if 'thumbnail_size' in image_data:
                    st.markdown(f"**Thumbnail:** {image_data['thumbnail_size'][0]} × {image_data['thumbnail_size'][1]}")
            
            # Display the full-size image
            if 'original_image' in image_data:
                original_image = image_data['original_image']
                
                # Resize image for display if too large
                display_image = self._prepare_image_for_display(original_image)
                
                # Center the image
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(display_image, caption=f"Full-size image: {image_key}", 
                            use_column_width=True)
            
            # Navigation and control buttons
            self._show_image_controls(image_data)
            
            st.markdown("---")
    
    def _prepare_image_for_display(self, image: Image.Image) -> Image.Image:
        """
        Prepare image for display by resizing if necessary.
        
        Args:
            image: PIL Image to prepare
            
        Returns:
            Prepared PIL Image
        """
        max_width, max_height = AppConfig.get_max_image_size()
        
        if image.width > max_width or image.height > max_height:
            # Calculate new size maintaining aspect ratio
            img_ratio = image.width / image.height
            target_ratio = max_width / max_height
            
            if img_ratio > target_ratio:
                # Image is wider than target
                new_width = max_width
                new_height = int(max_width / img_ratio)
            else:
                # Image is taller than target
                new_height = max_height
                new_width = int(max_height * img_ratio)
            
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def _show_image_controls(self, image_data: Dict[str, Any]) -> None:
        """
        Show image control buttons and options.
        
        Args:
            image_data: Image data dictionary
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("← Previous", key="prev_img"):
                self._navigate_image(-1)
        
        with col2:
            if st.button("Next →", key="next_img"):
                self._navigate_image(1)
        
        with col3:
            if st.button("Download", key="download_img"):
                self._download_image(image_data)
        
        with col4:
            if st.button("Close", key="close_img"):
                self._close_image_viewer()
        
        # Additional controls
        st.markdown("---")
        
        # Zoom controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            zoom_level = st.slider("Zoom", min_value=0.5, max_value=3.0, 
                                 value=1.0, step=0.1, key="zoom_slider")
        
        with col2:
            if st.button("Reset Zoom", key="reset_zoom"):
                st.session_state.zoom_slider = 1.0
                st.rerun()
        
        with col3:
            if st.button("Fit to Screen", key="fit_screen"):
                st.session_state.zoom_slider = 1.0
                st.rerun()
    
    def _navigate_image(self, direction: int) -> None:
        """
        Navigate to next/previous image.
        
        Args:
            direction: 1 for next, -1 for previous
        """
        # This would be implemented based on the current image context
        # For now, just show a message
        if direction > 0:
            st.info("Next image functionality to be implemented")
        else:
            st.info("Previous image functionality to be implemented")
    
    def _download_image(self, image_data: Dict[str, Any]) -> None:
        """
        Download the current image.
        
        Args:
            image_data: Image data dictionary
        """
        if 'original_image' in image_data:
            original_image = image_data['original_image']
            
            # Convert to bytes for download
            import io
            img_buffer = io.BytesIO()
            original_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Create download button
            st.download_button(
                label="Download Image",
                data=img_buffer.getvalue(),
                file_name=f"image_{self.current_image_key}.png",
                mime="image/png"
            )
        else:
            st.error("No image data available for download")
    
    def _close_image_viewer(self) -> None:
        """Close the image viewer and return to table view."""
        # Clear current image state
        self.current_image_key = None
        self.current_image_data = None
        
        # Clear the modal display
        st.rerun()
    
    def is_image_open(self) -> bool:
        """Check if an image modal is currently open."""
        return self.current_image_key is not None
    
    def get_current_image_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently displayed image."""
        if self.current_image_key and self.current_image_data:
            return {
                'key': self.current_image_key,
                'data': self.current_image_data
            }
        return None
    
    def close_current_image(self) -> None:
        """Close the currently displayed image."""
        self._close_image_viewer()
