import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
import numpy as np
import cv2
from PIL import Image


def grid(images, nrow=2, padding=2):
    """
    Create a grid of images using numpy/cv2 operations.
    
    Args:
        images: List of numpy arrays (images) or PIL Images
        nrow: Number of images per row
        padding: Padding between images in pixels
    
    Returns:
        PIL Image of the grid resized to (1920, 1080)
    """
    if len(images) == 1:
        nrow = 1
    if not images:
        return None
    
    # Convert PIL Images to numpy arrays if needed
    np_images = []
    for img in images:
        if isinstance(img, Image.Image):
            np_images.append(np.array(img))
        else:
            np_images.append(img)
    
    # Ensure all images have the same number of channels
    # Convert grayscale to RGB if needed
    processed_images = []
    for img in np_images:
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        processed_images.append(img)
    
    # Get dimensions - find max height and width for uniform sizing
    heights = [img.shape[0] for img in processed_images]
    widths = [img.shape[1] for img in processed_images]
    max_height = max(heights)
    max_width = max(widths)
    
    # Resize all images to the same size (optional - maintains aspect ratio)
    uniform_images = []
    for img in processed_images:
        # Pad to max dimensions to maintain aspect ratio
        h, w = img.shape[:2]
        
        # Create a blank canvas with max dimensions
        canvas = np.zeros((max_height, max_width, 3), dtype=img.dtype)
        
        # Center the image on the canvas
        y_offset = (max_height - h) // 2
        x_offset = (max_width - w) // 2
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img
        
        uniform_images.append(canvas)
    
    # Calculate grid dimensions
    n_images = len(uniform_images)
    ncol = nrow
    nrows_needed = (n_images + ncol - 1) // ncol  # Ceiling division
    
    # Calculate total grid dimensions
    img_height, img_width = max_height, max_width
    total_height = nrows_needed * img_height + (nrows_needed + 1) * padding
    total_width = ncol * img_width + (ncol + 1) * padding
    
    # Create the grid canvas
    grid_canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 0  # White background
    
    # Place images in the grid
    for idx, img in enumerate(uniform_images):
        row = idx // ncol
        col = idx % ncol
        
        # Calculate position
        y_start = row * (img_height + padding) + padding
        x_start = col * (img_width + padding) + padding
        y_end = y_start + img_height
        x_end = x_start + img_width
        
        # Place the image
        grid_canvas[y_start:y_end, x_start:x_end] = img
    
    # Convert to PIL Image and resize
    grid_pil = Image.fromarray(grid_canvas)
    return grid_pil.resize((max_width, max_height))


def draw_tensor_grid(tensor, nrow=2, padding=2) -> Image.Image:
    """
    Draw a grid of images from a tensor.
    
    Args:
        tensor: A 4D tensor of shape (B, C, H, W)
        nrow: Number of images per row
        padding: Padding between images in pixels
    
    Returns:
        PIL Image of the grid
    """
    images = [
        (tensor[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        for i in range(tensor.shape[0])
    ]
    return grid(images, nrow=nrow, padding=padding)