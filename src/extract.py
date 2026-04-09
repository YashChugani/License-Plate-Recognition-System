import cv2
import numpy as np

def extract_and_enhance(image, contour):
    """
    Crops the license plate from the main image and enhances its contrast
    using bicubic interpolation and spatial smoothing.
    """
    if contour is None:
        return None
        
    # 1. Geometric Extraction with Padding Heuristics
    (x, y, w, h) = cv2.boundingRect(contour)
    
    # Adding a 5-pixel buffer to prevent character clipping at the borders
    y_start = max(0, y - 5)
    y_end = min(image.shape[0], y + h + 5)
    x_start = max(0, x - 5)
    x_end = min(image.shape[1], x + w + 5)
    
    # Crop using Numpy matrix slicing
    cropped_plate = image[y_start:y_end, x_start:x_end]
    if cropped_plate.size == 0:
        return None
        
    # 2. Dimensionality Reduction
    gray_crop = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    
    # 3. Spatial Resolution Enhancement
    # Upscaling by 200% using Bicubic Interpolation (cv2.INTER_CUBIC). 
    # This mathematically estimates missing pixel values using the 16 nearest neighbors, 
    # which preserves text edges much better than linear interpolation.
    enhanced_plate = cv2.resize(gray_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 4. Spatial Filtering (High-Frequency Noise Reduction)
    # Applying a 5x5 Gaussian Blur kernel to smooth out the jagged, 
    # pixelated artifacts introduced during the upscaling process.
    enhanced_plate = cv2.GaussianBlur(enhanced_plate, (5, 5), 0)
    
    return cropped_plate, enhanced_plate