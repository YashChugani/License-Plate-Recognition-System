import cv2
import numpy as np

def extract_and_enhance(image, contour):
    """
    Crops the license plate from the main image and enhances its contrast.
    """
    if contour is None:
        return None
        
    # 1. Get the bounding box coordinates of the contour
    (x, y, w, h) = cv2.boundingRect(contour)
    
    # Add a tiny bit of padding (buffer) around the box just in case the edges were cut off
    y_start = max(0, y - 5)
    y_end = min(image.shape[0], y + h + 5)
    x_start = max(0, x - 5)
    x_end = min(image.shape[1], x + w + 5)
    
    # 2. Crop the image using Numpy slicing
    cropped_plate = image[y_start:y_end, x_start:x_end]
    
    if cropped_plate.size == 0:
        return None
        
    # 3. Enhance the cropped plate for OCR
    # Convert to grayscale
    gray_crop = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    
    # Resize it to make it larger (helps OCR engines read small text)
    enhanced_plate = cv2.resize(gray_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian Blur to remove pixelated noise from resizing
    enhanced_plate = cv2.GaussianBlur(enhanced_plate, (5, 5), 0)
    
    return cropped_plate, enhanced_plate