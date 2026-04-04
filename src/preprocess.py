import cv2
import imutils

def preprocess_image(image):
    """
    Resizes the image to a standard width to ensure consistent geometric 
    thresholds, converts to grayscale, and applies bilateral filtering.
    """
    # 1. Standardize resolution (crucial for datasets with mixed image sizes)
    image = imutils.resize(image, width=600)
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply Bilateral Filter to smooth the surface of the car but keep the plate edges sharp
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    return image, gray, bfilter