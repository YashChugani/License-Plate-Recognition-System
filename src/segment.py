import cv2
import numpy as np

def segment_characters(enhanced_gray):
    """
    Segments individual characters from the cropped license plate 
    using binary thresholding and geometric filtering.
    """
    # 1. Binarize the image
    # Indian plates are dark text on light backgrounds. We invert it (THRESH_BINARY_INV) 
    # so the characters become solid white blobs on a black background.
    _, thresh = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # 2. Clean up noise (remove tiny white specks like dirt or plate bolts)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Find contours of the white blobs (our potential characters)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_candidates = []
    plate_h, plate_w = enhanced_gray.shape
    
    # 4. Filter contours based on character geometry
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio = w / float(h)
        height_ratio = h / float(plate_h)
        
        # A typical character is taller than it is wide (aspect ratio < 1.0)
        # and should take up a significant portion of the plate's height (30% to 90%)
        if 0.1 <= aspect_ratio <= 1.0 and 0.3 <= height_ratio <= 0.9:
            char_candidates.append((x, y, w, h))
            
    # 5. Sort the characters from left to right based on their X coordinate
    char_candidates = sorted(char_candidates, key=lambda b: b[0])
    
    return thresh, char_candidates