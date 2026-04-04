import cv2
import imutils
import numpy as np

def get_plate_candidates(bfilter_img):
    """
    Finds plate candidates using Auto-Canny, but includes a maximum area 
    constraint to prevent cropping the entire vehicle.
    """
    v = np.median(bfilter_img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    edged = cv2.Canny(bfilter_img, lower, upper)
    edged = cv2.dilate(edged, None, iterations=1)
    
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    
    candidates = []
    
    # Calculate the total area of the image
    img_h, img_w = bfilter_img.shape
    total_image_area = img_h * img_w
    
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (w, h) = rect[1]
        if w < h:
            w, h = h, w
        if h == 0: continue
            
        aspect_ratio = w / float(h)
        x, y, bw, bh = cv2.boundingRect(contour)
        contour_area = bw * bh
        
        # ADDED RULE: contour_area must be less than 15% of the image (0.15)
        # This explicitly filters out the bounding box of the entire car.
        # If the image is smaller than 300,000 pixels (roughly 600x500), 
        # it's likely a tight crop. Allow up to 80%. Otherwise, strict 20%.
        max_area_ratio = 0.80 if total_image_area < 300000 else 0.20
        if 1.2 <= aspect_ratio <= 7.0 and bw > 40 and bh > 12 and contour_area < (total_image_area * max_area_ratio):
            candidates.append((x, y, bw, bh, contour))
            
    return edged, candidates


# import cv2
# import imutils
# import numpy as np

# def get_plate_candidates(bfilter_img):
#     """
#     Uses Auto-Canny to adapt to different lighting conditions and 
#     expands the candidate pool to ensure small plates are not missed.
#     """
#     # 1. Auto-Canny: Dynamically calculate thresholds based on image median brightness
#     v = np.median(bfilter_img)
#     sigma = 0.33
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
    
#     edged = cv2.Canny(bfilter_img, lower, upper)
    
#     # Dilate the edges to connect any broken lines around the plate border
#     edged = cv2.dilate(edged, None, iterations=1)
    
#     keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = imutils.grab_contours(keypoints)
    
#     # 2. Keep the top 20 contours so we don't drop small plates
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    
#     candidates = []
    
#     for contour in contours:
#         # Use minAreaRect to calculate the true aspect ratio even if the plate is skewed
#         rect = cv2.minAreaRect(contour)
#         (w, h) = rect[1]
        
#         if w < h:
#             w, h = h, w
            
#         if h == 0: 
#             continue
            
#         aspect_ratio = w / float(h)
        
#         # Get the upright bounding box for cropping later
#         x, y, bw, bh = cv2.boundingRect(contour)
        
#         # 3. Highly forgiving constraints: Catch anything remotely rectangular
#         if 1.2 <= aspect_ratio <= 7.0 and bw > 40 and bh > 12:
#             candidates.append((x, y, bw, bh, contour))
            
#     return edged, candidates


# import cv2
# import imutils

# def get_plate_candidates(bfilter_img):
#     """
#     Finds the top 5 rectangular candidates in the image. 
#     It intentionally uses loose constraints to catch skewed plates.
#     """
#     edged = cv2.Canny(bfilter_img, 30, 200)
    
#     keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = imutils.grab_contours(keypoints)
    
#     # Sort contours by area, keep top 20 to analyze
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    
#     candidates = []
    
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = w / float(h)
        
#         # Very loose constraints: just looking for rectangles that aren't tiny
#         if 1.2 < aspect_ratio < 6.0 and w > 60 and h > 15:
#             # Store the coordinates and the contour shape
#             candidates.append((x, y, w, h, contour))
            
#             # Stop once we have our top 5 suspects to keep processing fast
#             if len(candidates) == 5:
#                 break
                
#     return edged, candidates